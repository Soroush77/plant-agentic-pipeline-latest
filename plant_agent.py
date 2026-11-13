"""Agentic plant pest & disease diagnosis: one vision model + tools, over OpenRouter.

  python plant_agent.py diagnose photo.jpg [more.jpg] -q "what is wrong?" [--out trace.json]
  python plant_agent.py selftest          # offline check, no API key needed

Key: put OPENROUTER_API_KEY=... in a `.env` file next to this script (or in the environment).
Model: PLANT_AGENT_MODEL (default below). Paths: see CONFIG.
"""
from __future__ import annotations

import argparse
import base64
import difflib
import glob
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parent
if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

PIPE = Path(os.environ.get("PLANT_PIPELINE_ROOT", r"D:\workspace-plant-prod\pipeline-agentic-upgraded\plant-agentic-pipeline-full"))
CHUNKS_DIR = Path(os.environ.get("PLANT_CHUNKS_DIR", PIPE / "dataset_audit_preprocessing_fixes" / "output" / "chapters_chunks_fixed"))
BOOK_FILES = Path(os.environ.get("PLANT_BOOK_FILES", r"D:\workspace-plant-prod\dataset-collection-epub\data\cranshaw\files"))  # contains OEBPS/images
INAT_ROOT = Path(os.environ.get("PLANT_INAT_ROOT", PIPE))  # contains inaturalist_images/
MODEL = os.environ.get("PLANT_AGENT_MODEL", "anthropic/claude-opus-5.5")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Disease / disorder reference sets: (name, folder, verified labels?).
# Every sub-folder of ./datasets is added as a verified set (drop PlantVillage, PlantDoc, ... there).
DISEASE_SETS = [("houseplant_web", Path(r"D:\Plant_Disease_Database"), False)]
if (ROOT / "datasets").is_dir():
    DISEASE_SETS += [(p.name, p, True) for p in sorted((ROOT / "datasets").iterdir()) if p.is_dir()]

# Merged public datasets (PlantVillage, PlantDoc, FieldPlant, IP102, PlantWild, PlantSeg, ...): one spreadsheet of labels
# plus a folder of hash-named images. The spreadsheet is condensed once into papers_index.csv next to this script.
PAPERS_XLSX = Path(os.environ.get("PLANT_PAPERS_XLSX", r"D:\workspace-plant-prod\dataset-papers\merged_data_filled_with_unknown.xlsx"))
PAPERS_IMG_DIRS = [Path(p) for p in os.environ.get("PLANT_PAPERS_IMG_DIRS", os.pathsep.join([
    r"D:\workspace-plant-prod\dataset-papers\dataset_all_hashed",
    r"D:\workspace-plant-prod\dataset-papers\data_parts\dataset_all_hashed\dataset_all_hashed",
    r"D:\workspace-plant-prod\dataset-papers\data_parts\dataset_all_hashed"])).split(os.pathsep)]
PAPERS_INDEX = ROOT / "papers_index.csv"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_DIRS = {"train", "test", "val", "valid", "validation", "color", "images", "raw", "segmented", "grayscale"}
STOP = {"plant", "plants", "tree", "spp", "sp", "the", "of", "and", "a", "an", "on", "in", "with", "leaf", "leaves"}


def tok(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 1]


def data_url(path: Path | str, max_side: int = 768) -> str:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- KNOWLEDGE
class PestKB:
    """Cranshaw & Shetlar chunks + their book / iNaturalist photos, searched with BM25."""

    def __init__(self, chunks_dir: Path = CHUNKS_DIR):
        self.chunks: dict[str, dict] = {}
        for f in sorted(glob.glob(str(chunks_dir / "upgraded_chunks_*.json"))):
            for c in json.load(open(f, encoding="utf-8")):
                self.chunks[c["chunk_id"]] = c
        if not self.chunks:
            raise FileNotFoundError(f"no upgraded_chunks_*.json in {chunks_dir}")
        self.children = defaultdict(list)
        for cid, c in self.chunks.items():
            if c.get("parent_id") in self.chunks:
                self.children[c["parent_id"]].append(cid)
        self.vocab = {k: sorted({t for c in self.chunks.values() for t in (c.get("ontology_tags") or {}).get(k, [])})
                      for k in ("affected_plant_parts", "injury_symptoms", "diagnostic_signs")}
        # ponytail: BM25 only (847 docs). Add dense / BioCLIP image search from the existing repo if eval shows recall is the bottleneck.
        self.ids = list(self.chunks)
        docs = []
        for c in self.chunks.values():
            terms = " ".join(t.get("term", "") if isinstance(t, dict) else str(t) for t in c.get("index_terms") or [])
            docs.append(tok(c.get("search_text") or c["text"]) + tok(c["heading_text"]) * 2 + tok(terms))
        self.tf = [Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        self.avg = sum(self.dl) / len(self.dl)
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 6) -> list[tuple[str, float]]:
        q = [t for t in tok(query) if t in self.idf]
        scores = []
        for i, tf in enumerate(self.tf):
            s = sum(self.idf[t] * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * self.dl[i] / self.avg)) for t in q if t in tf)
            if s:
                scores.append((self.ids[i], s))
        return sorted(scores, key=lambda x: -x[1])[:k]

    def photos(self, cid: str, sources=("book", "inat")) -> list[tuple[Path, str]]:
        c, out = self.chunks[cid], []
        if "book" in sources:
            out += [(BOOK_FILES / im["src"], f"[field guide] {im.get('caption', '')}") for im in c.get("images") or []]
        if "inat" in sources:
            out += [(INAT_ROOT / im["src"], f"[iNaturalist] {im.get('caption', '').split(' Source:')[0]}") for im in c.get("inaturalist_images") or []]
        return [(p, cap) for p, cap in out if p.exists()]


def parse_class(root: Path, class_dir: Path) -> tuple[str, str]:
    """Folder -> (plant, condition). Handles Plant/Condition/, Plant___Condition/, and 'Plant condition' names."""
    name = class_dir.name
    if "___" in name:
        plant, cond = name.split("___", 1)
    elif class_dir.parent != root and class_dir.parent.name.lower() not in SPLIT_DIRS:
        plant, cond = class_dir.parent.name, name
    else:
        # ponytail: first word = plant; wrong for two-word plants ("Bell pepper"). Add a per-dataset mapping when one needs it.
        words = re.split(r"[\s_]+", name.strip())
        plant, cond = words[0], " ".join(words[1:]) or "unknown"
    clean = lambda s: re.sub(r"\s+", " ", s.replace("_", " ")).strip()
    return clean(plant), re.sub(r"\s+leaf$", "", clean(cond), flags=re.I) or clean(cond)


PLANT_ALIAS = {"corn (maize)": "corn", "maize": "corn", "pepper, bell": "bell pepper", "bell": "bell pepper", "soyabean": "soybean",
               "grapevine": "grape", "cherry (including sour)": "cherry"}
# Spelling variants of one label across the merged datasets. Synonyms (stripe rust = yellow rust) are left alone.
LABEL_ALIAS = {"becterial blight": "bacterial blight", "brownspot": "brown spot", "brown spots": "brown spot", "blight leaf": "leaf blight",
               "rust leaf": "leaf rust", "frogeye leaf spot": "frog eye leaf spot", "aphids": "aphid", "thirps": "thrips", "redrot": "red rot",
               "redrust": "red rust", "bollrot": "boll rot", "mealy bug": "mealybug", "mold": "leaf mold",
               "pepper leaf": "healthy",  # PlantDoc's "Bell_pepper leaf" class is the healthy leaf
               "haunglongbing (citrus greening)": "huanglongbing (citrus greening)", "black leaf streak black sigatoka": "black leaf streak",
               "cercospora leaf spot gray leaf spot": "gray leaf spot", "spider mites two-spotted spider mite": "two-spotted spider mite",
               "bacterial leaf streak black chaff black chaff": "bacterial leaf streak black chaff",
               "bacterial leaf streak black chaffblack chaff": "bacterial leaf streak black chaff",
               "septoria blotch blotch": "septoria blotch", "septoria blotchseptoria": "septoria blotch"}


def clean_plant(s) -> str:
    s = re.sub(r"\s+", " ", str(s).lower().replace("_", " ")).strip()
    return PLANT_ALIAS.get(s, s)


def clean_label(plant: str, s, keep_case: bool = False) -> str:
    """One spelling per label: drop crawl suffixes (' bing', ' - copy'), a leading plant name and a trailing ' leaf'."""
    s = re.sub(r"\s+", " ", str(s).replace("_", " ")).strip()
    if keep_case:  # IP102 scientific names
        return s
    s = re.sub(r"( bing)?( - copy)?$", "", s.lower()).strip()
    s = LABEL_ALIAS.get(s, s)
    if s.startswith(plant + " "):
        s = re.sub(r"(?<=\w) leaf$", "", s[len(plant) + 1:])
    return LABEL_ALIAS.get(s, s)


def load_papers_index() -> list[list[str]]:
    """Rows [image, dataset, plant, label, pool, wild] of the merged datasets; built from the spreadsheet on first use (~30 s)."""
    import csv
    if not PAPERS_INDEX.exists() or (PAPERS_XLSX.exists() and PAPERS_XLSX.stat().st_mtime > PAPERS_INDEX.stat().st_mtime):
        if not PAPERS_XLSX.exists():
            return []
        import openpyxl
        rows = openpyxl.load_workbook(PAPERS_XLSX, read_only=True).active.iter_rows(values_only=True)
        col = {str(h): i for i, h in enumerate(next(rows))}
        labels, recs = defaultdict(set), []
        for r in rows:
            iid, ds = r[col["id"]], str(r[col["original_dataset_name"]])
            if not iid:
                continue
            plant = clean_plant(r[col["plant_name"]])
            label = clean_label(plant, r[col["disease_name"]], keep_case=ds == "IP102")
            if (plant, label) not in labels[iid]:
                labels[iid].add((plant, label))
                wild = int(r[col["in_wild"]] or 0) or int(ds == "PlantWild")  # the sheet marks PlantWild as lab; it is an in-the-wild set
                recs.append([iid, ds, plant, label, str(r[col["train_or_validation_or_test"]]), wild])
        has_eval = {ds for _, ds, _, _, split, _ in recs if split != "train"}
        for rec in recs:  # pool: "eval" = held out for evaluation, never shown as a reference
            held = rec[4] != "train" if rec[1] in has_eval else int(rec[0][:2], 16) < 26  # train-only sets: hold out ~10% by hash
            rec[4] = "eval" if held and len(labels[rec[0]]) == 1 else "ref"  # multi-label images stay references
        with open(PAPERS_INDEX, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(recs)
    with open(PAPERS_INDEX, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def resolve_img(ref) -> Path | None:
    """Reference -> file on disk. Hash-named dataset images are looked up lazily (the archive may still be extracting)."""
    if isinstance(ref, Path):
        return ref if ref.exists() else None
    return next((d / ref for d in PAPERS_IMG_DIRS if (d / ref).exists()), None)


class DiseaseKB:
    """Labelled reference images for diseases, disorders and crop pests: folder-per-class sets plus the merged public datasets."""

    def __init__(self, sets=DISEASE_SETS, index: bool = True):
        # (plant, condition) -> [(image ref, verified label?, set name, in the wild?, pool)]
        self.items: dict[tuple[str, str], list[tuple]] = defaultdict(list)
        for name, root, verified in sets:
            root = Path(root)
            if not root.is_dir():
                continue
            for d, _, files in os.walk(root):
                imgs = [Path(d) / f for f in sorted(files) if Path(f).suffix.lower() in IMG_EXT]
                if imgs and Path(d) != root:
                    self.items[parse_class(root, Path(d))] += [(p, verified, name, True, "ref") for p in imgs]
        for iid, ds, plant, label, pool, wild in load_papers_index() if index else []:
            self.items[(plant, label)].append((iid, True, ds, wild == "1", pool))
        self.plants = sorted({p for p, _ in self.items if p != "unknown"})  # "unknown" = host not recorded (IP102 pests)
        self.class_tokens = {k: set(tok(f"{k[0]} {k[1]}")) - STOP for k in self.items}

    def match_plants(self, query: str) -> list[str]:
        q = set(tok(query)) - STOP or set(tok(query))
        scored = [(len(q & (set(tok(p)) - STOP)) / max(len(q), 1), p) for p in self.plants]
        best = max((s for s, _ in scored), default=0)
        if best > 0:
            return [p for s, p in scored if s == best][:5]
        return difflib.get_close_matches(query, self.plants, n=3, cutoff=0.6)

    def conditions(self, plant: str) -> dict[str, dict]:
        out = {}
        for (p, cond), imgs in self.items.items():
            if p == plant:
                out[cond] = {"photos": len(imgs), "sets": sorted({i[2] for i in imgs}),
                             "labels": "verified" if all(i[1] for i in imgs) else "unverified web images"}
        return out

    def find(self, query: str, k: int = 12) -> list[tuple[str, str]]:
        """Classes whose plant + condition name best match the words of the query (for pests with no recorded host, synonyms, ...)."""
        q = set(tok(query)) - STOP
        scored = [(len(q & t) / max(len(q), 1), len(self.items[key]), key) for key, t in self.class_tokens.items() if q & t]
        return [key for _, _, key in sorted(scored, reverse=True)[:k]]

    def photos(self, plant: str, condition: str) -> list[tuple]:
        """(image ref, caption) of the reference pool, in-the-wild photos first; resolve with resolve_img()."""
        conds = self.conditions(plant)
        hit = next((c for c in conds if c.lower() == condition.lower()), None) or next(
            iter(difflib.get_close_matches(condition, list(conds), n=1, cutoff=0.6)), None)
        if not hit:
            return []
        host = "" if plant == "unknown" else f"{plant}: "
        refs = sorted((i for i in self.items[(plant, hit)] if i[4] == "ref"), key=lambda i: (not i[3], str(i[0])))
        return [(ref, f"[{name}{'' if v else ', unverified label'}] {host}{hit}") for ref, v, name, _, _ in refs]


# ----------------------------------------------------------------------------- TOOLS
def _fn(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}


SUBMIT = _fn("submit_diagnosis", "Finish: give the final assessment. Call exactly once, when done.", {
    "host_plant": {"type": "string", "description": "Host plant, English common name (scientific name if sure), or 'unknown'."},
    "category": {"type": "string", "enum": ["healthy", "pest", "disease", "abiotic", "uncertain"]},
    "name": {"type": "string", "description": "English name of the problem (e.g. 'twospotted spider mite', 'powdery mildew', 'overwatering'); 'healthy' if none."},
    "scientific_name": {"type": "string", "description": "Scientific name of the organism, if any."},
    "book_entry_id": {"type": "string", "description": "entry_id from the pest field guide, when the diagnosis corresponds to one."},
    "condition_label": {"type": "string", "description": "Exact condition label from list_conditions, when the diagnosis corresponds to one."},
    "confidence": {"type": "number", "description": "0-1, calibrated."},
    "evidence": {"type": "array", "items": {"type": "string"}, "description": "Specific visible features supporting the diagnosis, and which references they matched."},
    "differentials": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "book_entry_id": {"type": "string"}, "condition_label": {"type": "string"},
        "why_less_likely": {"type": "string"}}, "required": ["name", "why_less_likely"]},
        "description": "Other plausible causes, most likely first."},
    "next_checks": {"type": "array", "items": {"type": "string"}, "description": "Observations that would confirm or rule out (e.g. check leaf underside, roots, watering history)."},
    "actions": {"type": "array", "items": {"type": "string"}, "description": "What to do about it, least drastic first."},
    "summary": {"type": "string", "description": "Short answer for the user, in the user's language."},
}, ["host_plant", "category", "name", "confidence", "evidence", "differentials", "next_checks", "actions", "summary"])


def make_tools(pest: PestKB) -> list[dict]:
    v = pest.vocab
    return [
        _fn("search_pest_book", "Keyword search of the insect/mite field guide (Cranshaw & Shetlar, Garden Insects of North America). "
            "Works best with host plant + the guide's own terms + candidate names. "
            f"Injury terms: {', '.join(v['injury_symptoms'])}. Sign terms: {', '.join(v['diagnostic_signs'])}. "
            f"Plant-part terms: {', '.join(v['affected_plant_parts'])}.",
            {"query": {"type": "string"}, "top_k": {"type": "integer", "description": "default 6, max 12"}}, ["query"]),
        _fn("read_pest_entry", "Full text of one field-guide entry (hosts, damage, appearance, life history), with its parent group and child entries.",
            {"entry_id": {"type": "string"}}, ["entry_id"]),
        _fn("view_pest_photos", "Show expert-captioned reference photos of one field-guide entry, to compare side by side with the user's photo.",
            {"entry_id": {"type": "string"}, "max_photos": {"type": "integer", "description": "1-6, default 4"},
             "skip": {"type": "integer", "description": "skip the first N photos to see more"}}, ["entry_id"]),
        _fn("list_conditions", "List the conditions that have labelled reference photos for a host plant: diseases, disorders, pests, "
            "and 'healthy' where available (with photo counts, source datasets, and whether labels are verified).",
            {"plant": {"type": "string"}}, ["plant"]),
        _fn("find_reference_classes", "Search all labelled reference classes by name, across plants. Use it for a suspected problem "
            "whatever the host (e.g. 'fall armyworm', 'powdery mildew'), and for crop pests whose host was not recorded (plant = 'unknown').",
            {"query": {"type": "string"}}, ["query"]),
        _fn("view_condition_photos", "Show labelled reference photos of one condition on one host plant (field photos first). "
            "Comparing with the 'healthy' class of the same plant is often the quickest way to tell whether anything is wrong.",
            {"plant": {"type": "string", "description": "plant name exactly as returned by list_conditions / find_reference_classes ('unknown' for host-less classes)"},
             "condition": {"type": "string"}, "max_photos": {"type": "integer", "description": "1-5, default 3"}}, ["plant", "condition"]),
        _fn("ask_user", "Ask the user one question (another view of the plant, watering, light, recent changes) when the answer would change the diagnosis.",
            {"question": {"type": "string"}}, ["question"]),
        SUBMIT,
    ]


class Session:
    """Tool executor for one diagnosis. Holds what was looked at, and the final answer."""

    def __init__(self, pest: PestKB, disease: DiseaseKB, ask=None, exclude=(), ref_sources=("book", "inat"), verified_only=False, on_tool=None):
        self.pest, self.disease, self.ask, self.on_tool = pest, disease, ask, on_tool  # on_tool(trace_entry): progress hook for UIs
        self.exclude = {Path(p).resolve() for p in exclude}  # never show these as references (eval leakage guard)
        self.ref_sources, self.verified_only = ref_sources, verified_only
        self.viewed, self.gate_used, self.final, self.trace = [], False, None, []

    def _pick(self, photos, n, skip=0):
        out = []
        for ref, cap in photos:  # stop as soon as enough are found: a class can hold thousands of candidates
            p = None if self.verified_only and "unverified" in cap else resolve_img(ref)
            if p is not None and p.resolve() not in self.exclude:
                out.append((p, cap))
                if len(out) >= skip + n:
                    break
        return out[skip:], len(photos)

    def call(self, name: str, a: dict) -> tuple[str, list[tuple[Path, str]]]:
        try:
            text, imgs = getattr(self, "t_" + name)(a)
        except AttributeError:
            text, imgs = f"Unknown tool {name!r}.", []
        except (KeyError, TypeError, ValueError) as e:
            text, imgs = f"Bad arguments for {name}: {e!r}", []
        self.log({"tool": name, "args": a, "result": text[:600], "photos": [str(p) for p, _ in imgs], "captions": [c for _, c in imgs]})
        return text, imgs

    def log(self, entry: dict) -> None:
        self.trace.append(entry)
        if self.on_tool:
            self.on_tool(entry)

    def t_search_pest_book(self, a):
        hits = self.pest.search(a["query"], min(int(a.get("top_k") or 6), 12))
        out = []
        for cid, score in hits:
            c = self.pest.chunks[cid]
            tags = c.get("ontology_tags") or {}
            out.append({"entry_id": cid, "title": c["heading_text"], "group": (c.get("parent") or {}).get("label", ""),
                        "taxonomy": c.get("taxonomy"), "injury": tags.get("injury_symptoms"), "signs": tags.get("diagnostic_signs"),
                        "excerpt": c["text"][:350], "photos": len(self.pest.photos(cid, self.ref_sources)), "score": round(score, 1)})
        return json.dumps(out, ensure_ascii=False) if out else "No entries matched. Try other terms.", []

    def t_read_pest_entry(self, a):
        c = self.pest.chunks[a["entry_id"]]
        par = self.pest.chunks.get(c.get("parent_id"))
        out = {"entry_id": c["chunk_id"], "title": c["heading_text"], "taxonomy": c.get("taxonomy"), "text": c["text"][:6000],
               "parent": par and {"entry_id": par["chunk_id"], "title": par["heading_text"], "text": par["text"][:1500]},
               "children": [{"entry_id": k, "title": self.pest.chunks[k]["heading_text"]} for k in self.pest.children.get(c["chunk_id"], [])],
               "photo_captions": [cap for _, cap in self.pest.photos(c["chunk_id"], self.ref_sources)][:12]}
        return json.dumps(out, ensure_ascii=False), []

    def t_view_pest_photos(self, a):
        cid = a["entry_id"]
        imgs, total = self._pick(self.pest.photos(cid, self.ref_sources), max(1, min(int(a.get("max_photos") or 4), 6)), int(a.get("skip") or 0))
        if not imgs:
            return "No reference photos for this entry.", []
        self.viewed.append(cid)
        return f"{len(imgs)} of {total} reference photos for {self.pest.chunks[cid]['heading_text']!r} follow in the next message.", imgs

    def t_list_conditions(self, a):
        plants = self.disease.match_plants(a["plant"])
        if not plants:
            return ("No reference set for this plant; try find_reference_classes, or rely on your own knowledge and say that no "
                    f"references were available. Plants with references: {', '.join(self.disease.plants)}"), []
        out = {p: self.disease.conditions(p) for p in plants}
        return json.dumps(out, ensure_ascii=False) + "\nThis list is only what has reference photos; the true cause may be absent from it.", []

    def t_find_reference_classes(self, a):
        out = [{"plant": p, "condition": c, **self.disease.conditions(p)[c]} for p, c in self.disease.find(a["query"])]
        return json.dumps(out, ensure_ascii=False) if out else "No labelled class matched. Try other words.", []

    def t_view_condition_photos(self, a):
        plants = ["unknown"] if a["plant"].strip().lower() in ("", "unknown") else self.disease.match_plants(a["plant"])
        photos = self.disease.photos(plants[0], a["condition"]) if plants else []
        imgs, total = self._pick(photos, max(1, min(int(a.get("max_photos") or 3), 5)))
        if not imgs:
            return "No reference photos on disk for that plant / condition. Use list_conditions or find_reference_classes for exact names.", []
        self.viewed.append(f"{plants[0]}|{a['condition']}")
        return f"{len(imgs)} of {total} reference photos follow in the next message.", imgs

    def t_ask_user(self, a):
        if not self.ask:
            return "The user cannot be reached. Decide with what you have, and put the question into next_checks.", []
        return f"User answered: {self.ask(a['question'])}", []

    def t_submit_diagnosis(self, a):
        missing = [k for k in SUBMIT["function"]["parameters"]["required"] if k not in a]
        if missing:
            return f"Not accepted: missing fields {missing}.", []
        # Verification gate: a confident pest/disease call must have been compared with reference photos (once).
        if not self.gate_used and a["category"] in ("pest", "disease") and float(a["confidence"]) >= 0.6 and not self.viewed:
            self.gate_used = True
            return ("Not accepted yet: no reference photos were compared with the user's photo. View references for your leading "
                    "candidate first, or resubmit with lower confidence if none exist."), []
        self.final = a
        return "Accepted.", []


# ----------------------------------------------------------------------------- AGENT LOOP
SYSTEM = """You diagnose plant health problems from photos, for crop growers and houseplant owners. People act on your answer \
(they spray, repot, or throw a plant away), so a wrong confident answer costs more than an honest "can't tell from this photo".

What you have:
- A field guide to insects and mites (Cranshaw & Shetlar) with expert-captioned photos plus research-grade iNaturalist photos. \
It is reliable for arthropods, and it covers North American garden pests only: it says nothing about fungal, bacterial, viral or care problems.
- Labelled reference photos per host plant, from published datasets (PlantVillage, PlantDoc, PlantWild, PlantSeg, FieldPlant, IP102 and \
others): crop diseases, crop pests, and healthy plants. These labels are verified. A separate houseplant set is marked unverified: web \
images, useful to see what a condition can look like, weak as proof.
- Your own knowledge, which you should use, and say so when no reference backed it.

How to work:
1. Look first. Note the host, the affected part, the symptoms, any signs (webbing, wax, frass, honeydew, spores, ooze) and any organism visible.
2. Decide which kind of cause fits: arthropod pest, pathogen, abiotic / care problem, or nothing wrong. Many photos show healthy plants: \
variegation, natural red or purple pigment, and the yellowing of one old lower leaf are normal. Do not reach for a pest because the guide is about pests.
3. Get candidates from the tools, then compare the user's photo side by side with reference photos of the two or three closest candidates. \
Name the specific features that match or do not. This comparison is what makes the answer trustworthy.
4. One photo often cannot separate causes (wilting and yellowing fit root rot, overwatering and drought alike). Then give ranked \
differentials and say which observation would separate them. If a user is available and one answer would change the diagnosis, ask.
5. Confidence is high only when specific features matched references. Use "uncertain" when the evidence does not single out a cause.

Finish by calling submit_diagnosis once. Write summary, evidence, next_checks and actions in {lang}; keep names and labels in English. \
You have about {budget} tool rounds. After a diagnosis, answer follow-up questions in plain text in {lang}; submit again only if new \
evidence changes the diagnosis."""


def chat(messages: list[dict], tools: list[dict], model: str = MODEL) -> tuple[dict, dict]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in plant_agent/.env).")
    last = ""
    for attempt in range(4):
        try:
            r = httpx.post(API_URL, headers={"Authorization": f"Bearer {key}"}, timeout=240,
                           json={"model": model, "messages": messages, "tools": tools, "max_tokens": 4000})
        except httpx.TransportError as e:
            last = repr(e)
        else:
            if r.status_code < 400 and "choices" in (data := r.json()):
                return data["choices"][0]["message"], data.get("usage") or {}
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code not in (408, 429, 500, 502, 503, 504, 200):
                break
        time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"OpenRouter call failed: {last}")


NO_DIAGNOSIS = {"host_plant": "unknown", "category": "uncertain", "name": "no diagnosis", "confidence": 0.0, "evidence": [],
                "differentials": [], "next_checks": [], "actions": [], "summary": "The model did not submit a diagnosis."}


def _refs(shown: list[tuple[Path, str]]) -> list[dict]:
    """Reference photos as user-turn content (tool results cannot carry images on every provider)."""
    parts = [{"type": "text", "text": "Reference photos you asked for, in order:"}] if shown else []
    for i, (p, cap) in enumerate(shown, 1):
        parts += [{"type": "text", "text": f"Reference {i}: {cap}"}, {"type": "image_url", "image_url": {"url": data_url(p)}}]
    return parts


class Conversation:
    """One diagnosis chat. send() runs the tool loop until a diagnosis, a question for the user (pause_on_ask), or a plain answer."""

    def __init__(self, session: Session, llm=None, max_steps: int = 10, lang: str = "Persian", use_tools: bool = True, pause_on_ask: bool = False):
        self.s, self.llm, self.max_steps, self.pause = session, llm or chat, max_steps, pause_on_ask
        self.tools = make_tools(session.pest) if use_tools else [SUBMIT]
        if not use_tools:
            session.gate_used = True  # no references available to the no-tools baseline
        self.messages = [{"role": "system", "content": SYSTEM.format(lang=lang, budget=max_steps - 1)}]
        self.usage, self.diagnosed = Counter(), False
        self.pending, self.held = None, []  # id of an ask_user call waiting for its answer; reference photos held until then

    def send(self, text: str = "", images=()) -> dict:
        photos = [{"type": "image_url", "image_url": {"url": data_url(p, 1024)}} for p in images]
        if self.pending:  # the user's message is the answer to ask_user: tool result first, then held references and new photos
            self.messages.append({"role": "tool", "tool_call_id": self.pending, "content": f"User answered: {text or '(no text)'}"})
            parts = _refs(self.held) + ([{"type": "text", "text": f"New photos from the user: {len(photos)}"}] + photos if photos else [])
            self.pending, self.held = None, []
            if parts:
                self.messages.append({"role": "user", "content": parts})
        else:
            self.s.final = None
            self.messages.append({"role": "user", "content": [
                {"type": "text", "text": f"User's message: {text or '(none given)'}\nUser's photos: {len(photos)}"}] + photos})
        tools = self.tools
        for step in range(self.max_steps):
            if step == self.max_steps - 1:  # out of budget: only the final tool stays (no forced tool_choice: several models reject it)
                tools = [SUBMIT]
                self.messages.append({"role": "user", "content": "The tool budget is used up. Call submit_diagnosis now with your best assessment."})
            msg, u = self.llm(self.messages, tools)
            self.usage.update({k: v for k, v in u.items() if isinstance(v, (int, float))})
            self.messages.append({k: v for k, v in msg.items() if v is not None})
            calls = msg.get("tool_calls") or []
            if not calls:
                if self.diagnosed and msg.get("content"):  # follow-up question after a diagnosis: plain answer
                    return self._out(step, text=msg["content"])
                self.messages.append({"role": "user", "content": "Continue with a tool, or call submit_diagnosis to finish."})
                continue
            shown, question = [], None
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = None
                if self.pause and c["function"]["name"] == "ask_user" and isinstance(args, dict) and question is None:
                    self.pending, question = c["id"], str(args.get("question", ""))  # answered by the next send()
                    self.s.log({"tool": "ask_user", "args": args, "result": "(waiting for the user)", "photos": [], "captions": []})
                    continue
                text_out, imgs = self.s.call(c["function"]["name"], args) if isinstance(args, dict) else ("Arguments were not valid JSON.", [])
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": text_out})
                shown += imgs
            if self.s.final:
                if self.pending:  # asked and submitted in one turn: close the question so the history stays valid
                    self.messages.append({"role": "tool", "tool_call_id": self.pending, "content": "Not asked: a diagnosis was submitted."})
                    self.pending = None
                self.diagnosed = True
                return self._out(step, diagnosis=self.s.final)
            if question is not None:
                self.held = shown
                return self._out(step, question=question)
            if shown:
                self.messages.append({"role": "user", "content": _refs(shown)})
        return self._out(step, diagnosis=dict(NO_DIAGNOSIS))

    def _out(self, step: int, **kw) -> dict:
        return {"diagnosis": None, "question": None, "text": None, "trace": self.s.trace, "usage": dict(self.usage),
                "steps": step + 1, "model": MODEL, **kw}


def diagnose(images: list[Path | str], question: str = "", *, session: Session, llm=None, max_steps: int = 10,
             lang: str = "Persian", use_tools: bool = True) -> dict:
    """One-shot diagnosis (CLI, eval): always returns a diagnosis."""
    return Conversation(session, llm, max_steps, lang, use_tools).send(question, images)


# ----------------------------------------------------------------------------- CLI
def show(d: dict) -> None:
    print(f"\n=== {d['category'].upper()}: {d['name']}" + (f" ({d['scientific_name']})" if d.get("scientific_name") else "")
          + f"   host: {d['host_plant']}   confidence: {float(d['confidence']):.0%}")
    print(d["summary"])
    for title, key in (("Evidence", "evidence"), ("Check next", "next_checks"), ("Actions", "actions")):
        if d.get(key):
            print(f"\n{title}:")
            print("\n".join(f"  - {x}" for x in d[key]))
    if d.get("differentials"):
        print("\nAlso possible:")
        print("\n".join(f"  - {x['name']}: {x['why_less_likely']}" for x in d["differentials"]))


def selftest() -> None:
    """Offline check of everything except the model: search, datasets, tool plumbing, verification gate, message format."""
    pest, dis = PestKB(), DiseaseKB()
    for q, want in [("fall webworm silken tent", "WEBWORM"), ("japanese beetle skeletonizing rose", "JAPANESE BEETLE"),
                    ("twospotted spider mite stippling fine webbing", "SPIDER MITE"), ("rose aphid honeydew leaf curling", "APHID")]:
        top = [pest.chunks[c]["heading_text"].upper() for c, _ in pest.search(q, 3)]
        assert any(want in t for t in top), (q, top)
    assert parse_class(Path("r"), Path("r/color/Tomato___Early_blight")) == ("Tomato", "Early blight")
    assert parse_class(Path("r"), Path("r/train/Corn_Gray_leaf_spot")) == ("Corn", "Gray leaf spot")
    assert parse_class(Path("r"), Path("r/Orchid/Black Rot")) == ("Orchid", "Black Rot")
    assert clean_label("tomato", "Tomato Early blight leaf") == "early blight" and clean_label("corn", "common_rust") == "common rust"
    assert clean_label("apple", "black rot bing  - copy") == "black rot" and clean_label("wheat", "septoria blotchseptoria") == "septoria blotch"
    assert clean_label("tomato", "Tomato mold leaf") == "leaf mold" and clean_plant("corn (maize)") == "corn"
    if PAPERS_INDEX.exists():  # merged public datasets: labels merged across sets, held-out images never offered as references
        assert {"healthy", "early blight", "late blight"} <= set(dis.conditions("tomato")), sorted(dis.conditions("tomato"))
        assert ("unknown", "rice leaf roller") in dis.find("rice leaf roller"), dis.find("rice leaf roller")
        held = {i[0] for i in dis.items[("tomato", "early blight")] if i[4] == "eval"}
        assert held and not held & {ref for ref, _ in dis.photos("tomato", "early blight")}
    if any(p.is_dir() for _, p, _ in DISEASE_SETS):
        assert "Rubber Plant (Ficus elastica)" in dis.match_plants("ficus elastica"), dis.match_plants("ficus elastica")
        assert any(p.startswith("Pothos") for p in dis.match_plants("pothos"))

    photo = next(p for c in pest.chunks for p, _ in pest.photos(c, ("book",)))
    sess = Session(pest, dis, exclude=[photo])
    final_args = {"host_plant": "rose", "category": "pest", "name": "rose aphid", "confidence": 0.9, "evidence": ["x"],
                  "differentials": [], "next_checks": [], "actions": [], "summary": "s"}
    script, state = iter(range(5)), {}

    def fake_llm(messages, tools):
        i = next(script)
        call = lambda n, a: {"id": f"c{i}{n}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        if i == 0:
            return {"role": "assistant", "content": None, "tool_calls": [call("search_pest_book", {"query": "rose aphid honeydew"}),
                                                                         call("list_conditions", {"plant": "pothos"})]}, {"total_tokens": 1}
        if i == 1:
            state["id"] = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))[0]["entry_id"]
            return {"role": "assistant", "content": None, "tool_calls": [call("read_pest_entry", {"entry_id": state["id"]})]}, {}
        if i == 2:  # confident answer with no references viewed -> the gate must refuse it
            return {"role": "assistant", "content": None, "tool_calls": [call("submit_diagnosis", final_args)]}, {}
        if i == 3:
            assert "Not accepted yet" in messages[-1]["content"], messages[-1]
            return {"role": "assistant", "content": None, "tool_calls": [call("view_pest_photos", {"entry_id": state["id"], "max_photos": 2}),
                                                                         call("nonexistent_tool", {})]}, {}
        refs = messages[-1]["content"]
        assert messages[-1]["role"] == "user" and sum(p["type"] == "image_url" for p in refs) == 2, "reference photos not delivered"
        assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in refs if p["type"] == "image_url")
        return {"role": "assistant", "content": None, "tool_calls": [call("submit_diagnosis", dict(final_args, book_entry_id=state["id"]))]}, {}

    out = diagnose([photo], "test", session=sess, llm=fake_llm)
    assert out["diagnosis"]["book_entry_id"] == state["id"] and out["steps"] == 5, out
    assert str(Path(photo)) not in [p for t in out["trace"] for p in t["photos"]], "excluded photo leaked into references"

    # Chat mode: ask_user pauses the run; the next send() is the answer and also delivers the held reference photos.
    turn = iter(range(2))

    def chat_llm(messages, tools):
        i = next(turn)
        call = lambda n, a: {"id": f"q{i}{n}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        if i == 0:
            return {"role": "assistant", "tool_calls": [call("view_pest_photos", {"entry_id": state["id"], "max_photos": 1}),
                                                        call("ask_user", {"question": "underside?"})]}, {}
        assert messages[-2] == {"role": "tool", "tool_call_id": "q0ask_user", "content": "User answered: mites"}, messages[-2]
        assert sum(p["type"] == "image_url" for p in messages[-1]["content"]) == 2, "expected 1 held reference + 1 new user photo"
        return {"role": "assistant", "tool_calls": [call("submit_diagnosis", final_args)]}, {}

    conv = Conversation(Session(pest, dis), chat_llm, pause_on_ask=True)
    assert conv.send("help", [photo])["question"] == "underside?"
    assert conv.send("mites", [photo])["diagnosis"]["name"] == "rose aphid"
    print(f"selftest ok: {len(pest.chunks)} pest entries; {len(dis.items)} labelled classes over {len(dis.plants)} plants, "
          f"{sum(len(v) for v in dis.items.values()):,} labelled photos from {sorted({i[2] for v in dis.items.values() for i in v})}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("images", nargs="+")
    d.add_argument("-q", "--question", default="")
    d.add_argument("--lang", default="Persian")
    d.add_argument("--no-ask", action="store_true", help="never ask the user follow-up questions")
    d.add_argument("--verified-only", action="store_true", help="hide unverified reference photos")
    d.add_argument("--out", help="write diagnosis + tool trace as JSON")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    ask = None if a.no_ask else (lambda q: input(f"\n[agent asks] {q}\n> "))
    sess = Session(PestKB(), DiseaseKB(), ask=ask, verified_only=a.verified_only)
    try:
        res = diagnose(a.images, a.question, session=sess, lang=a.lang)
    except RuntimeError as e:  # missing key / API failure: message, not traceback
        sys.exit(f"error: {e}")
    show(res["diagnosis"])
    print(f"\n[{res['steps']} steps, tools: {', '.join(t['tool'] for t in res['trace'])}; tokens: {res['usage'].get('total_tokens', '?')}]")
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
