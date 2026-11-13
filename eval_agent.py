"""Measure the agent on labelled images, next to a no-tools baseline of the same model.

  python eval_agent.py pest    --n 40 --baseline      # iNaturalist photos as queries, field-guide photos as references
  python eval_agent.py disease --n 40 --baseline      # held-out photos of the labelled sets, one per class
                                                      #   --set PlantDoc | FieldPlant | PlantWild | PlantSeg | PlantVillage | IP102 | MCCDI | NLB |
                                                      #         TomatoLeavesDataset | houseplant_web (unverified; only when named)
  python eval_agent.py healthy --dir ..\\healthy_frames --baseline   # false-alarm rate on healthy plants
  python eval_agent.py selftest                       # offline check, no API key needed

The query photo is never shown back as a reference. Results go to eval_results/*.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import plant_agent as pa

DROP = {"spp", "sp", "the", "of", "and"}


def stems(s: str) -> set[str]:
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in pa.tok(s)} - DROP


def name_hit(pred: str, names: list[str]) -> bool:
    """True if every word of some ground-truth name appears in the prediction (plural-insensitive).
    Long names also match with spacing ignored ('twospotted' = 'two-spotted'); short ones do not ('mite' is not in 'termite')."""
    p, glued = stems(pred), "".join(pa.tok(pred))
    return any(stems(n) and (stems(n) <= p or (len(g := "".join(pa.tok(n))) >= 12 and g in glued)) for n in names if n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# ----------------------------------------------------------------------------- cases
def pest_cases(pest: pa.PestKB, n: int, rng: random.Random) -> list[dict]:
    ok = [c for c in pest.chunks if pest.photos(c, ("book",)) and pest.photos(c, ("inat",))]
    out = []
    for cid in rng.sample(ok, min(n, len(ok))):
        c = pest.chunks[cid]
        im = rng.choice([i for i in c["inaturalist_images"] if (pa.INAT_ROOT / i["src"]).exists()])
        tx = im.get("resolved_taxon") or {}
        head = c["heading_text"]
        names = [head.split("(")[0], *re.findall(r"\(([^)]+)\)", head), tx.get("scientific_name", ""), tx.get("common_name", "")]
        out.append({"task": "pest", "image": str(pa.INAT_ROOT / im["src"]), "entry_id": cid, "names": [x for x in names if x],
                    "session": {"ref_sources": ("book",)}})  # references = field-guide photos only: no iNaturalist near-duplicates
    return out


def disease_cases(dis: pa.DiseaseKB, n: int, rng: random.Random, only_set: str | None) -> list[dict]:
    """One query photo per class: from the held-out pool of the merged datasets, or any photo of a folder set (leave-one-out)."""
    vocab = {}  # set name -> its label list, handed to the no-tools baseline so it is not penalised for wording
    for (_, cond), items in dis.items.items():
        for i in items:
            vocab.setdefault(i[2], set()).add(cond)
    out, classes = [], list(dis.items.items())
    rng.shuffle(classes)
    for (plant, cond), items in classes:
        items = [i for i in items if (i[2] == only_set if only_set else i[1])]  # default: sets with verified labels only
        cands = [i for i in items if i[4] == "eval"] or [i for i in items if isinstance(i[0], Path) and len(items) >= 4]
        rng.shuffle(cands)
        hit = next(((p, i) for i in cands[:30] if (p := pa.resolve_img(i[0]))), None)  # the archive may still be extracting
        if hit:
            labels = sorted(vocab[hit[1][2]])
            out.append({"task": "disease", "image": str(hit[0]), "plant": plant, "label": cond, "names": [cond], "set": hit[1][2],
                        "labels": labels if len(labels) <= 150 else None, "session": {}})
        if len(out) == n:
            break
    return out


def healthy_cases(folder: str, n: int, rng: random.Random) -> list[dict]:
    imgs = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in pa.IMG_EXT)
    return [{"task": "healthy", "image": str(p), "names": ["healthy"], "session": {}} for p in rng.sample(imgs, min(n, len(imgs)))]


# ----------------------------------------------------------------------------- scoring
def score(case: dict, d: dict, pest: pa.PestKB, dis: pa.DiseaseKB) -> dict:
    primary = " ".join(str(d.get(k, "")) for k in ("name", "scientific_name", "condition_label"))
    diffs = d.get("differentials") or []
    top3 = primary + " " + " ".join(f"{x.get('name', '')} {x.get('condition_label', '')}" for x in diffs[:2])
    s = {"name_hit": name_hit(primary, case["names"]), "name_hit_top3": name_hit(top3, case["names"])}
    if case["task"] == "pest":
        gt, pred = pest.chunks[case["entry_id"]], pest.chunks.get(d.get("book_entry_id") or "")
        s["entry_exact"] = bool(pred) and pred["chunk_id"] == gt["chunk_id"]
        s["entry_group"] = bool(pred) and (s["entry_exact"] or pred["chunk_id"] in (gt.get("parent_id"), gt.get("grandparent_id"))
                                           or (pred.get("parent_id") and pred.get("parent_id") in (gt.get("parent_id"), gt["chunk_id"])))
        s["correct"] = s["entry_exact"] or s["name_hit"]
        s["correct_top3"] = s["name_hit_top3"] or case["entry_id"] in [x.get("book_entry_id") for x in diffs[:2]] or s["correct"]
    elif case["task"] == "disease":
        s["label_exact"] = stems(d.get("condition_label") or "") == stems(case["label"])
        if case["plant"] != "unknown":  # IP102 pests have no recorded host
            s["host_hit"] = case["plant"] in dis.match_plants(d.get("host_plant") or "")
        if case["label"] == "healthy":  # a healthy plant is right only if nothing is diagnosed
            s["correct"] = s["correct_top3"] = d.get("category") == "healthy"
        else:
            s["correct"] = s["label_exact"] or s["name_hit"]
            s["correct_top3"] = s["correct"] or s["name_hit_top3"]
    else:
        s["correct"] = s["correct_top3"] = d.get("category") == "healthy"
    return s


def run_case(case: dict, mode: str, pest: pa.PestKB, dis: pa.DiseaseKB, llm=pa.chat) -> dict:
    sess = pa.Session(pest, dis, ask=None, exclude=[case["image"]], **case["session"])
    hint = ""
    if mode == "baseline" and case.get("labels"):  # closed-set baseline: knows the label wording, sees no references
        hint = "If one of these labels applies, give it exactly in condition_label (use 'healthy' for a healthy plant): " + "; ".join(case["labels"])
    t0 = time.time()
    try:
        res = pa.diagnose([case["image"]], hint, session=sess, llm=llm, lang="English", use_tools=mode == "agent")
    except Exception as e:  # one failed call must not sink the run
        return {"case": case, "mode": mode, "error": repr(e)[:300]}
    d = res["diagnosis"]
    return {"case": case, "mode": mode, "diagnosis": d, "scores": score(case, d, pest, dis), "steps": res["steps"],
            "tools": [t["tool"] for t in res["trace"]], "tokens": res["usage"].get("total_tokens", 0), "seconds": round(time.time() - t0, 1)}


def summarize(rows: list[dict]) -> None:
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode and "scores" in r]
        errs = sum(1 for r in rows if r["mode"] == mode and "error" in r)
        print(f"\n[{mode}] n={len(rs)} errors={errs} tokens={sum(r['tokens'] for r in rs):,} mean steps={sum(r['steps'] for r in rs) / max(len(rs), 1):.1f}")
        for m in sorted({k for r in rs for k in r["scores"]}):
            vals = [r["scores"][m] for r in rs if m in r["scores"]]  # a metric counts only the cases it applies to
            k = sum(map(bool, vals))
            lo, hi = wilson(k, len(vals))
            print(f"  {m:14s} {k:3d}/{len(vals):<3d} = {k / len(vals):5.1%}   95% CI {lo:.0%}-{hi:.0%}")
        right = [float(r["diagnosis"]["confidence"]) for r in rs if r["scores"]["correct"]]
        wrong = [float(r["diagnosis"]["confidence"]) for r in rs if not r["scores"]["correct"]]
        if right and wrong:  # a usable confidence is higher when right than when wrong
            print(f"  mean confidence: correct {sum(right) / len(right):.2f}, wrong {sum(wrong) / len(wrong):.2f}")


def selftest() -> None:
    assert name_hit("Fungus gnat (Bradysia sp.)", ["FUNGUS GNATS ", "Bradysia spp."])
    assert name_hit("Pythium root rot", ["Root Rot"]) and not name_hit("leaf spot", ["Root Rot"])
    assert name_hit("twospotted spider mite", ["two-spotted spider mite"]) and not name_hit("termite", ["mite"])
    assert wilson(0, 0) == (0.0, 0.0) and 0.2 < wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1] < 0.8
    pest, dis = pa.PestKB(), pa.DiseaseKB()
    cases = pest_cases(pest, 3, random.Random(0)) + disease_cases(dis, 2, random.Random(0), None)
    rows = []
    for c in cases:  # oracle "model" submits the ground truth: every case must score correct, proving the scoring path
        ans = {"host_plant": c.get("plant", "unknown"), "category": "healthy" if c.get("label") == "healthy" else "pest", "name": "x",
               "confidence": 0.5, "evidence": [], "differentials": [],
               "next_checks": [], "actions": [], "summary": "", "book_entry_id": c.get("entry_id", ""), "condition_label": c.get("label", "")}
        oracle = lambda m, t, ans=ans: ({"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "submit_diagnosis", "arguments": json.dumps(ans)}}]}, {})
        rows.append(run_case(c, "agent", pest, dis, llm=oracle))
    assert all(r["scores"]["correct"] for r in rows), [r.get("scores") or r.get("error") for r in rows]
    assert all(r["scores"].get("host_hit", True) for r in rows if r["case"]["task"] == "disease")
    assert all(Path(c["image"]).resolve() not in {p.resolve() for ref, _ in dis.photos(c["plant"], c["label"]) if (p := pa.resolve_img(ref))}
               or c.get("set") == "houseplant_web" for c in cases if c["task"] == "disease"), "held-out query is also a reference"
    summarize(rows)
    print("eval selftest ok")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["pest", "disease", "healthy", "selftest"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", help="disease: only this reference set")
    ap.add_argument("--dir", help="healthy: folder of images of healthy plants")
    ap.add_argument("--baseline", action="store_true", help="also run the same model with no tools")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.task == "selftest":
        return selftest()
    pest, dis, rng = pa.PestKB(), pa.DiseaseKB(), random.Random(a.seed)
    cases = (pest_cases(pest, a.n, rng) if a.task == "pest" else disease_cases(dis, a.n, rng, a.set) if a.task == "disease"
             else healthy_cases(a.dir or sys.exit("healthy needs --dir"), a.n, rng))
    jobs = [(c, m) for c in cases for m in (["agent", "baseline"] if a.baseline else ["agent"])]
    out = pa.ROOT / "eval_results" / f"{a.task}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex, open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(ex.map(lambda j: run_case(*j, pest, dis), jobs), 1):
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            f.flush()
            mark = "ERR" if "error" in r else "ok " if r["scores"]["correct"] else "-- "
            print(f"{i:3d}/{len(jobs)} {mark} {r['mode']:8s} {Path(r['case']['image']).name[:50]}", flush=True)
    summarize(rows)
    print(f"\nmodel: {pa.MODEL}\nsaved: {out}")


if __name__ == "__main__":
    main()
