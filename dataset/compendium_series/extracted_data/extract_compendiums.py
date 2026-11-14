"""Extract clean Markdown and section-aware RAG chunks from APS Compendium PDFs.

Key features:
- Filters out front-matter noise (cover, copyright, financial sponsors, author address
  directories, table of contents), OCR gibberish on color plates, bibliographic
  reference lists (optional via --include-references), and back-of-book index pages.
- Preserves true document hierarchy: Disease/Topic Section -> Subsections
  (Overview, Symptoms, Causal Organism, Disease Cycle, Epidemiology, Control/Management).
- Packs subsections within the same disease section into cohesive RAG chunks,
  tracks exact page ranges per chunk, splits strictly on paragraph/sentence/word
  boundaries (never mid-word), and merges tiny trailing fragments.

Usage:
    E:\\anaconda3\\envs\\ragbot\\python.exe extract_compendiums.py
    E:\\anaconda3\\envs\\ragbot\\python.exe extract_compendiums.py --limit 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

CONTENT_SUBSECTIONS = {
    "symptoms",
    "symptoms and signs",
    "signs and symptoms",
    "causal organism",
    "causal organisms",
    "causal agent",
    "causal agents",
    "pathogen",
    "pathogens",
    "etiology",
    "disease cycle",
    "disease cycle and epidemiology",
    "epidemiology",
    "epidemiology and disease cycle",
    "control",
    "management",
    "disease management",
    "integrated management",
    "control and management",
    "host range",
    "hosts",
    "host range and distribution",
    "geographic distribution",
    "distribution",
    "economic importance",
    "importance",
    "biology",
    "life cycle",
    "description",
    "diagnosis",
    "identification",
    "transmission",
    "vectors",
    "varietal resistance",
    "host resistance",
}

REFERENCE_SUBSECTIONS = {
    "selected references",
    "selected reference",
    "references",
    "reference",
    "general references",
    "literature cited",
    "further reading",
}

ALL_SUBSECTIONS = CONTENT_SUBSECTIONS | REFERENCE_SUBSECTIONS

ARCHIVE_BOILERPLATE_RE = re.compile(
    r"digitized by the internet archive|https?://archive\.org/details/",
    re.IGNORECASE,
)
PREPARED_BY_RE = re.compile(r"^\(Prepared by .+\)$", re.IGNORECASE)
PART_HEADER_RE = re.compile(r"^Part\s+[IVX0-9|l]+\.\s+.+$", re.IGNORECASE)
FIGURE_CAPTION_RE = re.compile(r"^(Fig\.|Figure|Plate|Table)\s+\d+", re.IGNORECASE)
INDEX_HEADER_RE = re.compile(r"^(?:\d+\s+)?Index(?:\s+\d+)?$", re.IGNORECASE)
GLOSSARY_HEADER_RE = re.compile(r"^(?:\d+\s+)?Glossary(?:\s+\d+)?$", re.IGNORECASE)
CONTENTS_HEADER_RE = re.compile(
    r"^(?:Table\s+of\s+)?Contents$", re.IGNORECASE
)
INTRODUCTION_HEADER_RE = re.compile(r"^Introduction$", re.IGNORECASE)
PREFACE_HEADER_RE = re.compile(r"^Preface$", re.IGNORECASE)
INDEX_LINE_RE = re.compile(
    r"(?:,\s*(?:Pl\.|Pls\.|Fig\.|Figs\.)?\s*\d+(?:-\d+)?(?:\s*,\s*\d+(?:-\d+)?)*\s*$)|(?:\bSee\s+also\b)|(?:\.\s+See\s+[A-Z])"
)
AUTHOR_ADDRESS_RE = re.compile(
    r"\b(?:Department of|University of|State University|Agricultural Research Service|Agriculture Canada|Experiment Station|Box \d+|Avenue|Street|Road|Hall|Lab\.|[A-Z]{2}\s+\d{5})\b",
    re.IGNORECASE,
)


@dataclass
class ParagraphItem:
    """A cleaned paragraph tied to its exact 1-based page number and subsection."""

    page: int
    subsection: str
    text: str
    is_reference: bool = False


@dataclass
class DiseaseSection:
    """A complete disease or topic section containing ordered ParagraphItems."""

    book_title: str
    crop: str
    source_file: str
    section: str
    items: list[ParagraphItem]

    @property
    def page_start(self) -> int:
        return min(i.page for i in self.items) if self.items else 1

    @property
    def page_end(self) -> int:
        return max(i.page for i in self.items) if self.items else 1


@dataclass
class RagChunk:
    """A cohesive, section-aware chunk ready for RAG or LLM pipelines."""

    chunk_id: str
    source_file: str
    book_title: str
    crop: str
    section: str
    subsection: str | None
    subsections: list[str]
    page_start: int
    page_end: int
    char_count: int
    word_count: int
    text: str


def slugify(name: str) -> str:
    """Convert a filename stem into a clean lowercase slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return slug.strip("_")


def clean_book_title(stem: str) -> str:
    """Normalize a PDF filename stem into a clean human-readable title."""
    title = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip()
    return re.sub(r"\s+", " ", title)


def infer_crop_from_filename(stem: str) -> str:
    """Infer the crop/plant name from a compendium PDF filename."""
    cleaned = re.sub(r"^compendium\s+of\s+", "", stem, flags=re.IGNORECASE)
    cleaned = re.sub(
        r",?\s*\d+(?:st|nd|rd|th)\s+edition.*$", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", cleaned)
    cleaned = re.sub(
        r"\s+(diseases?|pests?|insects?)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip().title() or stem


def is_ocr_noise_line(line: str) -> bool:
    """Return True if a line looks like OCR gibberish from a scanned plate/cover."""
    s = line.strip()
    if not s:
        return True
    if re.search(r"DATE\s+DUE", s, re.IGNORECASE):
        return True
    alpha_chars = sum(1 for c in s if c.isalpha())
    if len(s) >= 12 and alpha_chars / len(s) < 0.48:
        return True
    tokens = s.split()
    if len(tokens) >= 4:
        short_or_nonalpha = sum(
            1 for t in tokens if len(t) <= 2 or not any(c.isalpha() for c in t)
        )
        if short_or_nonalpha / len(tokens) > 0.58:
            return True
    return False


def is_front_matter_noise_page(raw_text: str, page_num: int) -> bool:
    """Identify cover, copyright, sponsor, author-directory, and TOC pages in front matter."""
    if page_num > 15:
        return False
    s = raw_text.strip()
    if not s or len(s) < 350:
        return True
    lower = s.lower()
    # Copyright / title / sponsor pages
    if any(
        kw in lower
        for kw in (
            "library of congress catalog",
            "all rights reserved",
            "the disease compendium series",
            "financial sponsors",
            "thank you to the sponsors",
            "international standard book number",
        )
    ) and "preface" not in lower and "introduction" not in lower:
        return True

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    # Half-title / title pages
    if page_num <= 5 and any(
        re.match(r"^(?:edited by|first edition|second edition|third edition|fourth edition|aps press)$", ln, re.I)
        for ln in lines
    ) and "preface" not in lower:
        return True

    # Table of contents pages before page 15
    if any(CONTENTS_HEADER_RE.match(ln) for ln in lines):
        return True
    if "color plates" in lower and "glossary" in lower and "index" in lower and page_num <= 14:
        return True
    # Continuation of TOC (many standalone page numbers)
    num_tokens = re.findall(r"\b\d{1,3}\b", s)
    if page_num <= 13 and len(num_tokens) >= 15 and len(s) < 3500:
        if "introduction" not in lower or re.search(r"introduction\s+1\b", lower):
            return True

    # Author directory pages (many university/department/address lines)
    addr_matches = sum(1 for ln in lines if AUTHOR_ADDRESS_RE.search(ln))
    if page_num <= 14 and addr_matches >= 5 and "preface" not in lower and "introduction" not in lower:
        return True

    return False


def is_index_or_back_matter_page(raw_text: str, page_num: int, total_pages: int) -> bool:
    """Identify back-of-book alphabetical Index pages and back cover."""
    if page_num < max(15, int(total_pages * 0.78)):
        return False
    s = raw_text.strip()
    if not s:
        return True
    lower = s.lower()
    if "the disease compendium series" in lower or re.search(r"^isbn\s+[0-9-]+", lower, re.M):
        return True

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return True
    if any(INDEX_HEADER_RE.match(ln) for ln in lines[:5]):
        return True

    # Count lines that look like index entries ("Term, 12, 15-16" or "See also")
    index_like = sum(1 for ln in lines if INDEX_LINE_RE.search(ln))
    if len(lines) >= 8 and (index_like / len(lines)) >= 0.30:
        return True
    return False


def clean_page_lines(raw_text: str) -> list[str]:
    """Clean raw pypdf page text and rejoin hyphenated line wraps."""
    if not raw_text:
        return []
    if ARCHIVE_BOILERPLATE_RE.search(raw_text) and len(raw_text.strip()) < 400:
        return []

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        ln
        for ln in (re.sub(r"[ \t]+", " ", raw_ln).strip() for raw_ln in text.split("\n"))
        if ln and not is_ocr_noise_line(ln)
    ]

    # Drop standalone page numbers at top or bottom of page
    while lines and re.fullmatch(r"\d{1,4}|[ivxlcdm]+", lines[0], re.IGNORECASE):
        lines.pop(0)
    while lines and re.fullmatch(r"\d{1,4}|[ivxlcdm]+", lines[-1], re.IGNORECASE):
        lines.pop()

    # Rejoin words hyphenated across line breaks
    joined: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.endswith("-")
            and len(line) >= 2
            and line[-2].isalpha()
            and i + 1 < len(lines)
            and lines[i + 1]
            and lines[i + 1][0].islower()
        ):
            next_line = lines[i + 1]
            parts = next_line.split(" ", 1)
            merged_word = line[:-1] + parts[0]
            if len(parts) > 1:
                lines[i + 1] = parts[1]
                joined.append(merged_word)
            else:
                joined.append(merged_word)
                i += 1
        else:
            joined.append(line)
        i += 1

    return [ln for ln in joined if ln]


def match_subsection(line: str) -> tuple[str | None, bool]:
    """If line is a standard subsection heading, return (canonical_title, is_reference)."""
    s = line.strip().rstrip(":")
    if not s or not s[0].isupper():
        return None, False
    norm = s.lower()
    if norm in REFERENCE_SUBSECTIONS:
        return s, True
    if norm in CONTENT_SUBSECTIONS:
        return s, False
    return None, False


def is_disease_or_topic_heading(
    line: str,
    prev_line: str,
    next_line: str,
    in_glossary: bool,
    in_preface: bool,
) -> bool:
    """Detect major disease or chapter headings while avoiding false positives."""
    s = line.strip()
    if not s or len(s) < 3 or len(s) > 85:
        return False
    if GLOSSARY_HEADER_RE.match(s) or INTRODUCTION_HEADER_RE.match(s) or PREFACE_HEADER_RE.match(s):
        return True
    if in_glossary:
        return False
    if PART_HEADER_RE.match(s):
        return True
    if in_preface:
        return False
    if match_subsection(s)[0] is not None:
        return False
    if FIGURE_CAPTION_RE.match(s) or PREPARED_BY_RE.match(s):
        return False
    # Reject lines with sentence punctuation or unmatched parentheses
    if s[-1] in ".;,?!:)" or ". " in s or "(" in s or ")" in s:
        return False
    if any(ch.isdigit() for ch in s):
        return False

    words = s.split()
    if len(words) > 9:
        return False

    minor = {
        "and",
        "or",
        "of",
        "the",
        "in",
        "on",
        "by",
        "to",
        "for",
        "with",
        "a",
        "an",
        "caused",
        "associated",
        "that",
        "principally",
        "occur",
    }
    if not words[0][0].isupper():
        return False
    cap_count = sum(1 for w in words if w[0].isupper() or w.lower() in minor)
    if cap_count < len(words):
        return False

    # Strong signal: immediately follows "(Prepared by ...)"
    if prev_line and PREPARED_BY_RE.match(prev_line):
        return True

    # Otherwise must be followed by a prose sentence or a known subsection heading
    if next_line:
        if match_subsection(next_line)[0] is not None:
            return True
        if len(next_line) >= 40 and next_line[0].isupper():
            return True
    return False


def lines_to_paragraphs(lines: list[str], in_glossary: bool = False) -> list[str]:
    """Group cleaned lines into coherent paragraphs."""
    if not lines:
        return []
    if in_glossary:
        # In Glossary, each "term—definition" starts a new entry
        entries: list[str] = []
        buf: list[str] = []
        for ln in lines:
            if re.match(r"^[A-Za-z°][A-Za-z0-9\s,().'-]{0,45}[—–-]\s*\S", ln) and buf:
                entries.append(" ".join(buf))
                buf = [ln]
            else:
                buf.append(ln)
        if buf:
            entries.append(" ".join(buf))
        return entries

    paragraphs: list[str] = []
    buf = []

    def flush() -> None:
        if buf:
            paragraphs.append(" ".join(buf))
            buf.clear()

    for ln in lines:
        if PREPARED_BY_RE.match(ln) or FIGURE_CAPTION_RE.match(ln):
            flush()
            paragraphs.append(ln)
            continue
        buf.append(ln)
        if ln.endswith((".", "!", "?")) and len(ln) < 58:
            flush()

    flush()
    return paragraphs


def extract_disease_sections(
    pdf_path: Path, include_references: bool = False
) -> tuple[list[DiseaseSection], dict]:
    """Parse a compendium PDF into structured DiseaseSections."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    book_title = clean_book_title(pdf_path.stem)
    crop = infer_crop_from_filename(pdf_path.stem)

    sections: list[DiseaseSection] = []
    current_section_name = "Introduction"
    current_subsection = "Overview"
    is_ref_sub = False
    in_glossary = False
    in_preface = False
    started_body = False

    current_items: list[ParagraphItem] = []
    line_buf: list[str] = []
    buf_page = 1

    skipped_front_pages = 0
    skipped_back_pages = 0
    empty_pages = 0
    extracted_chars = 0

    def flush_lines() -> None:
        nonlocal line_buf
        if not line_buf:
            return
        paras = lines_to_paragraphs(line_buf, in_glossary=in_glossary)
        line_buf = []
        if is_ref_sub and not include_references:
            return
        for p in paras:
            if len(p.strip()) < 3:
                continue
            current_items.append(
                ParagraphItem(
                    page=buf_page,
                    subsection="Glossary" if in_glossary else current_subsection,
                    text=p.strip(),
                    is_reference=is_ref_sub,
                )
            )

    def flush_section() -> None:
        nonlocal current_items
        flush_lines()
        if current_items:
            sections.append(
                DiseaseSection(
                    book_title=book_title,
                    crop=crop,
                    source_file=pdf_path.name,
                    section=current_section_name,
                    items=current_items,
                )
            )
            current_items = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw = page.extract_text() or ""

        if not raw.strip():
            empty_pages += 1
            continue

        if is_front_matter_noise_page(raw, page_num):
            skipped_front_pages += 1
            continue

        if is_index_or_back_matter_page(raw, page_num, total_pages):
            skipped_back_pages += 1
            # Once we hit the alphabetical Index at the back of the book, stop body extraction
            if page_num >= int(total_pages * 0.82) and skipped_back_pages >= 2:
                break
            continue

        lines = clean_page_lines(raw)
        if not lines:
            empty_pages += 1
            continue

        # Flush buffered lines at page boundary so paragraph page numbers stay exact
        flush_lines()
        buf_page = page_num
        extracted_chars += sum(len(ln) for ln in lines)

        for i, line in enumerate(lines):
            prev_line = lines[i - 1] if i > 0 else ""
            next_line = lines[i + 1] if i + 1 < len(lines) else ""

            if INDEX_HEADER_RE.match(line) and page_num >= int(total_pages * 0.75):
                flush_section()
                break

            if GLOSSARY_HEADER_RE.match(line) and page_num >= int(total_pages * 0.65):
                flush_section()
                current_section_name = "Glossary"
                current_subsection = "Glossary"
                in_glossary = True
                in_preface = False
                is_ref_sub = False
                buf_page = page_num
                continue

            if PREFACE_HEADER_RE.match(line) and page_num <= 12:
                flush_section()
                current_section_name = "Preface"
                current_subsection = "Overview"
                in_preface = True
                is_ref_sub = False
                buf_page = page_num
                continue

            if INTRODUCTION_HEADER_RE.match(line) and page_num <= 18:
                flush_section()
                current_section_name = "Introduction"
                current_subsection = "Overview"
                in_preface = False
                in_glossary = False
                is_ref_sub = False
                started_body = True
                buf_page = page_num
                continue

            # If we are still in Preface and hit the Authors list or address blocks, stop Preface
            if in_preface and (
                re.match(r"^(?:Authors|Contributors)$", line.strip(), re.I)
                or AUTHOR_ADDRESS_RE.search(line)
                or (i + 1 < len(lines) and AUTHOR_ADDRESS_RE.search(lines[i + 1]))
            ):
                flush_section()
                in_preface = False
                break

            sub_title, is_ref = match_subsection(line)
            if sub_title is not None and not in_glossary:
                flush_lines()
                current_subsection = sub_title
                is_ref_sub = is_ref
                buf_page = page_num
                started_body = True
                continue

            if is_disease_or_topic_heading(
                line, prev_line, next_line, in_glossary=in_glossary, in_preface=in_preface
            ):
                flush_section()
                current_section_name = re.sub(
                    r"^Part\s+[|l]\.", "Part I.", line.strip()
                )
                current_subsection = "Overview"
                is_ref_sub = False
                in_preface = False
                started_body = True
                buf_page = page_num
                continue

            line_buf.append(line)

    flush_section()

    stats = {
        "source_file": pdf_path.name,
        "book_title": book_title,
        "crop": crop,
        "total_pages": total_pages,
        "empty_pages": empty_pages,
        "skipped_front_matter_pages": skipped_front_pages,
        "skipped_back_index_pages": skipped_back_pages,
        "extracted_chars": extracted_chars,
        "sections_count": len(sections),
    }
    return sections, stats


def split_oversized_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split a long paragraph on sentence boundaries first, then whole words (never mid-word)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur_sents: list[str] = []
    cur_len = 0

    def flush_sents() -> None:
        nonlocal cur_sents, cur_len
        if cur_sents:
            chunks.append(" ".join(cur_sents))

    for sent in sentences:
        if len(sent) > max_chars:
            flush_sents()
            cur_sents = []
            cur_len = 0
            # Split huge sentence on whole words
            words = sent.split()
            start = 0
            while start < len(words):
                end = start
                wlen = 0
                while end < len(words):
                    add = len(words[end]) + (1 if end > start else 0)
                    if wlen + add > max_chars and end > start:
                        break
                    wlen += add
                    end += 1
                chunks.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                back = end
                ov = 0
                while back > start + 1 and ov < overlap_chars:
                    back -= 1
                    ov += len(words[back]) + 1
                start = back
            continue

        add_len = len(sent) + (1 if cur_sents else 0)
        if cur_len + add_len <= max_chars:
            cur_sents.append(sent)
            cur_len += add_len
        else:
            flush_sents()
            # Compute clean sentence/word overlap from cur_sents
            overlap_buf: list[str] = []
            ov_len = 0
            for prev_s in reversed(cur_sents):
                if ov_len + len(prev_s) + 1 <= overlap_chars:
                    overlap_buf.insert(0, prev_s)
                    ov_len += len(prev_s) + 1
                else:
                    break
            cur_sents = overlap_buf + [sent]
            cur_len = len(" ".join(cur_sents))

    if cur_sents:
        chunks.append(" ".join(cur_sents))
    return chunks


def build_section_chunks(
    sections: list[DiseaseSection],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int = 280,
) -> list[RagChunk]:
    """Pack paragraphs/subsections within each DiseaseSection into cohesive RagChunks."""
    all_chunks: list[RagChunk] = []
    chunk_counter = 0

    for sec in sections:
        slug = slugify(sec.book_title)
        # Expand any single item larger than chunk_size into sentence-aligned sub-items
        expanded_items: list[ParagraphItem] = []
        for item in sec.items:
            if len(item.text) > chunk_size:
                for sub_txt in split_oversized_text(item.text, chunk_size, chunk_overlap):
                    expanded_items.append(
                        ParagraphItem(
                            page=item.page,
                            subsection=item.subsection,
                            text=sub_txt,
                            is_reference=item.is_reference,
                        )
                    )
            else:
                expanded_items.append(item)

        # Group items into chunks up to chunk_size, formatting subsection headers inline
        raw_section_chunks: list[tuple[list[ParagraphItem], str]] = []
        buf_items: list[ParagraphItem] = []

        def render_items(items: list[ParagraphItem]) -> str:
            lines: list[str] = []
            last_sub: str | None = None
            for it in items:
                if it.subsection != last_sub:
                    if it.subsection not in ("Overview", "Glossary"):
                        lines.append(f"### {it.subsection}")
                    last_sub = it.subsection
                lines.append(it.text)
            return "\n\n".join(lines).strip()

        for item in expanded_items:
            candidate_items = buf_items + [item]
            rendered = render_items(candidate_items)
            if len(rendered) <= chunk_size or not buf_items:
                buf_items = candidate_items
            else:
                raw_section_chunks.append((buf_items, render_items(buf_items)))
                # Carry over trailing paragraph(s) as whole-paragraph overlap if they fit
                overlap_items: list[ParagraphItem] = []
                ov_chars = 0
                for prev_it in reversed(buf_items):
                    if ov_chars + len(prev_it.text) <= chunk_overlap:
                        overlap_items.insert(0, prev_it)
                        ov_chars += len(prev_it.text)
                    else:
                        break
                buf_items = overlap_items + [item]
                if len(render_items(buf_items)) > chunk_size:
                    buf_items = [item]

        if buf_items:
            raw_section_chunks.append((buf_items, render_items(buf_items)))

        # Merge tiny trailing chunk into previous chunk of the same section if possible
        if (
            len(raw_section_chunks) >= 2
            and len(raw_section_chunks[-1][1]) < min_chunk_chars
        ):
            prev_items, _ = raw_section_chunks[-2]
            tail_items, _ = raw_section_chunks[-1]
            merged_items = prev_items + [
                it for it in tail_items if it not in prev_items
            ]
            merged_text = render_items(merged_items)
            if len(merged_text) <= int(chunk_size * 1.35):
                raw_section_chunks[-2] = (merged_items, merged_text)
                raw_section_chunks.pop()

        for items_group, body_text in raw_section_chunks:
            if len(body_text.strip()) < 40:
                continue
            chunk_counter += 1
            subs: list[str] = []
            for it in items_group:
                if it.subsection not in subs:
                    subs.append(it.subsection)
            p_start = min(it.page for it in items_group)
            p_end = max(it.page for it in items_group)
            sub_label = ", ".join(subs) if subs else None
            header = (
                f"[{sec.book_title} | Crop: {sec.crop} | Section: {sec.section}"
                + (f" | Subsections: {sub_label}" if sub_label else "")
                + f" | pp. {p_start}-{p_end}]"
            )
            full_text = f"{header}\n{body_text}"
            all_chunks.append(
                RagChunk(
                    chunk_id=f"{slug}_{chunk_counter:05d}",
                    source_file=sec.source_file,
                    book_title=sec.book_title,
                    crop=sec.crop,
                    section=sec.section,
                    subsection=sub_label,
                    subsections=subs,
                    page_start=p_start,
                    page_end=p_end,
                    char_count=len(full_text),
                    word_count=len(full_text.split()),
                    text=full_text,
                )
            )

    return all_chunks


def render_markdown(book_title: str, crop: str, sections: list[DiseaseSection]) -> str:
    """Render extracted DiseaseSections into a clean Markdown document."""
    source_file = sections[0].source_file if sections else ""
    out: list[str] = [
        f"# {book_title}",
        "",
        f"- **Crop / Host:** {crop}",
        f"- **Source PDF:** `{source_file}`",
        "",
    ]
    for sec in sections:
        out.append(f"## {sec.section} *(pp. {sec.page_start}-{sec.page_end})*")
        out.append("")
        last_sub: str | None = None
        for item in sec.items:
            if item.subsection != last_sub:
                if item.subsection not in ("Overview", "Glossary"):
                    out.append(f"### {item.subsection}")
                    out.append("")
                last_sub = item.subsection
            out.append(item.text)
            out.append("")
    return "\n".join(out).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clean Markdown and section-aware RAG chunks from compendium PDFs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/compendium_series"),
        help="Directory containing compendium PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/compendium_series/extracted_data"),
        help="Directory inside dataset/compendium_series for extracted data.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1400,
        help="Target maximum characters per RAG chunk body (default: 1400).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Maximum character overlap on sentence/paragraph boundaries (default: 150).",
    )
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="Include 'Selected References' bibliographic lists in chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of PDFs to process (for quick testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if args.limit is not None:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        raise SystemExit(f"No PDF files found in {args.input_dir}")

    md_dir = args.output_dir / "markdown"
    chunks_dir = args.output_dir / "chunks"
    md_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Also save a copy of this extraction script inside the output folder for reproducibility
    script_path = Path(__file__).resolve()
    try:
        shutil.copy2(script_path, args.output_dir / script_path.name)
    except OSError:
        pass

    all_chunks_path = chunks_dir / "compendium_chunks.jsonl"
    summaries: list[dict] = []
    total_chunks = 0

    with all_chunks_path.open("w", encoding="utf-8") as all_out:
        for pdf_path in pdf_files:
            sections, stats = extract_disease_sections(
                pdf_path, include_references=args.include_references
            )
            slug = slugify(pdf_path.stem)

            md_text = render_markdown(stats["book_title"], stats["crop"], sections)
            (md_dir / f"{slug}.md").write_text(md_text, encoding="utf-8")

            chunks = build_section_chunks(
                sections,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            per_book_jsonl = chunks_dir / f"{slug}.jsonl"
            with per_book_jsonl.open("w", encoding="utf-8") as book_out:
                for chunk in chunks:
                    line = json.dumps(asdict(chunk), ensure_ascii=False)
                    book_out.write(line + "\n")
                    all_out.write(line + "\n")

            char_counts = [c.char_count for c in chunks]
            stats["chunks_count"] = len(chunks)
            stats["min_chunk_chars"] = min(char_counts) if char_counts else 0
            stats["avg_chunk_chars"] = (
                sum(char_counts) // len(char_counts) if char_counts else 0
            )
            stats["max_chunk_chars"] = max(char_counts) if char_counts else 0
            summaries.append(stats)
            total_chunks += len(chunks)
            print(
                f"Processed {pdf_path.name}: "
                f"{stats['total_pages']} pages -> "
                f"{stats['sections_count']} sections, "
                f"{stats['chunks_count']} chunks "
                f"(avg {stats['avg_chunk_chars']} chars)"
            )

    summary_payload = {
        "files_processed": len(summaries),
        "total_chunks": total_chunks,
        "chunk_size_target": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "include_references": args.include_references,
        "files": summaries,
    }
    (args.output_dir / "extraction_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDone. Wrote {total_chunks} total chunks to {all_chunks_path}")


if __name__ == "__main__":
    main()
