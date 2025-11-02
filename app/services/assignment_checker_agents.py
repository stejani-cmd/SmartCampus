# app/services/assignment_checker_agents.py
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re


class BaseAgent:
    name: str = "base"

    def run(self, **kwargs):
        raise NotImplementedError


# =========================================================
# 1. INGESTION AGENT
#    - extracts text
#    - detects dominant font name + size (for docx/pdf)
# =========================================================
class IngestionAgent(BaseAgent):
    name = "ingestion"

    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir

    # ---------- PDF ----------
    def _extract_pdf(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        text_parts: List[str] = []
        font_sizes_counter: Counter[int] = Counter()
        font_names_counter: Counter[str] = Counter()
        meta: Dict[str, Any] = {"notes": []}

        # 1) PyPDF2 for raw text
        try:
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    try:
                        text_parts.append(page.extract_text() or "")
                    except Exception:
                        pass
        except Exception:
            pass

        # 2) pdfplumber for font/size
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    # characters include font + size
                    chars = page.chars
                    for ch in chars:
                        size = ch.get("size")
                        fontname = ch.get("fontname")
                        if size:
                            try:
                                size_int = int(round(float(size)))
                                font_sizes_counter[size_int] += 1
                            except Exception:
                                pass
                        if fontname:
                            font_names_counter[fontname] += 1
        except Exception:
            meta["notes"].append("Could not read PDF font info (pdfplumber missing or PDF is image).")

        text = "\n".join(text_parts).strip()

        # pick dominant
        dominant_size = font_sizes_counter.most_common(1)[0][0] if font_sizes_counter else None
        dominant_font = font_names_counter.most_common(1)[0][0] if font_names_counter else None

        meta["font_sizes_counter"] = dict(font_sizes_counter)
        meta["font_names_counter"] = dict(font_names_counter)
        meta["dominant_font_size"] = dominant_size
        meta["dominant_font_name"] = dominant_font

        return text, meta

    # ---------- DOCX ----------
    def _extract_docx(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        import docx
        doc = docx.Document(str(path))

        text_parts: List[str] = []
        font_sizes: Counter[int] = Counter()
        font_names: Counter[str] = Counter()
        meta: Dict[str, Any] = {"notes": []}

        for para in doc.paragraphs:
            text_parts.append(para.text)

            # paragraph default
            para_default_size = None
            para_default_font = None
            if para.style is not None and para.style.font is not None:
                if para.style.font.size is not None:
                    try:
                        para_default_size = int(round(para.style.font.size.pt))
                    except Exception:
                        para_default_size = None
                if para.style.font.name is not None:
                    para_default_font = para.style.font.name

            for run in para.runs:
                # text
                # (we already added full para text above)

                # run size
                run_size = None
                run_font = None

                if run.font is not None and run.font.size is not None:
                    try:
                        run_size = int(round(run.font.size.pt))
                    except Exception:
                        run_size = None

                if run.font is not None and run.font.name:
                    run_font = run.font.name

                # fallback to para
                if run_size is None:
                    run_size = para_default_size
                if run_font is None:
                    run_font = para_default_font

                if run_size is not None:
                    font_sizes[run_size] += 1
                if run_font is not None:
                    font_names[run_font] += 1

        dominant_size = font_sizes.most_common(1)[0][0] if font_sizes else None
        dominant_font = font_names.most_common(1)[0][0] if font_names else None

        meta["font_sizes_counter"] = dict(font_sizes)
        meta["font_names_counter"] = dict(font_names)
        meta["dominant_font_size"] = dominant_size
        meta["dominant_font_name"] = dominant_font

        return "\n".join(text_parts), meta

    def _extract_from_file(self, path: Path, ext: str) -> Tuple[str, Dict[str, Any]]:
        ext = ext.lower()

        # txt
        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore"), {
                "notes": ["plain text file, no font metadata."],
                "dominant_font_size": None,
                "dominant_font_name": None,
            }

        # pdf
        if ext == ".pdf":
            return self._extract_pdf(path)

        # docx
        if ext == ".docx":
            try:
                return self._extract_docx(path)
            except ImportError:
                return "", {"notes": ["python-docx not installed."], "dominant_font_size": None, "dominant_font_name": None}

        # other
        return "", {"notes": [f"File type {ext} not supported for font detection."],
                    "dominant_font_size": None, "dominant_font_name": None}

    def run(
        self,
        *,
        file_path: Optional[Path],
        ext: Optional[str],
        text_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"notes": []}

        # if user pasted text
        if text_override:
            if file_path and ext:
                _, submeta = self._extract_from_file(file_path, ext)
                meta.update(submeta)
            return {
                "text": text_override,
                "meta": meta,
            }

        if not file_path or not ext:
            meta["notes"].append("No file provided.")
            return {"text": "", "meta": meta}

        text, submeta = self._extract_from_file(file_path, ext)
        meta.update(submeta)
        return {"text": text, "meta": meta}


# =========================================================
# 2. REQUIREMENT AGENT
#    - understands "font size 12", "font: calibri", "use arial 11"
# =========================================================
class RequirementAgent(BaseAgent):
    name = "requirements"

    FONT_SIZE_PAT = re.compile(r"(font\s*size\s*:?|size\s*)(\d+)", re.IGNORECASE)
    FONT_NAME_PAT = re.compile(r"(font\s*:?|use\s+)([a-zA-Z0-9 \-]+)", re.IGNORECASE)

    def run(self, *, raw_requirements: str) -> List[Dict[str, Any]]:
        lines = [l.strip() for l in raw_requirements.splitlines() if l.strip()]
        structured: List[Dict[str, Any]] = []

        for line in lines:
            low = line.lower()
            req_type = "text"
            target: Dict[str, Any] = {}

            # detect font size
            m = self.FONT_SIZE_PAT.search(line)
            if m:
                size = int(m.group(2))
                structured.append({
                    "raw": line,
                    "type": "format_font_size",
                    "target": {"size": size},
                })
                continue

            # detect font name (e.g. font: calibri, use times new roman)
            if "font" in low or low.startswith("use "):
                m2 = self.FONT_NAME_PAT.search(line)
                if m2:
                    fontname = m2.group(2).strip()
                    structured.append({
                        "raw": line,
                        "type": "format_font_name",
                        "target": {"font": fontname.lower()},
                    })
                    continue

            # default → text
            structured.append({
                "raw": line,
                "type": "text",
                "target": {},
            })

        return structured


# =========================================================
# 3. SCORING AGENT
# =========================================================
class ScoringAgent(BaseAgent):
    name = "scoring"

    STOPWORDS = {
        "include", "includes", "included",
        "add", "added", "make", "create",
        "the", "a", "an", "to", "of", "and", "for", "with", "in", "on", "at",
        "that", "this", "should", "must", "please", "write", "explain",
        "minimum", "maximum", "requirement", "requirements", "be", "is", "are"
    }

    def _tokenize_content_words(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if t not in self.STOPWORDS]

    def run(
        self,
        *,
        doc_text: str,
        doc_meta: Dict[str, Any],
        requirements: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not requirements:
            return {"score": 0, "details": []}

        text = (doc_text or "").lower()
        dom_size = doc_meta.get("dominant_font_size")
        dom_font = doc_meta.get("dominant_font_name")
        notes = doc_meta.get("notes", [])

        details: List[Dict[str, Any]] = []
        passed_count = 0

        for req in requirements:
            raw = req["raw"]
            rtype = req["type"]

            # ---------- FONT SIZE ----------
            if rtype == "format_font_size":
                wanted = req["target"].get("size")
                if wanted is None:
                    details.append({
                        "requirement": raw,
                        "passed": False,
                        "reason": "Requirement asked for font size but no numeric value was found.",
                    })
                    continue

                if dom_size is None:
                    # no font info
                    reason = "Could not detect font size from document."
                    if notes:
                        reason += " " + " ".join(notes)
                    details.append({
                        "requirement": raw,
                        "passed": False,
                        "reason": reason,
                        "actual": None,
                    })
                else:
                    if dom_size == wanted:
                        details.append({
                            "requirement": raw,
                            "passed": True,
                            "reason": f"Document dominant font size is {dom_size}pt which matches requirement.",
                            "actual": dom_size,
                        })
                        passed_count += 1
                    else:
                        details.append({
                            "requirement": raw,
                            "passed": False,
                            "reason": f"Document dominant font size is {dom_size}pt, required {wanted}pt.",
                            "actual": dom_size,
                        })
                continue

            # ---------- FONT NAME ----------
            if rtype == "format_font_name":
                wanted_font = req["target"].get("font")
                if not wanted_font:
                    details.append({
                        "requirement": raw,
                        "passed": False,
                        "reason": "Requirement asked for font but no font name was found.",
                    })
                    continue

                if dom_font is None:
                    reason = "Could not detect font name from document."
                    if notes:
                        reason += " " + " ".join(notes)
                    details.append({
                        "requirement": raw,
                        "passed": False,
                        "reason": reason,
                        "actual": None,
                    })
                else:
                    # compare lowercase, also allow partial
                    actual_low = dom_font.lower()
                    if wanted_font in actual_low:
                        details.append({
                            "requirement": raw,
                            "passed": True,
                            "reason": f"Document dominant font is '{dom_font}' which matches '{wanted_font}'.",
                            "actual": dom_font,
                        })
                        passed_count += 1
                    else:
                        details.append({
                            "requirement": raw,
                            "passed": False,
                            "reason": f"Document dominant font is '{dom_font}', required '{wanted_font}'.",
                            "actual": dom_font,
                        })
                continue

            # ---------- TEXT REQUIREMENTS ----------
            # 1) exact phrase
            if raw.lower() in text:
                details.append({
                    "requirement": raw,
                    "passed": True,
                    "reason": "Exact phrase found.",
                })
                passed_count += 1
                continue

            # 2) content match
            req_content = self._tokenize_content_words(raw)
            if not req_content:
                details.append({
                    "requirement": raw,
                    "passed": False,
                    "reason": "Requirement had no meaningful words.",
                })
                continue

            doc_tokens = set(self._tokenize_content_words(text))
            overlap = [w for w in req_content if w in doc_tokens]
            overlap_count = len(overlap)
            needed = max(1, int(len(req_content) * 0.6))  # 60%

            if overlap_count >= needed:
                details.append({
                    "requirement": raw,
                    "passed": True,
                    "reason": f"Matched words: {overlap}",
                })
                passed_count += 1
            else:
                miss = f"Needed {needed} words, found {overlap_count} ({overlap})."
                if notes:
                    miss += " " + " ".join(notes)
                details.append({
                    "requirement": raw,
                    "passed": False,
                    "reason": miss,
                })

        score = round((passed_count / len(requirements)) * 100)
        return {
            "score": score,
            "details": details,
        }


# =========================================================
# 4. FEEDBACK AGENT
# =========================================================
class FeedbackAgent(BaseAgent):
    name = "feedback"

    def run(self, *, details: List[Dict[str, Any]], notes: List[str] | None = None) -> Dict[str, Any]:
        unmet = [d for d in details if not d["passed"]]
        met = [d for d in details if d["passed"]]

        to_fix = []
        for d in unmet:
            # show actual if present
            actual = d.get("actual")
            if actual is not None:
                to_fix.append(f"❌ {d['requirement']} — {d['reason']} (found: {actual})")
            else:
                to_fix.append(f"❌ {d['requirement']} — {d['reason']}")

        already = [f"✅ {d['requirement']}" for d in met]

        return {
            "to_fix": to_fix,
            "met": already,
            "notes": notes or [],
        }
