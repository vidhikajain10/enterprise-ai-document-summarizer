 #!/usr/bin/env python3
# defenseint_summarizer.py
# Hybrid-ready: uses Mistral via HF Inference API when internet+token available,
# falls back to DistilBART local summarizer when offline or on failure.

import os
import io
import sys
import re
import tempfile
import logging
import argparse
import datetime
import urllib.request
from typing import List, Tuple, Optional, Dict

# Core libs for pipeline
import pdfplumber
import docx
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import requests
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import requests
# ---------------------------
# Logging
# ---------------------------
LOGFILE = "defenseint.log"
logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("DefenseInt_v3")
logger.setLevel(logging.DEBUG)

# ---------------------------
# Config
# ---------------------------
PRECISION_RATIO = {
    "Low": 0.20,      # 20% of document word count
    "Medium": 0.10,   # 10%
    "High": 0.04      # 4%
}
LOCAL_FALLBACK_MODEL = "sshleifer/distilbart-cnn-12-6"

# HF / Mistral settings
HF_MISTRAL_MODEL = os.environ.get("MISTRAL_HF_MODEL", "mistralai/mistral-7b-instruct")
HF_MISTRAL_API = f"https://api-inference.huggingface.co/models/{HF_MISTRAL_MODEL}"
HF_TIMEOUT = 90

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)

# ---------------------------
# Helpers: internet & mode decision (INJECTED)
# ---------------------------

import socket

def has_internet(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """
    Quick socket attempt to a public DNS server (Google DNS 8.8.8.8) on port 53.
    This only checks basic IP connectivity & routing, not HTTPS endpoints.
    """
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except Exception as e:
        logger.debug("has_internet socket check failed: %s", e)
        return False


    

def choose_mode_preference(ui_choice: str, hf_token: Optional[str], force_offline_flag: bool=False) -> str:
    """
    Decide effective mode for a request.
    Returns 'online' or 'offline'.
    Behavior:
    - If force_offline_flag True -> 'offline'
    - If ui_choice indicates Offline -> 'offline'
    - If ui_choice indicates Online -> attempt online if hf_token+internet available; else fallback to offline with logged warning.
    - If ui_choice Auto -> prefer online if hf_token+internet present, else offline.
    """
    if force_offline_flag:
        logger.info("choose_mode_preference: force_offline_flag set -> offline")
        return "offline"

    choice = (ui_choice or "").lower()
    online_requested = "online" in choice and "auto" not in choice
    offline_requested = "offline" in choice

    if offline_requested:
        return "offline"

    # If explicit online request
    if online_requested:
        if not hf_token:
            logger.warning("Online mode requested but HF token missing -> falling back to offline.")
            return "offline"
        if not has_internet():
            logger.warning("Online mode requested but no internet -> falling back to offline.")
            return "offline"
        return "online"

    # Auto / default
    if hf_token and has_internet():
        return "online"
    return "offline"

# ---------------------------
# Utility helpers (existing)
# ---------------------------
def make_output_filename_preserve(src_name: str, precision_label: str) -> str:
    """Make a safe output filename preserving source name and precision."""
    base = os.path.splitext(os.path.basename(src_name or "pasted_input"))[0]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = next(tempfile._get_candidate_names())
    safe_base = re.sub(r'[^A-Za-z0-9_.-]+', '_', base)
    fname = f"{safe_base}_{precision_label}_{stamp}_{uid}.txt"
    return os.path.join("outputs", fname)

def atomic_write_text(path: str, text: str):
    """Write to a temp file then move to path atomically (best-effort)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="atomic_", dir=os.path.dirname(path))
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def compute_word_token_bertscore(summary: str, reference: str, lang: str="en") -> dict:
    """Simple metrics placeholder. Replace with real BERTScore if you like."""
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(reference, summary)
        words_summary = len(summary.split())
        words_ref = len(reference.split())
        return {"words_summary": words_summary, "words_reference": words_ref, "rougeL": score['rougeL'].fmeasure}
    except Exception as e:
        logger.exception("compute metrics failed: %s", e)
        return {"words_summary": len(summary.split()), "words_reference": len(reference.split()), "rougeL": None}

def format_metrics_for_status(metrics: dict) -> str:
    return "\n".join([f"{k}: {v}" for k, v in metrics.items()])

# ---------------------------
# _read_bytes_from_uploaded (robust)
# ---------------------------
def _read_bytes_from_uploaded(uploaded):
    """
    Normalizes many possible upload shapes into (bytes, filename).
    """
    if uploaded is None:
        raise RuntimeError("No file uploaded.")
    if isinstance(uploaded, (bytes, bytearray)):
        return bytes(uploaded), "uploaded_bytes"
    if isinstance(uploaded, dict):
        for k in ("tmp_path", "tempfile", "tempfile_path", "file"):
            if k in uploaded and uploaded.get(k):
                path = uploaded.get(k)
                if isinstance(path, str) and os.path.exists(path):
                    with open(path, "rb") as f:
                        return f.read(), uploaded.get("name", os.path.basename(path))
        if "data" in uploaded and uploaded["data"] is not None:
            d = uploaded["data"]
            if isinstance(d, (bytes, bytearray)):
                return bytes(d), uploaded.get("name", "uploaded_file")
            if isinstance(d, str):
                try:
                    return d.encode("utf-8"), uploaded.get("name", "uploaded_file")
                except Exception:
                    pass
        if "file" in uploaded and hasattr(uploaded["file"], "read"):
            fobj = uploaded["file"]
            data = fobj.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            filename = uploaded.get("name") or getattr(fobj, "name", "uploaded_file")
            return data, os.path.basename(filename)
    if isinstance(uploaded, (list, tuple)):
        if len(uploaded) == 2 and isinstance(uploaded[0], (bytes, bytearray)) and isinstance(uploaded[1], str):
            return bytes(uploaded[0]), uploaded[1]
        return _read_bytes_from_uploaded(uploaded[0])
    if isinstance(uploaded, str):
        if os.path.exists(uploaded):
            with open(uploaded, "rb") as f:
                return f.read(), os.path.basename(uploaded)
        return str(uploaded).encode("utf-8"), "uploaded_string.txt"
    if hasattr(uploaded, "read"):
        data = uploaded.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        filename = getattr(uploaded, "name", None) or getattr(uploaded, "filename", None) or "uploaded_file"
        try_path = getattr(uploaded, "name", None)
        if try_path and isinstance(try_path, str) and os.path.exists(try_path):
            with open(try_path, "rb") as f:
                return f.read(), os.path.basename(try_path)
        return bytes(data), os.path.basename(filename)
    try:
        import pathlib
        if isinstance(uploaded, pathlib.Path):
            if uploaded.exists():
                with open(uploaded, "rb") as f:
                    return f.read(), uploaded.name
    except Exception:
        pass
    for attr in ("data", "content", "_content", "body"):
        if hasattr(uploaded, attr):
            candidate = getattr(uploaded, attr)
            if isinstance(candidate, (bytes, bytearray)):
                return bytes(candidate), getattr(uploaded, "name", "uploaded_file")
            if isinstance(candidate, str):
                return candidate.encode("utf-8"), getattr(uploaded, "name", "uploaded_file")
    typename = type(uploaded).__name__
    repr_snip = repr(uploaded)[:500]
    logger.error("Unsupported uploaded file shape encountered: type=%s repr=%s", typename, repr_snip)
    raise RuntimeError(f"Unsupported upload shape: {typename}. Check logs for repr (first 500 chars).")

# ---------------------------
# extract_text (PDF/DOCX/TXT) with OCR fallback
# ---------------------------
def extract_text(uploaded) -> Tuple[str, str]:
    data_bytes, filename = _read_bytes_from_uploaded(uploaded)
    ext = (filename or "").lower().split('.')[-1] if filename else ""
    if ext == "pdf":
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
                for page in pdf.pages:
                    try:
                        pt = page.extract_text()
                        if pt:
                            text += pt + "\n"
                    except Exception as e:
                        logger.debug("pdfplumber page extract failed: %s", e)
        except Exception as e:
            logger.exception("pdfplumber open failed: %s", e)
        # OCR fallback
        if len((text or "").strip()) < 50:
            try:
                logger.info("PDF low-text; running OCR fallback.")
                images = convert_from_bytes(data_bytes)
                ocr_text_parts = []
                for img in images:
                    try:
                        ocr_page = pytesseract.image_to_string(img)
                        ocr_text_parts.append(ocr_page)
                    except Exception as e:
                        logger.exception("pytesseract failed on page: %s", e)
                ocr_text = "\n".join(ocr_text_parts).strip()
                if ocr_text:
                    return ocr_text, filename
            except Exception as e:
                logger.exception("PDF->image conversion or OCR failed: %s", e)
        return (text.strip(), filename)
    if ext == "docx":
        try:
            doc = docx.Document(io.BytesIO(data_bytes))
            paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            return ("\n".join(paras)).strip(), filename
        except Exception as e:
            logger.exception("DOCX parsing failed: %s", e)
    # TXT fallback
    try:
        decoded = data_bytes.decode("utf-8")
    except Exception:
        decoded = data_bytes.decode("latin-1", errors="ignore")
    return decoded.strip(), filename

# ---------------------------
# Table extraction utilities
# ---------------------------
def extract_tables_from_pdf_bytes(raw_bytes: bytes) -> List[Tuple[int, List[List[str]]]]:
    tables = []
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    page_tables = page.extract_tables()
                    for tbl in page_tables:
                        rows = [[("" if cell is None else str(cell)).strip() for cell in row] for row in tbl]
                        if any(any(cell for cell in row) for row in rows):
                            tables.append((i, rows))
                except Exception as e:
                    logger.debug("Failed to extract tables on page %s: %s", i, e)
    except Exception as e:
        logger.exception("pdfplumber failed to open PDF for tables: %s", e)
    return tables

def table_rows_to_sentences(header: Optional[List[str]], rows: List[List[str]]) -> List[str]:
    sentences = []
    for row in rows:
        if not any(cell and str(cell).strip() for cell in row):
            continue
        if header and len(header) == len(row):
            parts = []
            for h, c in zip(header, row):
                if c and str(c).strip():
                    parts.append(f"{h.strip()}: {str(c).strip()}")
            if parts:
                sentences.append(", ".join(parts) + ".")
        else:
            joined = "; ".join([str(c).strip() for c in row if c and str(c).strip()])
            if joined:
                sentences.append(joined + ".")
    return sentences

def is_header_row(header_row: List[str], next_row: List[str]) -> bool:
    if not header_row or not next_row:
        return False
    header_row = [str(cell or "").strip() for cell in header_row]
    next_row   = [str(cell or "").strip() for cell in next_row]
    header_alpha = sum(any(char.isalpha() for char in cell) for cell in header_row)
    next_alpha   = sum(any(char.isalpha() for char in cell) for cell in next_row)
    header_numeric = sum(cell.replace(".", "").isdigit() for cell in header_row)
    next_numeric   = sum(cell.replace(".", "").isdigit() for cell in next_row)
    if header_alpha > next_alpha:
        return True
    if next_numeric > header_numeric:
        return True
    if any(not cell.replace(".", "").isdigit() for cell in header_row):
        return True
    return False

# ---------------------------
# Chapter splitting & chunking
# ---------------------------
def split_into_chapters(text: str) -> List[Tuple[str, str]]:
    if not text or not text.strip():
        return [("Document", "")]
    txt = text.replace("\r\n", "\n").replace("\r", "\n")
    heading_regex = re.compile(
        r'^\s*(CHAPTER\s+[0-9IVXLC]+|Chapter\s+[0-9A-Za-z]+|SECTION\s+\d+|Section\s+\d+)\s*$',
        flags=re.IGNORECASE | re.MULTILINE
    )
    matches = list(heading_regex.finditer(txt))
    if matches:
        starts = [m.start() for m in matches]
        starts.append(len(txt))
        titles = []
        for m in matches:
            line_end = txt.find("\n", m.start())
            title = (txt[m.start():line_end].strip() if line_end != -1 else txt[m.start():m.start()+80].strip())
            titles.append(title or f"Chapter_{len(titles)+1}")
        chapters = []
        for i in range(len(matches)):
            s = starts[i]
            e = starts[i+1] if i+1 < len(starts) else len(txt)
            body = txt[s:e].strip()
            title = titles[i] if i < len(titles) else f"Chapter_{i+1}"
            if len(body.split()) < 120 and chapters:
                prev_title, prev_body = chapters[-1]
                chapters[-1] = (prev_title, prev_body + "\n\n" + body)
            else:
                chapters.append((title, body))
        if chapters:
            return chapters
    lines = txt.split("\n")
    caps_indices = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if len(s) > 3 and s.upper() == s and sum(c.isalpha() for c in s) >= 3:
            caps_indices.append(idx)
    if caps_indices:
        caps_indices.append(len(lines))
        chapters = []
        last = 0
        for idx in caps_indices:
            title = lines[idx].strip()
            next_idx = caps_indices[caps_indices.index(idx)+1] if caps_indices.index(idx)+1 < len(caps_indices) else len(lines)
            body_lines = lines[idx+1:next_idx]
            body = "\n".join(body_lines).strip()
            if len(body.split()) < 120 and chapters:
                prev_title, prev_body = chapters[-1]
                chapters[-1] = (prev_title, prev_body + "\n\n" + body)
            else:
                chapters.append((title or f"Section_{idx}", body))
        if chapters:
            return chapters
    paras = [p.strip() for p in txt.split("\n\n") if len(p.strip()) > 200]
    if paras:
        return [(f"Section {i+1}", paras[i]) for i in range(len(paras))]
    return [("Document", txt.strip())]

def chunk_text(text: str, max_words: int = 600, overlap: int = 120) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    L = len(words)
    while start < L:
        end = min(L, start + max_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

# ---------------------------
# Local fallback summarizer (small model)
# ---------------------------
_local_tokenizer = None
_local_model = None
def _load_local_fallback():
    global _local_tokenizer, _local_model
    if _local_model is None:
        try:
            logger.info("Loading local fallback model: %s", LOCAL_FALLBACK_MODEL)
            _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_FALLBACK_MODEL)
            _local_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_FALLBACK_MODEL).to(device)
            _local_model.eval()
        except Exception as e:
            logger.exception("Failed to load local fallback: %s", e)
            _local_tokenizer = None
            _local_model = None

def local_summarize(text: str, max_len: int = 150, min_len: int = 40) -> str:
    _load_local_fallback()
    if _local_model is None or _local_tokenizer is None:
        return text.strip()[: min(len(text), max_len*4)]
    try:
        inputs = _local_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        out = _local_model.generate(**inputs, max_length=max_len, min_length=min_len, num_beams=4)
        return _local_tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        logger.exception("Local summarizer failed: %s", e)
        return text.strip()[: min(len(text), max_len*4)]

# ---------------------------
# HF Mistral wrapper (remote inference)
# ---------------------------
def call_hf_mistral(prompt: str, hf_token: str, max_new_tokens: int = 256) -> str:
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN not set; cannot call HF Inference API.")
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 0.95,
            "repetition_penalty": 1.05
        },
        "options": {"wait_for_model": True}
    }
    try:
        r = requests.post(HF_MISTRAL_API, headers=headers, json=payload, timeout=HF_TIMEOUT)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, dict) and "generated_text" in out:
            return out["generated_text"].strip()
        if isinstance(out, list) and len(out) and isinstance(out[0], dict):
            for k in ("generated_text", "summary_text", "text"):
                if k in out[0]:
                    return out[0][k].strip()
            return str(out[0])
        return str(out)
    except Exception as e:
        logger.exception("HF Mistral API call failed: %s", e)
        raise

def summarize_chunk(chunk_text_str: str, max_tokens: int = 220, hf_token: Optional[str]=None, offline: bool=False) -> str:
    prompt = (
        "You are an expert summarizer. Summarize the following text into a short, factual summary.\n\n"
        f"TEXT:\n{chunk_text_str}\n\nSUMMARY:"
    )
    if not offline and hf_token:
        try:
            return call_hf_mistral(prompt, hf_token, max_new_tokens=max_tokens)
        except Exception:
            logger.warning("HF call failed; using local fallback summarizer.")
    return local_summarize(chunk_text_str, max_len=max_tokens)

def merge_summaries(summaries: List[str], hf_token: Optional[str]=None, polish: bool = True, offline: bool=False) -> str:
    if not summaries:
        return ""
    if len(summaries) == 1:
        return summaries[0].strip()
    merged = "\n".join(summaries)
    if not polish:
        return merged
    polish_prompt = (
        "You are an expert editor. Combine the following partial summaries into one clear, coherent, and concise chapter summary. "
        "Remove repetition and keep important facts.\n\n"
        f"{merged}\n\nFinal summary:"
    )
    if not offline and hf_token:
        try:
            return call_hf_mistral(polish_prompt, hf_token, max_new_tokens=300)
        except Exception:
            logger.warning("HF polish failed; using unpolished merge.")
            return merged
    return local_summarize(polish_prompt, max_len=300)

# ---------------------------
# Main summarize_document_pipeline
# ---------------------------
def summarize_document_pipeline(uploaded_obj, pasted_text: str = "", precision_label: str = "Medium",
                                include_tables: bool = True, polish: bool = True,
                                hf_token: Optional[str] = None, offline: bool = False) -> Tuple[str, List[Tuple[str,str]]]:
    raw_bytes = None
    fname = "pasted_input"
    try:
        if uploaded_obj is not None:
            raw_bytes, fname = _read_bytes_from_uploaded(uploaded_obj)
            text, _ = extract_text(uploaded_obj)
        elif pasted_text and pasted_text.strip():
            text = pasted_text
        else:
            raise RuntimeError("No input provided (file or pasted text).")
    except Exception as e:
        logger.exception("Input read failed: %s", e)
        raise RuntimeError(f"Input read failed: {e}") from e

    text = (text or "").strip()
    if not text:
        raise RuntimeError("No textual content extracted from input.")

    total_words = len(text.split())
    ratio = PRECISION_RATIO.get(precision_label, 0.10) if isinstance(PRECISION_RATIO, dict) else 0.10
    total_budget_words = max(200, int(total_words * ratio))

    page_table_sentences: Dict[int, List[str]] = {}
    if include_tables and raw_bytes is not None:
        try:
            tables = extract_tables_from_pdf_bytes(raw_bytes) or []
            for page_no, rows in tables:
                if not rows or not isinstance(rows, list):
                    continue
                header = None
                body_rows = rows
                if len(rows) >= 2:
                    try:
                        if is_header_row(rows[0], rows[1]):
                            header = rows[0]; body_rows = rows[1:]
                    except Exception:
                        header = None; body_rows = rows
                sentences = table_rows_to_sentences(header, body_rows)
                if sentences:
                    page_table_sentences.setdefault(page_no, []).extend(sentences)
        except Exception as e:
            logger.exception("Table extraction step failed: %s", e)

    all_table_sentences = []
    for p, sents in page_table_sentences.items():
        if isinstance(sents, list):
            all_table_sentences.extend([f"(Table p{p}) {s}" for s in sents])

    try:
        chapters = split_into_chapters(text) or [("Document", text)]
    except Exception as e:
        logger.exception("Chapter splitting failed — using whole document: %s", e)
        chapters = [("Document", text)]

    sections: List[Tuple[str, str]] = []
    for idx, (title, body) in enumerate(chapters, start=1):
        if not body or len(body.strip()) < 20:
            if sections and body and body.strip():
                prev_title, prev_body = sections[-1]
                sections[-1] = (prev_title, prev_body + "\n\n" + body.strip())
            continue

        approx_chunk_words = max(400, int(total_budget_words))
        approx_chunk_words = min(1200, approx_chunk_words)
        overlap_words = max(80, int(approx_chunk_words * 0.12))

        try:
            chunks = chunk_text(body, max_words=approx_chunk_words, overlap=overlap_words) or [body]
        except Exception as e:
            logger.exception("Chunking failed for chapter '%s': %s", title, e)
            chunks = [body]

        if not chunks:
            chunks = [body]

        chunk_summaries = []
        for ch in chunks:
            if not ch or not ch.strip():
                continue
            try:
                per_chunk_tokens = max(120, int(total_budget_words / max(1, len(chunks)) * 0.9))
                s = summarize_chunk(ch, max_tokens=per_chunk_tokens, hf_token=hf_token, offline=offline)
            except Exception as e:
                logger.exception("Chunk summarization failed for chapter '%s': %s", title, e)
                s = ch[:min(len(ch), 500)]
            if s and s.strip():
                chunk_summaries.append(s)

        if not chunk_summaries:
            try:
                fallback = summarize_chunk(body, max_tokens=max(200, int(total_budget_words*0.3)), hf_token=hf_token, offline=offline)
                chunk_summaries = [fallback] if fallback else [body[:500]]
            except Exception as e:
                logger.exception("Fallback summarization failed: %s", e)
                chunk_summaries = [body[:500]]

        try:
            merged = merge_summaries(chunk_summaries, hf_token=hf_token, polish=polish, offline=offline)
        except Exception as e:
            logger.exception("Merging summaries failed for '%s': %s", title, e)
            merged = "\n\n".join(chunk_summaries)

        related_tables = []
        if all_table_sentences:
            body_lower = body.lower()
            for ts in all_table_sentences:
                try:
                    first_words = " ".join(ts.split()[:6]).lower()
                    if len(first_words) > 3 and (first_words in body_lower or any(w in body_lower for w in first_words.split() if len(w) > 3)):
                        related_tables.append(ts)
                except Exception:
                    continue
            if related_tables:
                merged += "\n\nTables (related):\n" + "\n".join(related_tables[:10])

        sections.append((title, merged))

    if all_table_sentences and not any("Tables (" in sec for _, sec in sections):
        table_block = "\n\nTables (Document-level):\n" + "\n".join(all_table_sentences[:25])
        sections.append(("Tables (Document-level)", table_block))

    final_blocks = []
    for t, s in sections:
        final_blocks.append(f"## {t}\n\n{s}\n")
    final_summary_text = "\n".join(final_blocks).strip()

    # Save defensively
    try:
        out_fname = make_output_filename_preserve(fname, precision_label)
        header = f"Source File: {fname}\nModel: Mistral (HF or local fallback)\nPrecision: {precision_label}\nGenerated: {datetime.datetime.now().isoformat()}\n\n"
        os.makedirs(os.path.dirname(out_fname) or ".", exist_ok=True)
        atomic_write_text(out_fname, header + final_summary_text)
        logger.info("Saved summary to %s", out_fname)
    except Exception as e:
        logger.exception("Failed to save summary file: %s", e)
        out_fname = None

    return final_summary_text, sections

# ---------------------------
# Gradio UI wrapper & wiring (uses choose_mode_preference)
# ---------------------------
def process_summary_and_save(uploaded_file, model_choice, precision_choice, compute_metrics_flag, reference_text, hf_token=None, offline=False):
    try:
        hf_token_local = None if offline else hf_token
        final_summary_text, sections = summarize_document_pipeline(
            uploaded_file, pasted_text="", precision_label=precision_choice, include_tables=True, polish=True, hf_token=hf_token_local, offline=offline
        )
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return "", None, f"Error: pipeline failed: {e}"

    out_fname = None
    try:
        src_name = "pasted_input"
        try:
            if uploaded_file:
                _, src_name = _read_bytes_from_uploaded(uploaded_file)
        except Exception:
            pass
        candidate = make_output_filename_preserve(src_name, precision_choice)
        out_dir = os.path.dirname(candidate) or "."
        os.makedirs(out_dir, exist_ok=True)
        try:
            atomic_write_text(candidate, f"Source File: {src_name}\nModel: {model_choice}\nPrecision: {precision_choice}\nGenerated: {datetime.datetime.now().isoformat()}\n\n" + final_summary_text)
            out_fname = os.path.abspath(candidate)
            if not os.path.exists(out_fname):
                raise RuntimeError("Saved file missing after atomic_write_text.")
        except Exception as e:
            logger.exception("atomic_write_text failed, fallback: %s", e)
            tmp = tempfile.NamedTemporaryFile(delete=False, prefix="defenseint_", suffix=".txt", mode="w", encoding="utf-8")
            tmp.write(f"Source File: {src_name}\nModel: {model_choice}\nPrecision: {precision_choice}\nGenerated: {datetime.datetime.now().isoformat()}\n\n")
            tmp.write(final_summary_text)
            tmp.flush(); tmp.close()
            out_fname = os.path.abspath(tmp.name)
            if not os.path.exists(out_fname):
                out_fname = None
    except Exception as e:
        logger.exception("Saving summary completely failed: %s", e)
        out_fname = None

    status_parts = []
    if out_fname:
        status_parts.append(f"Done. Saved: {out_fname}")
    else:
        status_parts.append("Done. (saving failed) — check logs for details.")

    if compute_metrics_flag and reference_text and reference_text.strip():
        try:
            metrics = compute_word_token_bertscore(final_summary_text, reference_text.strip(), lang="en")
            metrics_text = format_metrics_for_status(metrics)
            status_parts.append("Metrics:\n" + metrics_text)
        except Exception as e:
            logger.exception("Metrics computation failed: %s", e)
            status_parts.append(f"Metrics failed: {e}")

    status_msg = "\n".join(status_parts)
    gr_file_path = out_fname if out_fname and os.path.exists(out_fname) else None
    return final_summary_text, gr_file_path, status_msg

def start_gradio(hf_token=None, offline=False):
    with gr.Blocks(title="DefenseInt Summarizer — Local / Hybrid") as app:
        gr.Markdown("## DefenseInt Summarizer — Hybrid (Mistral online + DistilBART offline)")
        with gr.Row():
            file_input = gr.File(label="1) Browse file (PDF / DOCX / TXT)", file_types=[".pdf", ".docx", ".txt"])
            file_name_display = gr.Textbox(label="File name (preserved)", interactive=False)
        with gr.Row():
            model_select = gr.Dropdown(label="Mode", choices=["Auto (Recommended)", "Online (HF Mistral)", "Offline (Local fallback)"], value="Auto (Recommended)")
            precision_select = gr.Dropdown(label="Precision", choices=["Low","Medium","High"], value="Medium")
        with gr.Row():
            summarize_btn = gr.Button("Summarize")
            status_box = gr.Textbox(label="Status / Logs", interactive=False)
        with gr.Row():
            summary_out = gr.Textbox(label="Summary Output", lines=18, interactive=False)
        with gr.Row():
            metrics_checkbox = gr.Checkbox(label="Compute metrics (simple)", value=False)
            metrics_ref = gr.Textbox(label="Reference text for metrics (optional)", lines=3)
        with gr.Row():
            download_file = gr.File(label="Download saved .txt", interactive=False)

        def _show_filename(uploaded):
            if not uploaded:
                return ""
            if isinstance(uploaded, dict):
                return uploaded.get("name") or uploaded.get("filename") or ""
            if hasattr(uploaded, "name"):
                return os.path.basename(uploaded.name)
            return str(uploaded)

        file_input.change(fn=_show_filename, inputs=file_input, outputs=file_name_display)

        def _model_choice_to_offline(selected, global_offline, hf_token_local):
            # Determine effective mode using choose_mode_preference:
            try:
                effective = choose_mode_preference(selected, hf_token_local, force_offline_flag=global_offline)
                return True if effective == "offline" else False
            except Exception as e:
                logger.exception("Mode resolution failed: %s", e)
                return True

        # wrap click to inject hf_token / offline using the hybrid decision
        def _click(uploaded_file, model_choice, precision_choice, metrics_checkbox_val, metrics_ref_val):
            # Resolve effective mode now
            hf = hf_token
            offline_flag = _model_choice_to_offline(model_choice, offline, hf)
            # If we attempted online but had to fallback, inform user in status (process_summary will log)
            return process_summary_and_save(uploaded_file, model_choice, precision_choice, metrics_checkbox_val, metrics_ref_val, hf_token=hf, offline=offline_flag)

        summarize_btn.click(fn=_click, inputs=[file_input, model_select, precision_select, metrics_checkbox, metrics_ref], outputs=[summary_out, download_file, status_box])

    # keep the server running and bind to localhost; don't use prevent_thread_lock in scripts
        app.launch(share=False, debug=False, prevent_thread_lock=False)
# OR simply:
# app.launch(share=False, debug=False)

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="DefenseInt Summarizer — local hybrid runner")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_API_TOKEN"), help="HuggingFace API token (optional)")
    parser.add_argument("--offline", action="store_true", help="Force offline mode (use local fallback only)")
    parser.add_argument("--no-gui", action="store_true", help="Do not start Gradio UI; exit (useful if embedding)")
    args = parser.parse_args()
    hf_token = args.hf_token
    # If user didn't pass hf token, we still may run in online if env var set; but decide offline if none
    # Check internet and decide if online is possible
    internet_ok = has_internet()
    if hf_token and not internet_ok:
        logger.warning("HF token provided but internet unreachable. Falling back to offline mode.")
    offline = args.offline or (hf_token is None) or (not internet_ok)

    print(f"[INFO] Device: {device}  |  HF token provided: {'yes' if hf_token else 'no'}  |  internet: {'yes' if internet_ok else 'no'}  |  offline={offline}")
    logger.info("Starting app; offline=%s internet=%s hf_token=%s", offline, internet_ok, "yes" if hf_token else "no")
    if args.no_gui:
        print("No GUI mode: nothing to do. (You can call summarize_document_pipeline programmatically.)")
        return
    start_gradio(hf_token=hf_token, offline=offline)

if __name__ == "__main__":
    main()
