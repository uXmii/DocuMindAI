"""
Multimodal Processor - PRODUCTION VERSION
Treats every PDF page as an image and uses a vision LLM to truly understand
content — charts, diagrams, tables, handwriting.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  For each PDF page:                                      │
  │  1. Gemini/GPT-4o DESCRIBES the page → rich text chunk  │
  │  2. CLIP encodes the raw page image → image embedding   │
  │     stored alongside the text embedding in ChromaDB     │
  │                                                         │
  │  At query time:                                         │
  │  - Text query → all-MiniLM embedding  → text chunks    │
  │  - Text query → CLIP text embedding   → image chunks   │
  │  → merge both result sets (true multimodal retrieval)   │
  └─────────────────────────────────────────────────────────┘

Priority order:
  Vision LLM : GEMINI_API_KEY (free) → ANTHROPIC_API_KEY → OPENAI_API_KEY
  CLIP model : sentence-transformers clip-ViT-B-32 (local, no API key needed)
  Fallback   : Tesseract OCR if no vision API key
"""

import os
import io
import re
import base64
import hashlib
from typing import List, Dict, Optional

from PIL import Image
from pdf2image import convert_from_path


# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

# ── CLIP for true multimodal embeddings ───────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    _CLIP_MODEL = _ST("clip-ViT-B-32")
    CLIP_AVAILABLE = True
    print("✓ CLIP multimodal embeddings: clip-ViT-B-32 loaded")
except Exception:
    _CLIP_MODEL = None
    CLIP_AVAILABLE = False
    print("⚠  CLIP not available — install sentence-transformers for true multimodal retrieval")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Vision LLM client wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VisionLLMClient:
    """
    Unified client for vision-capable LLMs.
    Priority: Google Gemini → Anthropic → OpenAI → Tesseract OCR fallback
    """

    def __init__(self):
        self.gemini_key    = os.getenv("GEMINI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key    = os.getenv("OPENAI_API_KEY")

        self.provider = None
        self.client   = None

        if self.gemini_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.gemini_key)
            _m = _init_gemini_model()
            if _m:
                self.client   = _m
                self.provider = "gemini"
                print(f"🔮 Vision LLM: {_m.model_name}")
            else:
                print("⚠   Gemini vision init failed — falling back to OCR")
        elif self.anthropic_key and ANTHROPIC_AVAILABLE:
            self.client   = anthropic.Anthropic(api_key=self.anthropic_key)
            self.provider = "anthropic"
            print("🔮 Vision LLM: Anthropic claude-haiku-4-5-20251001")
        elif self.openai_key and OPENAI_AVAILABLE:
            self.client   = openai.OpenAI(api_key=self.openai_key)
            self.provider = "openai"
            print("🔮 Vision LLM: OpenAI GPT-4o")
        else:
            print("⚠  No vision LLM API key found — falling back to Tesseract OCR")

    @property
    def available(self) -> bool:
        return self.provider is not None

    # ── Image → base64 ────────────────────────────────────────────────────────

    @staticmethod
    def image_to_base64(image: Image.Image, max_dim: int = 1568) -> str:
        """
        Resize to at most max_dim on the longest side (Anthropic recommends ≤1568px)
        and encode as base64 PNG.
        """
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # ── Core describe method ───────────────────────────────────────────────────

    def describe_page(self, image: Image.Image, page_num: int) -> str:
        """
        Send a page image to the vision LLM and get back structured text.
        Returns empty string on failure so callers can fall back gracefully.
        """
        if not self.available:
            return ""

        b64  = self.image_to_base64(image)
        prompt = (
            "You are analyzing a single page from a PDF document. "
            "Your job is to extract ALL information faithfully.\n\n"
            "Instructions:\n"
            "1. Transcribe all visible text exactly as written.\n"
            "2. For tables: describe headers and every row as 'Header: value | Header: value'.\n"
            "3. For charts/graphs: describe the type, axes labels, and key data points/trends.\n"
            "4. For diagrams/figures: describe what is shown, labels, and relationships.\n"
            "5. For equations/formulas: transcribe them using plain text math notation.\n"
            "6. Preserve the logical reading order (top-left to bottom-right).\n"
            "7. Do NOT add commentary or interpretation — only describe what is there.\n\n"
            "Output the extracted content now:"
        )

        try:
            if self.provider == "gemini":
                return self._describe_gemini(image, page_num)
            elif self.provider == "anthropic":
                return self._describe_anthropic(b64, prompt)
            elif self.provider == "openai":
                return self._describe_openai(b64, prompt)
        except Exception as e:
            print(f"   ⚠  Vision LLM error on page {page_num}: {e}")
            return ""

    def _describe_gemini(self, image: Image.Image, page_num: int) -> str:
        """Use Gemini's native PIL image support — no base64 needed."""
        prompt = (
            "You are analyzing a single page from a PDF document. "
            "Your job is to extract ALL information faithfully.\n\n"
            "Instructions:\n"
            "1. Transcribe all visible text exactly as written.\n"
            "2. For tables: describe headers and every row as 'Header: value | Header: value'.\n"
            "3. For charts/graphs: describe the type, axes labels, and key data points/trends.\n"
            "4. For diagrams/figures: describe what is shown, labels, and relationships.\n"
            "5. For equations/formulas: transcribe them using plain text math notation.\n"
            "6. Preserve the logical reading order (top-left to bottom-right).\n"
            "7. Do NOT add commentary or interpretation — only describe what is there.\n\n"
            "Output the extracted content now:"
        )
        response = self.client.generate_content([prompt, image])
        return response.text.strip()

    def _describe_anthropic(self, b64: str, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":   "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":       b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text.strip()

    def _describe_openai(self, b64: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }],
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CLIP encoder — true multimodal embeddings (image ↔ text in same space)
# ─────────────────────────────────────────────────────────────────────────────

class CLIPEncoder:
    """
    Wraps sentence-transformers CLIP to encode both images and text queries
    into the same 512-dim vector space.

    This is what makes retrieval truly multimodal:
    - A text query like "economic activities flowchart" gets a CLIP text embedding
    - The stored page image also has a CLIP image embedding
    - Cosine similarity between them works across modalities
    """

    def __init__(self):
        self.model     = _CLIP_MODEL
        self.available = CLIP_AVAILABLE
        self.dim       = 512   # clip-ViT-B-32 output dimension

    def encode_image(self, image: Image.Image) -> Optional[list]:
        """Encode a PIL image → 512-dim float list, or None if CLIP unavailable."""
        if not self.available:
            return None
        try:
            emb = self.model.encode(image, convert_to_numpy=True)
            return emb.tolist()
        except Exception as e:
            print(f"   ⚠  CLIP image encode error: {e}")
            return None

    def encode_text(self, text: str) -> Optional[list]:
        """Encode a text query → 512-dim float list in the same CLIP space."""
        if not self.available:
            return None
        try:
            emb = self.model.encode(text, convert_to_numpy=True)
            return emb.tolist()
        except Exception as e:
            print(f"   ⚠  CLIP text encode error: {e}")
            return None


# Singleton — shared across the app so the model loads once
CLIP_ENCODER = CLIPEncoder()

def _init_gemini_model(lib=None):
    """
    Find a working Gemini model WITHOUT burning quota.
    Uses list_models() which is free, then picks the best available flash model.
    Falls back to trying known names without probing if list_models fails.
    """
    import google.generativeai as _g
    if lib is None:
        lib = _g

    PREFERRED = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    # Try list_models() first — zero quota cost
    try:
        available = []
        for m in lib.list_models():
            if "generateContent" in m.supported_generation_methods:
                available.append(m.name.replace("models/", ""))
        # Pick best preferred model that's available
        for name in PREFERRED:
            if name in available:
                model = lib.GenerativeModel(name)
                print(f"   ✔  Gemini model selected: {name}")
                return model
        # If none of our preferred list matched, use first available
        if available:
            name = available[0]
            model = lib.GenerativeModel(name)
            print(f"   ✔  Gemini model selected (first available): {name}")
            return model
    except Exception as e:
        print(f"   ⚠  list_models() failed ({e}), trying known names without probe...")

    # Fallback: try names without live probe (no quota used)
    for name in PREFERRED:
        try:
            model = lib.GenerativeModel(name)
            print(f"   ✔  Gemini model assumed: {name} (unverified)")
            return model
        except Exception:
            continue

    print("   ✗  No Gemini model could be initialised")
    return None





# ─────────────────────────────────────────────────────────────────────────────
# Main processor
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalProcessor:
    """
    Production-grade multimodal PDF processor.

    Strategy:
    - Convert every page to an image at 200 dpi
    - If a vision LLM is available, send each page image to it for rich
      semantic understanding (charts, diagrams, tables, handwriting, etc.)
    - If no vision LLM, fall back to Tesseract OCR
    - Additionally attempt Camelot table extraction as a supplementary pass
    """

    def __init__(self):
        self.vision = VisionLLMClient()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_multimodal_document(self, pdf_path: str) -> Dict:
        """
        Process a PDF and return OCR/vision chunks + table chunks.
        Compatible with the existing DocumentProcessor interface.
        """
        filename = os.path.basename(pdf_path)
        results  = {
            "filename":    filename,
            "ocr_chunks":  [],      # renamed but kept for API compat — now "vision chunks"
            "table_chunks": [],
            "vision_used": self.vision.available,
        }

        # ── Step 1: Page-level image processing ───────────────────────────────
        print("📸 Converting PDF pages to images…")
        page_images = self._convert_pages(pdf_path)

        if not page_images:
            print("   ⚠  Could not convert PDF pages — skipping image processing")
        else:
            print(f"   ✔  {len(page_images)} pages converted")
            for page_data in page_images:
                chunk = self._process_page(page_data, filename)
                if chunk:
                    results["ocr_chunks"].append(chunk)

        # ── Step 2: Camelot table extraction (supplementary) ──────────────────
        if CAMELOT_AVAILABLE:
            print("📊 Running Camelot table extraction…")
            table_chunks = self._extract_tables(pdf_path, filename)
            results["table_chunks"].extend(table_chunks)

        print(f"   ✔  {len(results['ocr_chunks'])} vision/OCR chunks")
        print(f"   ✔  {len(results['table_chunks'])} table chunks")
        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _convert_pages(self, pdf_path: str) -> List[Dict]:
        """Convert PDF pages to PIL images at 200 dpi.
        On Windows, poppler must be installed and in PATH.
        Download from: https://github.com/oschwartz10612/poppler-windows/releases
        Extract and add the bin/ folder to your system PATH.
        """
        # Common Windows poppler paths to try automatically
        WIN_POPPLER_PATHS = [
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler-24\Library\bin",
            r"C:\poppler\Library\bin",
            r"C:\tools\poppler\Library\bin",
        ]
        import os, sys

        def try_convert(poppler_path=None):
            kwargs = {"dpi": 200}
            if poppler_path:
                kwargs["poppler_path"] = poppler_path
            return convert_from_path(pdf_path, **kwargs)

        # First try without explicit path (uses system PATH)
        try:
            images = try_convert()
            return [{"page_number": i + 1, "image": img, "size": img.size}
                    for i, img in enumerate(images)]
        except Exception as first_err:
            pass

        # On Windows, try known poppler locations
        if sys.platform == "win32":
            for pp in WIN_POPPLER_PATHS:
                if os.path.isdir(pp):
                    try:
                        images = try_convert(poppler_path=pp)
                        print(f"   ✔  Found poppler at {pp}")
                        return [{"page_number": i + 1, "image": img, "size": img.size}
                                for i, img in enumerate(images)]
                    except Exception:
                        continue

            print(f"   ⚠  pdf2image error: {first_err}")
            print("   ⚠  Poppler not found. To enable vision processing on Windows:")
            print("       1. Download: https://github.com/oschwartz10612/poppler-windows/releases")
            print("       2. Extract to C:\\\\Program Files\\\\poppler\\\\")
            print("       2. Extract to C:\\\\Program Files\\\\poppler\\\\")
            print("   ⚠  Continuing without page-image vision processing…")
        else:
            print(f"   ⚠  pdf2image error: {first_err}")
            print("   ⚠  Install poppler: sudo apt install poppler-utils (Linux) or brew install poppler (Mac)")
        return []

    def _process_page(self, page_data: Dict, filename: str) -> Optional[Dict]:
        """
        Extract text from one page — vision LLM if available, else Tesseract.
        Returns a chunk dict or None if nothing useful was extracted.
        """
        page_num = page_data["page_number"]
        image    = page_data["image"]

        # Vision LLM path
        if self.vision.available:
            print(f"   🔮 Vision LLM processing page {page_num}…", end=" ")
            text = self.vision.describe_page(image, page_num)
            method = "vision_llm"        # Tesseract fallback
        elif TESSERACT_AVAILABLE:
            print(f"   🔤 OCR processing page {page_num}…", end=" ")
            try:
                text = pytesseract.image_to_string(image)
                text = self._clean_ocr_text(text)
                method = "tesseract"
            except Exception as e:
                print(f"FAIL ({e})")
                return None
        else:
            return None

        text = (text or "").strip()
        if len(text) < 50:
            print("(too short, skipping)")
            return None

        print(f"✔  {len(text)} chars")
        chunk_hash = hashlib.md5(text.encode()).hexdigest()[:8]

        # ── CLIP image embedding (true multimodal retrieval) ──────────────────
        clip_embedding = CLIP_ENCODER.encode_image(image)

        return {
            "id":   f"vision_{page_num}_{chunk_hash}",
            "text": text,
            "clip_embedding": clip_embedding,   # None if CLIP unavailable
            "metadata": {
                "source":           filename,
                "page":             page_num,
                "type":             method,    # 'vision_llm' | 'tesseract'
                "image_size":       page_data["size"],
                "has_clip_embedding": clip_embedding is not None,
            },
        }

    # ── Table extraction ───────────────────────────────────────────────────────

    def _extract_tables(self, pdf_path: str, filename: str) -> List[Dict]:
        """Extract tables using Camelot (lattice then stream fallback)."""
        chunks = []
        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
                for idx, table in enumerate(tables):
                    text = self._table_to_text(table.df)
                    if text and len(text) > 30:
                        chunk_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                        chunks.append({
                            "id":   f"table_{idx}_{chunk_hash}",
                            "text": text,
                            "metadata": {
                                "source":  filename,
                                "page":    table.page,
                                "type":    "table",
                                "rows":    len(table.df),
                                "cols":    len(table.df.columns),
                                "flavor":  flavor,
                            },
                        })
                if chunks:
                    print(f"   ✔  {len(chunks)} tables via {flavor} mode")
                    return chunks
            except Exception:
                continue
        return chunks

    # ── Text cleaning ──────────────────────────────────────────────────────────

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Clean Tesseract OCR output."""
        if not text:
            return text
        text = re.sub(r"\s*\|\s*", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b([A-Z])\s+\1(\s+\1)+\b", r"\1", text)
        text = re.sub(r"\b([A-Z])\s+\1\b", r"\1", text)
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", text)
        lines = []
        for line in text.split("\n"):
            words = line.split()
            if not words:
                continue
            singles = sum(1 for w in words if len(w) == 1 and w.isalpha())
            if singles / len(words) < 0.5:
                lines.append(line)
        text = "\n".join(lines)
        text = "".join(c for c in text if c.isprintable() or c == "\n")
        return re.sub(r"\s+", " ", text).strip()

    # ── Table → text ──────────────────────────────────────────────────────────

    def _table_to_text(self, df) -> str:
        """Convert a pandas DataFrame (from Camelot) to readable prose text."""
        import pandas as pd

        if df.empty:
            return ""

        df = df.map(lambda x: str(x).strip())
        df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
        df = df.fillna("")

        if df.empty:
            return ""

        rows       = df.values.tolist()
        first_row  = [str(v).strip() for v in rows[0]]

        # Detect header row
        has_header = (
            len(set(first_row)) == len(first_row)
            and all(len(v) < 40 for v in first_row if v)
            and not any(v[0].isdigit() for v in first_row if v)
        )
        headers   = first_row           if has_header else [f"Column {i+1}" for i in range(len(rows[0]))]
        data_rows = rows[1:]            if has_header else rows

        parts = []
        non_empty_headers = [h for h in headers if h]
        if non_empty_headers:
            parts.append(f"Table columns: {' | '.join(non_empty_headers)}\n")

        for row in data_rows:
            cells     = [str(v).strip() for v in row]
            non_empty = [c for c in cells if c]
            if len(non_empty) < 2:
                continue
            entry = " | ".join(
                f"{h}: {c}" if h else c
                for h, c in zip(headers, cells) if c
            )
            if entry:
                parts.append(entry)

        return "\n".join(parts).strip()

    # ── Legacy compat ──────────────────────────────────────────────────────────

    def describe_image_content(self, image: Image.Image) -> str:
        """Legacy shim kept for backward compatibility."""
        w, h = image.size
        return f"Image: {w}x{h} pixels"