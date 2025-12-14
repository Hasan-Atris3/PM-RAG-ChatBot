# srd_engine_final.py
# ============================================================
# CedroPass SRD – Final RAG Engine (Stable, Section-Aware)
# ============================================================

import os
import re
import io
import base64
import time
import shutil
import warnings
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

# -------------------- Data Processing --------------------
import pdfplumber
import camelot
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# -------------------- NLP & Retrieval --------------------
import spacy
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from rapidfuzz import process as fuzz_process

# -------------------- Claude --------------------
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# -------------------- CONFIG --------------------
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

POPPLER_PATH = os.getenv("POPPLER_PATH")
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# -------------------- NLP MODEL --------------------
print("[SYSTEM] Loading NLP pipelines...")
try:
    NLP_EN = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    NLP_EN = spacy.load("en_core_web_sm")


# ============================================================
# TEXT UTILS
# ============================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def lemmatize_text(text: str) -> str:
    doc = NLP_EN(text[:50000])
    return " ".join(
        t.lemma_.lower()
        for t in doc
        if not t.is_space and not t.is_punct
    )


# ============================================================
# SECTION-AWARE SRD SPLITTER (CRITICAL FIX)
# ============================================================
class SmartSRDSplitter:
    """
    Guarantees that ALL child paragraphs inherit the correct
    section_type until a new header appears.
    """

    HEADER_REGEX = re.compile(
        r"^(\d+(\.\d+)*|FR-\d+|NFR-\d+|[A-Z][A-Za-z\s]{3,}:)",
        re.IGNORECASE,
    )

    def split_text(self, text: str) -> List[Document]:
        docs: List[Document] = []
        lines = text.splitlines()

        buffer: List[str] = []
        current_section_title = "General"
        current_section_type = "general"

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            if self.HEADER_REGEX.match(line):
                # Flush previous chunk
                if buffer:
                    docs.append(
                        Document(
                            page_content="\n".join(buffer),
                            metadata={
                                "type": "text",
                                "section": current_section_title,
                                "section_type": current_section_type,
                                "source": "SRD_Main",
                            },
                        )
                    )

                buffer = [line]
                current_section_title = line[:80]

                lowered = line.lower()
                if "functional requirement" in lowered or "fr-" in lowered:
                    current_section_type = "functional"
                elif "non-functional" in lowered or "nfr-" in lowered:
                    current_section_type = "nonfunctional"
                else:
                    current_section_type = "general"
            else:
                buffer.append(line)

        # Final flush
        if buffer:
            docs.append(
                Document(
                    page_content="\n".join(buffer),
                    metadata={
                        "type": "text",
                        "section": current_section_title,
                        "section_type": current_section_type,
                        "source": "SRD_Main",
                    },
                )
            )

        return docs


# ============================================================
# PDF EXTRACTORS
# ============================================================
def extract_pdf_text(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
    return text


def extract_tables(path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        tables = camelot.read_pdf(path, pages="all", flavor="stream")
        for i, t in enumerate(tables):
            md = t.df.to_markdown(index=False)
            if len(md) > 30:
                docs.append(
                    Document(
                        page_content=md,
                        metadata={
                            "type": "table",
                            "section_type": "general",
                            "source": "SRD_Table",
                        },
                    )
                )
    except Exception:
        pass
    return docs


# ============================================================
# DIAGRAM INTERPRETER (TEXT-ONLY SAFE)
# ============================================================
class DiagramInterpreter:
    def __init__(self):
        self.client = (
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            if Anthropic and os.getenv("ANTHROPIC_API_KEY")
            else None
        )

    def describe(self, image: Image.Image, label: str) -> str:
        if not self.client:
            return pytesseract.image_to_string(image)

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=600,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Explain this {label} diagram for an SRD."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                    ],
                }
            ],
        )
        return resp.content[0].text


# ============================================================
# CORE RAG ENGINE
# ============================================================
class SRDChatbotEngine:
    def __init__(self, chroma_dir: str = "chroma_db_final"):
        print("[ENGINE] Initializing retrievers...")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.chroma_dir = chroma_dir
        self.vectorstore: Optional[Chroma] = None
        self.chroma_retriever = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.vocab = set()

    # -------------------- BUILD INDEX --------------------
    def build_index(
        self,
        pdf_path: str,
        diagrams: Optional[List[str]] = None,
    ):
        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)

        splitter = SmartSRDSplitter()
        docs = splitter.split_text(extract_pdf_text(pdf_path))
        docs.extend(extract_tables(pdf_path))

        for d in docs:
            d.metadata["lemma"] = lemmatize_text(d.page_content)
            for w in d.page_content.split():
                if w.isalnum():
                    self.vocab.add(w.lower())

        self.vectorstore = Chroma.from_documents(
            docs,
            embedding=self.embedding_model,
            persist_directory=self.chroma_dir,
            collection_name="srd_final",
        )

        self.chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 20

        print(f"✅ Indexed {len(docs)} SRD chunks")

    # -------------------- INTENT --------------------
    def detect_intent(self, q: str) -> str:
        q = q.lower()
        if any(w in q for w in ["list", "enumerate", "all functional", "requirements of"]):
            return "enumeration"
        return "qa"

    # -------------------- ENUMERATION (NO SIM SEARCH) --------------------
    def list_functional_requirements(self) -> List[str]:
        data = self.vectorstore.get(
            where={"section_type": "functional"}
        )
        return data.get("documents", [])

    # -------------------- QUERY --------------------
    def answer(self, query: str, claude) -> str:
        intent = self.detect_intent(query)

        if intent == "enumeration":
            items = self.list_functional_requirements()
            if not items:
                return "I could not find sufficient information in the provided SRD."

            prompt = f"""
You are a Senior Project Architect.

List ALL functional requirements below.
Do not merge, summarize, or invent anything.

REQUIREMENTS:
{chr(10).join(items)}
"""
            return claude.generate_raw(prompt)

        # ---------- Normal QA ----------
        dense = self.chroma_retriever.invoke(query)
        sparse = self.bm25_retriever.invoke(query)

        pool = dense + sparse
        pairs = [[query, d.page_content] for d in pool]
        scores = self.reranker.predict(pairs)

        top = [
            d.page_content
            for d, s in sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
            if s > -6
        ][:8]

        if not top:
            return "I could not find sufficient information in the provided SRD."

        ctx = "\n---\n".join(top[:4000])

        prompt = f"""
Answer using ONLY the SRD context below.
If unsupported, say so explicitly.

CONTEXT:
{ctx}

QUESTION:
{query}
"""
        return claude.generate_raw(prompt)


# ============================================================
# CLAUDE ANSWERER
# ============================================================
class ClaudeAnswerer:
    def __init__(self):
        if Anthropic is None:
            raise RuntimeError("anthropic not installed")

        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-5-20250929"

    def generate_raw(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
