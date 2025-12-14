# srd_engine_v2.py
import os
import re
import io
import base64
import hashlib
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# -------------------- Data Processing --------------------
import pdfplumber
import camelot
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract
from PIL import Image

# -------------------- Vector Store --------------------
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# -------------------- Claude --------------------
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from srd_engine_final import SRDChatbotEngine, ClaudeAnswerer

POPPLER_PATH = os.getenv("POPPLER_PATH")


# =====================================================
# UTILS
# =====================================================
def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def resize_for_claude(image: Image.Image, max_dim: int = 7900) -> Image.Image:
    w, h = image.size
    if w <= max_dim and h <= max_dim:
        return image
    scale = min(max_dim / w, max_dim / h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# =====================================================
# SECTION / HEADER DETECTION
# =====================================================
SECTION_PATTERNS = {
    "functional": re.compile(r"(functional\s+requirements|FR-\d+)", re.I),
    "nonfunctional": re.compile(r"(non[-\s]?functional\s+requirements|NFR-\d+)", re.I),
}


def detect_section_type(text: str) -> str:
    for k, pat in SECTION_PATTERNS.items():
        if pat.search(text):
            return k
    return "general"


# =====================================================
# SRD-AWARE SPLITTER (REQUIREMENT SAFE)
# =====================================================
class SmartSRDSplitter:
    HEADER_REGEX = re.compile(
        r"(FR-\d+|NFR-\d+|\d+\.\d+|[A-Z][A-Za-z\s]{3,}:)",
        re.I
    )

    def split_text(self, text: str) -> List[Document]:
        docs: List[Document] = []
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        buffer: List[str] = []
        current_header = "General"

        for line in lines:
            if self.HEADER_REGEX.match(line):
                if buffer:
                    content = "\n".join(buffer)
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "type": "text",
                                "header": current_header,
                                "section_type": detect_section_type(content),
                            },
                        )
                    )
                buffer = [line]
                current_header = line[:80]
            else:
                buffer.append(line)

        if buffer:
            content = "\n".join(buffer)
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "type": "text",
                        "header": current_header,
                        "section_type": detect_section_type(content),
                    },
                )
            )

        return docs


# =====================================================
# DIAGRAM INTERPRETER
# =====================================================
class DiagramInterpreter:
    def __init__(self):
        self._anthropic = None

    def process_image(
        self,
        image: Image.Image,
        label: str,
        use_qwen: bool,
        use_claude: bool
    ) -> str:
        sections: List[str] = []

        if use_claude:
            if Anthropic is None:
                sections.append("Claude Vision requested but anthropic package is not installed.")
            else:
                if not self._anthropic:
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if not api_key:
                        sections.append("Claude Vision requested but ANTHROPIC_API_KEY is not set.")
                    else:
                        self._anthropic = Anthropic(api_key=api_key)

                if self._anthropic:
                    safe_image = resize_for_claude(image)
                    buf = io.BytesIO()
                    safe_image.convert("RGB").save(buf, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode()

                    resp = self._anthropic.messages.create(
                        model=os.getenv("CLAUDE_VISION_MODEL", "claude-sonnet-4-5-20250929"),
                        max_tokens=600,
                        temperature=0.2,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"Explain this {label} diagram for an SRD."},
                                    {
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                                    },
                                ],
                            }
                        ],
                    )

                    try:
                        text0 = resp.content[0].text  # type: ignore[attr-defined]
                    except Exception:
                        text0 = ""
                        for block in getattr(resp, "content", []):
                            t = getattr(block, "text", None)
                            if t:
                                text0 += t + "\n"
                        text0 = text0.strip()

                    if text0:
                        sections.append(text0)

        if not sections:
            sections.append(pytesseract.image_to_string(image))

        return "\n\n".join([s for s in sections if s.strip()]).strip()


# =====================================================
# SMART KNOWLEDGE BASE (MULTI-USER + MULTI-CHAT SAFE)
# =====================================================
class SmartKnowledgeBase(SRDChatbotEngine):
    def __init__(self, chroma_dir="chroma_global_db"):
        super().__init__(chroma_dir)
        self.current_project_id: Optional[str] = None
        self.current_chat_id: Optional[str] = None   # ✅ NEW
        self.current_user_id: Optional[str] = None   # ✅ NEW

        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=self.embedding_model,
            collection_name="srd_knowledge"
        )

        self.interpreter = DiagramInterpreter()

    # ------------------------------
    # SESSION SCOPING
    # ------------------------------
    def set_current_project(self, name: str):
        self.current_project_id = name.lower().replace(" ", "_")

    def set_current_chat(self, chat_id: str):
        self.current_chat_id = chat_id

    def set_current_user(self, user_id: str):
        self.current_user_id = user_id

    def _require_scope(self):
        if not self.current_project_id:
            raise RuntimeError("Project not set. Call set_current_project(...) first.")
        if not self.current_chat_id:
            raise RuntimeError("Chat not set. Call set_current_chat(...) first.")
        if not self.current_user_id:
            raise RuntimeError("User not set. Call set_current_user(...) first.")

    def _where_scope(self) -> dict:
        # Chroma where filter (strict isolation)
        return {
            "$and": [
                {"project_id": {"$eq": self.current_project_id}},
                {"chat_id": {"$eq": self.current_chat_id}},
                {"user_id": {"$eq": self.current_user_id}},
            ]
        }

    # ------------------------------
    # LEARNING / USER CORRECTION
    # ------------------------------
    def learn_from_interaction(self, query: str, correction_text: str):
        self._require_scope()

        if not correction_text or not correction_text.strip():
            return

        inferred = detect_section_type(correction_text)
        if inferred == "general":
            inferred = self.detect_requirement_type(query)

        doc = Document(
            page_content=correction_text.strip(),
            metadata={
                "type": "user_correction",
                "section_type": inferred,
                "project_id": self.current_project_id,
                "chat_id": self.current_chat_id,
                "user_id": self.current_user_id,
                "source": "user_feedback",
                "timestamp": datetime.now().isoformat(),
                "original_query": query,
                "priority": "high",
            },
        )

        self.vectorstore.add_documents([doc])
        self.vectorstore.persist()

    # ------------------------------
    # INGESTION
    # ------------------------------
    def process_document_step(self, path, ftype, label, use_qwen, use_claude):
        self._require_scope()

        docs: List[Document] = []

        if ftype == "pdf_text":
            with pdfplumber.open(path) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)

            splitter = SmartSRDSplitter()
            docs = splitter.split_text(text)

            try:
                tables = camelot.read_pdf(path, pages="all", flavor="stream")
                for t in tables:
                    docs.append(
                        Document(
                            page_content=t.df.to_markdown(),
                            metadata={"type": "table", "section_type": "general"},
                        )
                    )
            except Exception:
                pass

        elif ftype == "diagram":
            if path.lower().endswith(".pdf"):
                info = pdfinfo_from_path(path, poppler_path=POPPLER_PATH)
                for page in range(1, info["Pages"] + 1):
                    imgs = convert_from_path(
                        path,
                        first_page=page,
                        last_page=page,
                        dpi=150,
                        poppler_path=POPPLER_PATH,
                    )
                    for img in imgs:
                        txt = self.interpreter.process_image(img, label, use_qwen, use_claude)
                        docs.append(Document(page_content=txt, metadata={"type": "diagram", "section_type": "general"}))
            else:
                img = Image.open(path)
                txt = self.interpreter.process_image(img, label, use_qwen, use_claude)
                docs.append(Document(page_content=txt, metadata={"type": "diagram", "section_type": "general"}))

        # ------------------------------
        # Dedup + metadata
        # ------------------------------
        seen = set()
        final_docs: List[Document] = []

        for d in docs:
            h = content_hash(d.page_content or "")
            if h in seen:
                continue
            seen.add(h)

            d.metadata["project_id"] = self.current_project_id
            d.metadata["chat_id"] = self.current_chat_id
            d.metadata["user_id"] = self.current_user_id
            d.metadata["timestamp"] = datetime.now().isoformat()

            final_docs.append(d)

        if final_docs:
            self.vectorstore.add_documents(final_docs)
            self.vectorstore.persist()

        return final_docs

    # ------------------------------
    # INTENT DETECTION
    # ------------------------------
    def detect_intent(self, query: str) -> str:
        q = (query or "").lower()
        if any(w in q for w in ["list", "show all", "enumerate", "give me all", "all of the"]):
            return "enumeration"
        if any(w in q for w in ["explain", "describe", "how", "why", "what is", "what are"]):
            return "explanation"
        return "lookup"

    # ------------------------------
    # REQUIREMENT TYPE DETECTION
    # ------------------------------
    def detect_requirement_type(self, query: str) -> str:
        q = (query or "").lower()

        if any(w in q for w in [
            "non functional", "non-functional", "nonfunctional", "nfr", "nfrs",
            "quality attributes", "quality requirements"
        ]):
            return "nonfunctional"

        if any(w in q for w in [
            "performance", "security", "availability", "reliability", "scalability",
            "usability", "maintainability", "portability", "compliance", "privacy",
            "latency", "throughput", "encryption", "audit", "logging", "backup",
        ]):
            return "nonfunctional"

        if any(w in q for w in ["functional", "fr-", "frs", "use case", "features"]):
            return "functional"

        return "functional"

    # ------------------------------
    # SMART RESPONSE (CHAT-ISOLATED)
    # ------------------------------
    def generate_smart_response(self, query: str, claude: ClaudeAnswerer) -> str:
        self._require_scope()

        intent = self.detect_intent(query)

        # =============== ENUMERATION MODE ===============
        if intent == "enumeration":
            req_type = self.detect_requirement_type(query)

            raw = self.vectorstore.get(
                where={
                    "$and": [
                        {"project_id": {"$eq": self.current_project_id}},
                        {"chat_id": {"$eq": self.current_chat_id}},
                        {"user_id": {"$eq": self.current_user_id}},
                        {"section_type": {"$eq": req_type}},
                    ]
                }
            )

            docs = raw.get("documents", []) or []

            if not docs and req_type == "nonfunctional":
                raw2 = self.vectorstore.get(
                    where={
                        "$and": [
                            {"project_id": {"$eq": self.current_project_id}},
                            {"chat_id": {"$eq": self.current_chat_id}},
                            {"user_id": {"$eq": self.current_user_id}},
                            {"section_type": {"$eq": "general"}},
                        ]
                    }
                )
                docs2 = raw2.get("documents", []) or []
                if docs2:
                    docs = docs2

            if not docs:
                return "I could not find sufficient information in the provided SRD."

            title = "FUNCTIONAL REQUIREMENTS" if req_type == "functional" else "NON-FUNCTIONAL REQUIREMENTS"

            prompt = f"""
You are a Senior Project Architect.

Return a COMPLETE numbered list of the {title} found below.
Do NOT invent items. Do NOT omit items. If duplicates exist, keep only one copy.

REQUIREMENTS:
{chr(10).join(docs)}
"""
            return claude.client.messages.create(
                model=claude.model,
                max_tokens=1400,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            ).content[0].text

        # =============== NORMAL QA MODE ===============
        docs = self.vectorstore.similarity_search(
            query,
            k=12,
            filter=self._where_scope(),  # ✅ chat + user + project scoped
        )

        if not docs:
            return "I could not find sufficient information in the provided SRD."

        ctx = ""
        for d in docs[:8]:
            ctx += f"[{d.metadata.get('header', 'SRD')}]\n{d.page_content}\n---\n"

        prompt = f"""
You are a Senior Project Architect.

Answer ONLY using the SRD context.
If unsupported, say so explicitly.

CONTEXT:
{ctx}

QUESTION:
{query}
"""
        return claude.client.messages.create(
            model=claude.model,
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        ).content[0].text
