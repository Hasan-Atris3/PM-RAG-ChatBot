# PM-RAG-ChatBot

A powerful RAG (Retrieval Augmented Generation) chatbot for querying Software Requirements Documents (SRDs) and project documentation. Built with Chainlit, Claude AI, and advanced document processing capabilities.

## 🚀 Features

- **📄 SRD Document Processing**: Upload and index PDF documents with intelligent section-aware text splitting
- **🎨 Diagram Understanding**: Process diagrams using Claude Vision and/or Qwen2-VL vision models
- **💬 Interactive Chat Interface**: Web-based chat interface powered by Chainlit
- **🔍 Hybrid Search**: Combines dense vector search (semantic) and sparse search (BM25) with cross-encoder reranking
- **📚 Multi-Project Support**: Manage multiple projects with isolated knowledge bases
- **💾 Persistent Chat History**: SQLite database stores all conversations and messages
- **🧠 Learning from Feedback**: Optional learning mode that improves responses based on user corrections
- **🔐 User Isolation**: Multi-user support with strict data isolation per user, project, and chat
- **⚡ Smart Intent Detection**: Automatically detects enumeration queries vs. regular Q&A
- **📊 Table Extraction**: Extracts and indexes tables from PDF documents

## 🏗️ Architecture

The system uses a hybrid RAG architecture:

1. **Document Ingestion**: PDFs are processed with section-aware splitting, preserving functional/non-functional requirement context
2. **Vector Storage**: ChromaDB stores embeddings with metadata for filtering and scoping
3. **Hybrid Retrieval**: 
   - Dense retrieval using sentence transformers (all-MiniLM-L6-v2)
   - Sparse retrieval using BM25
   - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
4. **Answer Generation**: Claude Sonnet 4.5 generates context-aware answers
5. **Vision Processing**: Optional diagram interpretation using Claude Vision or OCR

## 📋 Prerequisites

- **Python 3.8+**
- **System Dependencies**:
  - **Poppler** (for PDF to image conversion)
    - Windows: Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases/)
    - Add `bin` folder to PATH or set `POPPLER_PATH` environment variable
  - **Tesseract OCR** (optional, for OCR fallback)
    - Windows: Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Anthropic API Key**: Get one from [Anthropic Console](https://console.anthropic.com/)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PM-RAG-ChatBot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Activate on Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Activate on Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Activate on Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Spacy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
POPPLER_PATH=C:\path\to\poppler\bin  # Optional, if not in PATH
CLAUDE_VISION_MODEL=claude-sonnet-4-5-20250929  # Optional, defaults to this
```

## 🎯 Usage

### Web Interface (Recommended)

Start the Chainlit web application:

```bash
chainlit run app.py
```

The application will start at `http://localhost:8000` (default port).

**Features:**
- Create new project chats or resume existing ones
- Upload SRD PDF documents
- Add diagrams with vision processing options
- Interactive Q&A with chat history
- Provide feedback to improve responses

### Command Line Interface

For a terminal-based interface:

```bash
python main_final.py
```

**Options:**
1. Index Documents - Upload SRD PDF and optional diagrams
2. Ask Question - Query the indexed documents
3. Exit

## 📁 Project Structure

```
PM-RAG-ChatBot/
├── app.py                  # Main Chainlit web application
├── main_final.py           # Command-line interface
├── srd_engine_v2.py        # Smart knowledge base with multi-user support
├── srd_engine_final.py     # Core RAG engine and document processing
├── db.py                   # SQLite database models (User, Chat, Message)
├── requirements.txt        # Python dependencies
├── SETUP.md               # Detailed setup instructions
├── chainlit.md            # Chainlit welcome screen content
├── Dockerfile             # Docker container configuration
└── README.md              # This file
```

## 🔑 Key Components

### `app.py`
- Chainlit web application
- User session management
- Chat creation and resumption
- Document ingestion workflow
- Message handling and feedback collection

### `srd_engine_v2.py`
- `SmartKnowledgeBase`: Multi-user, multi-chat knowledge base
- `DiagramInterpreter`: Vision model integration for diagram processing
- `SmartSRDSplitter`: Section-aware document splitting
- Scoped retrieval with user/project/chat isolation

### `srd_engine_final.py`
- `SRDChatbotEngine`: Core RAG engine with hybrid search
- `ClaudeAnswerer`: Claude API wrapper
- Intent detection and requirement type classification
- Cross-encoder reranking

### `db.py`
- SQLAlchemy models for:
  - `User`: User identification
  - `Chat`: Project chat sessions
  - `Message`: Chat messages and feedback

## 🛠️ Technologies Used

- **Chainlit**: Web UI framework for LLM applications
- **LangChain**: Document processing and retrieval framework
- **ChromaDB**: Vector database for embeddings
- **Anthropic Claude**: LLM for answer generation and vision
- **Sentence Transformers**: Embedding models
- **spaCy**: NLP for text processing
- **pdfplumber**: PDF text extraction
- **camelot-py**: Table extraction from PDFs
- **pytesseract**: OCR for diagrams
- **SQLAlchemy**: Database ORM
- **python-dotenv**: Environment variable management

## 🐳 Docker Support

Build and run with Docker:

```bash
docker build -t pm-rag-chatbot .
docker run -p 7860:7860 --env-file .env pm-rag-chatbot
```

The application will be available at `http://localhost:7860`.

## 🔍 How It Works

1. **Document Upload**: User uploads SRD PDF and optional diagrams
2. **Processing**: 
   - PDF text is extracted and split into sections
   - Tables are extracted separately
   - Diagrams are processed with vision models or OCR
3. **Indexing**: Documents are embedded and stored in ChromaDB with metadata
4. **Querying**: 
   - User asks a question
   - System performs hybrid search (dense + sparse)
   - Results are reranked using cross-encoder
   - Top results are sent to Claude for answer generation
5. **Learning**: User feedback can be incorporated to improve future responses (if enabled)

## 🐛 Troubleshooting

### "ghostscript not found" error
- Install Ghostscript: https://www.ghostscript.com/download/gsdnld.html

### "Poppler not found" error
- Ensure Poppler is installed and in PATH
- Or set `POPPLER_PATH` in `.env` file

### "ANTHROPIC_API_KEY not set" error
- Create `.env` file with your API key
- Ensure `.env` is in the project root

### Import errors
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Spacy model not found
- Run: `python -m spacy download en_core_web_sm`

### Database errors
- Delete `cedropass.db` to reset the database
- Ensure write permissions in the project directory

## 📝 Notes

- The system uses a global ChromaDB collection but filters by `project_id`, `chat_id`, and `user_id` for isolation
- Learning mode is opt-in per chat session
- Vision processing is optional - you can use OCR only for faster processing
- Large PDFs may take time to process during ingestion

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contributing guidelines if applicable]

## 📧 Contact


[Add contact information if applicable]
