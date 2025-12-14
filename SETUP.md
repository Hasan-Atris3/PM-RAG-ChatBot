# PM-RAG-ChatBot Setup Guide

## Prerequisites

### 1. Python 3.8 or higher
Make sure Python is installed on your system.

### 2. System Dependencies (for PDF processing)

#### Windows:
- **Poppler** (for PDF to image conversion):
  - Download from: https://github.com/oschwartz10612/poppler-windows/releases/
  - Extract and add `bin` folder to your system PATH
  - Or set `POPPLER_PATH` environment variable to the `bin` folder

- **Tesseract OCR** (optional, for OCR):
  - Download from: https://github.com/UB-Mannheim/tesseract/wiki
  - Install and add to PATH

#### Alternative (using conda):
```bash
conda install -c conda-forge poppler tesseract
```

### 3. API Keys

Create a `.env` file in the project root with:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
POPPLER_PATH=C:\path\to\poppler\bin  # Optional, if not in PATH
CLAUDE_VISION_MODEL=claude-sonnet-4-5-20250929  # Optional, defaults to this
```

**Get your Anthropic API key:**
- Sign up at https://console.anthropic.com/
- Create an API key
- Add it to your `.env` file

## Installation Steps

### 1. Navigate to the project directory
```bash
cd PM-RAG-ChatBot
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

**Note:** Some packages may require additional setup:
- `camelot-py[cv]` requires `ghostscript` and `tcl-tk` on some systems
- `spacy` may need a language model: `python -m spacy download en_core_web_sm`

### 5. Install Spacy language model (if needed)
```bash
python -m spacy download en_core_web_sm
```

## Running the Application

### Option 1: Web Interface (Chainlit) - Recommended
```bash
chainlit run app.py
```

This will start a web server (usually at http://localhost:8000)

### Option 2: Command Line Interface
```bash
python main_final.py
```

## Project Structure

- `app.py` - Main Chainlit web application
- `main_final.py` - Command-line interface
- `srd_engine_v2.py` - Main RAG engine with knowledge base
- `srd_engine_final.py` - Core SRD chatbot engine
- `db.py` - SQLite database models
- `requirements.txt` - Python dependencies

## Features

- Upload SRD PDF documents
- Process diagrams with vision models (Qwen2-VL and/or Claude Vision)
- Chat interface for querying documents
- Persistent chat history
- Learning from user feedback

## Troubleshooting

1. **"ghostscript not found" error:**
   - Install Ghostscript: https://www.ghostscript.com/download/gsdnld.html

2. **"Poppler not found" error:**
   - Make sure Poppler is installed and in PATH, or set `POPPLER_PATH` in `.env`

3. **"ANTHROPIC_API_KEY not set" error:**
   - Create a `.env` file with your API key

4. **Import errors:**
   - Make sure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

