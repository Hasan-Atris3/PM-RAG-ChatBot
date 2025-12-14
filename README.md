---
title: CedroPM Bot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

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
