---
title: CedroPM Bot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# PM-RAG-ChatBot (CedroPM)

A specialized RAG (Retrieval Augmented Generation) chatbot designed for Project Managers and Software Architects. It processes Software Requirements Documents (SRDs) and provides expert-level analysis by combining project-specific data with a pre-seeded "Gold Standard" knowledge base of industry best practices.

## 🚀 Key Features

* **📄 Intelligent SRD Analysis**: Upload PDF requirements and get answers grounded strictly in your document.
* **🧠 Global Expert Knowledge**: Pre-seeded with 30+ industry standards (ISO, OWASP, NIST, Agile) to provide "Senior Architect" advice even when your PDF is silent.
* **🎨 Diagram Vision**: Understands and explains architecture diagrams using Claude Vision or Qwen2-VL.
* **🔍 Hybrid Search**: Uses a dual-scope retrieval engine to search your **Project PDF** and **Global Best Practices** simultaneously.
* **💬 Interactive Chat**: Built with Chainlit for a clean, chat-like interface with history retention.
* **🔐 Secure & Private**: Multi-user isolation ensures one project's data never leaks to another.

## 🏗️ System Architecture

1.  **Ingestion Layer**:
    * **PDFs**: Processed with `pdfplumber` and `SmartSRDSplitter` (Section-aware chunking).
    * **Tables**: Extracted via `camelot` and converted to Markdown.
    * **Diagrams**: Processed via OCR (`Tesseract`) or Vision LLM (`Claude 3.5 Sonnet`).
2.  **Knowledge Store**:
    * **ChromaDB**: vector storage for semantic search.
    * **Dual-Scope**: Queries filter by `(User + Project)` OR `(Global_Expert_Knowledge)`.
3.  **Reasoning Engine**:
    * **Claude 3.5 Sonnet**: Generates detailed, structured responses.
    * **Seed Data**: A Python script injects "Golden Rules" for Security, DevOps, and PM methodologies.

## 📋 Prerequisites

* **Python 3.10+**
* **System Tools**:
    * `Poppler` (for PDF rendering)
    * `Tesseract OCR` (for image text extraction)
* **API Key**: Anthropic API Key (for Claude).

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd PM-RAG-ChatBot
