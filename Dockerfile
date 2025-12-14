# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POPPLER_PATH=/usr/bin

# Set the working directory
WORKDIR /app

# 1. Install System Dependencies (Poppler & Tesseract)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Download the Spacy model during build to save time on startup
RUN python -m spacy download en_core_web_sm

# 4. Copy the rest of the application code
COPY . .

# 5. Create the directory for the global DB if it doesn't exist
RUN mkdir -p chroma_global_db
RUN chmod -R 777 chroma_global_db

# 6. Expose the port Chainlit runs on
EXPOSE 7860

# 7. Run the application
# Chainlit on HuggingFace must run on port 7860
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]