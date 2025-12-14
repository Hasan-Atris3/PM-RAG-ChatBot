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

# 3. Download the Spacy model during build
RUN python -m spacy download en_core_web_sm

# 4. Create a non-root user (Required for Hugging Face)
RUN useradd -m -u 1000 user

# 5. Fix Permissions BEFORE switching user
# Create the folders as root, then give them to the user
RUN mkdir -p /app/chroma_global_db
RUN mkdir -p /data
RUN chown -R 1000:1000 /app
RUN chown -R 1000:1000 /data

# 6. Switch to the non-root user
USER user

# 7. Copy the application code
COPY --chown=user . .

# 8. Expose the port
EXPOSE 7860

# 9. Run the application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]