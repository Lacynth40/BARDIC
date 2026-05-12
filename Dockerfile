# Use a slim Python image
FROM python:3.12-slim

# Install system dependencies 
# Added libgomp1 for FAISS and clean up to keep image small
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data (Anubis needs this)
RUN python3 -c "import nltk; nltk.download('cmudict')"

# PRE-CACHE MODELS: Bake the brains into the image
# This prevents downloading ~1GB of weights every time the container starts
# Trying a different model head. 
RUN python3 -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; \
    T5Tokenizer.from_pretrained('google/flan-t5-large'); \
    T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')"

RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the project
COPY . .

# Set the entry point
CMD ["python", "main.py"]