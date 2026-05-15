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


RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the project
COPY . .

# Set the entry point
CMD ["python", "main.py"]