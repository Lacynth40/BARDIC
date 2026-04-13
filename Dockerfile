# Use a slim Python image to keep the footprint small
FROM python:3.12-slim

# Install system dependencies (needed for FAISS and NLTK downloads)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements first to leverage Docker's cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (core, config, and main.py)
COPY . .

# Pre-download the NLTK data and Model weights
# This prevents the container from trying to download them at runtime
RUN python3 -c "import nltk; nltk.download('cmudict')"
# Optional: Run a script to pre-cache the T5 model weights here

# Set the entry point
CMD ["python", "main.py"]