# Use a Python base image with a compatible version
FROM python:3.9-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create the /app directory
RUN mkdir -p /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/

RUN pip install --upgrade pip 

# Use a Python base image with a compatible version
FROM python:3.9-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create the /app directory
RUN mkdir -p /app

# Set working directory
WORKDIR /app

# Update pip and install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r /app/requirements.txt

# Copy the application code and ChromaDB/storage directories
COPY app.py /app/
COPY chroma_db /app/chroma_db/
COPY storage /app/storage/

# Set working directory
WORKDIR /app

# Set permissions for chroma_db and storage directories
RUN chown -R appuser:appuser /app/chroma_db /app/storage && chmod -R 775 /app/chroma_db /app/storage

# Expose the Streamlit port

EXPOSE 8501

# Add health check for Render
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the Streamlit app

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]