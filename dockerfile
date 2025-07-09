# Use a Python base image with a compatible version
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and ChromaDB/storage directories
COPY app.py .
COPY chroma_db ./chroma_db
COPY storage ./storage

# Set permissions for chroma_db and storage directories
RUN chown -R appuser:appuser /app/chroma_db /app/storage && chmod -R 775 /app/chroma_db /app/storage

# Expose the Streamlit port

EXPOSE 8501

# Command to run the Streamlit app

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]