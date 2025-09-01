FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Redis, Python, and Llama model
RUN apt-get update && apt-get install -y \
    redis-tools \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/index.html templates/index.html

# Expose port for Flask app
EXPOSE 8080

# Set environment variables (optional, can be overridden in docker-compose.yml)
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

COPY templates /app/templates

CMD ["python", "app.py"]
