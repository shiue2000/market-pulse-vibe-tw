# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
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

# Command to run the Flask app with Gunicorn for Railway
# CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
COPY templates /app/templates

CMD ["python", "app.py"]
