FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Expose the port (Railway assigns dynamically, default to 7860)
EXPOSE 8080
ENV PORT=8080

# Set environment variables for Flask and logging
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

COPY templates /app/templates

CMD ["python", "app.py"]
