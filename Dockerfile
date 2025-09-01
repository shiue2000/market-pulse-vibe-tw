FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/index.html templates/index.html

EXPOSE 8080


# Command to run the Flask app with Gunicorn for Railway
# CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
CMD ["python", "app.py"]
