# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8080
EXPOSE 8080

# Run Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "--log-level=info", "app:app"]
