# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py .
COPY .env .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the bot
CMD ["python", "main.py"]