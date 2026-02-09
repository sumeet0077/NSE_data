
# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for pyarrow/pandas)
# netcat is useful for debugging, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (if you had a requirements.txt, checking manually for now)
# We will install dependencies directly
RUN pip install --no-cache-dir pandas requests pyarrow

# Copy the entire project code
COPY . /app

# Ensure directories exist
RUN mkdir -p /app/processed/daily /app/raw_downloads

# Set environment variables
ENV TZ=Asia/Kolkata
ENV PYTHONUNBUFFERED=1

# Run the service directly
# "python nse_daily_update_service.py --service"
CMD ["python", "nse_daily_update_service.py", "--service"]
