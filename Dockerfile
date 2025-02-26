# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables to prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . /app

CMD ["python", "task4_main.py"]
