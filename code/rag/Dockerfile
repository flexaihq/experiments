# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/
COPY run_rag.py ./run_rag.py
COPY logo.png ./logo.png

# Expose the port the app runs on
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the application
CMD ["python", "run_rag.py"]
