# Use an official Python runtime as a parent image
FROM python:3.9

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

# Copy and install dependencies first to leverage Docker layer caching
COPY --chown=user ./app/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code from the local 'app' directory into the container's WORKDIR
COPY --chown=user ./app .

# Command to run the application
# This tells uvicorn to run the 'app' object from the 'api.py' file
# and listen on port 7860, as required by Hugging Face Spaces.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]

