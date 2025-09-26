# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Copy the dependencies file to the working directory
COPY ./app/requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code from the 'app' directory into the container
COPY ./app /code/app

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run your app using uvicorn
# It will look for the 'app' object in the 'app.api' module
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

