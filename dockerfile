# Use official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the application files
COPY app.py model.pth /app/

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir flask flask-cors torch tiktoken

# Expose the port your Flask app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
























# # Use official PyTorch base image with CUDA support
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# # Set the working directory
# WORKDIR /app

# # Copy the application files
# COPY app.py model.pth /app/

# # Install system dependencies
# RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# # Install required Python packages
# RUN pip install --no-cache-dir flask flask-cors torch tiktoken

# # Expose the port your Flask app runs on
# EXPOSE 5000

# # Define the command to run the application
# CMD ["python", "app.py"]
