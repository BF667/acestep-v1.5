# HuggingFace Space Docker SDK
# Use slim Python image - HuggingFace GPU Spaces provide CUDA runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TORCHAUDIO_USE_TORCHCODEC=0

# Install system dependencies
# build-essential is required for triton to compile CUDA kernels
# ffmpeg and libav* dev packages are required for torchaudio's ffmpeg backend
# Note: torchaudio's ffmpeg backend needs shared libraries, not just the ffmpeg binary
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libsndfile1 build-essential && \
    apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev && \
    rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 (HuggingFace Space requirement)
RUN useradd -m -u 1000 user

# Create /data directory with proper permissions for persistent storage
RUN mkdir -p /data && chown user:user /data && chmod 755 /data

# Set environment variables for user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Set the working directory
WORKDIR $HOME/app

# Copy requirements first for better Docker layer caching
COPY --chown=user:user requirements.txt .

# Copy the local nano-vllm package
COPY --chown=user:user acestep/third_parts/nano-vllm ./acestep/third_parts/nano-vllm

# Switch to user before installing packages
USER user

# Install dependencies from requirements.txt (includes PyTorch with CUDA from --extra-index-url)
RUN pip install --no-cache-dir --user -r requirements.txt

# Install nano-vllm with --no-deps since all dependencies are already installed
RUN pip install --no-deps ./acestep/third_parts/nano-vllm

# Copy the rest of the application
COPY --chown=user:user . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
