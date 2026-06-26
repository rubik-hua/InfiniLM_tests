#!/bin/bash

# ============================================================
# Global AI Dev Environment Launcher (Pure Ubuntu)
# ============================================================

CONTAINER_NAME="ai-dev-instance"
IMAGE_NAME="ai-dev-env"
IMAGE_TAG="v26.05.26"

# Stop existing container to avoid conflicts
docker rm -f $CONTAINER_NAME >/dev/null 2>&1

# Run container: Mount current directory to /workspace
docker run -it \
  --name $CONTAINER_NAME \
  -e TERM=xterm-256color \
  -v $(pwd):/workspace \
  -v /data/rubik:/data/rubik \
  -v ai-nvim-data:/home/rubik/.local/share/nvim \
  -v $(pwd)/nvim-config:/home/rubik/.config/nvim \
  -w /workspace \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL} \
  -e OPENAI_BASE_URL=${OPENAI_BASE_URL} \
  -e ANTHROPIC_MODEL=${ANTHROPIC_MODEL} \
  -e OPENAI_MODEL=${OPENAI_MODEL} \
  $IMAGE_NAME:$IMAGE_TAG bash

# Attach to the container
#docker exec -it $CONTAINER_NAME bash

