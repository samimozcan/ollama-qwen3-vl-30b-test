#!/bin/bash

# Qwen Invoice Processing Script
# Usage: ./qwen_invoice.sh <image_path>

if [ -z "$1" ]; then
    echo "Usage: ./qwen_invoice.sh <image_path>"
    echo "Example: ./qwen_invoice.sh invoice.png"
    exit 1
fi

IMAGE_PATH="$1"

# Check if image file exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

echo "========================================"
echo "Qwen Invoice Processing"
echo "========================================"
echo "Image: $IMAGE_PATH"
echo "Model: qwen3-vl:30b-a3b-instruct-q8_0"
echo "========================================"

# Create output directory if it doesn't exist
mkdir -p out

# Run the Python engine with the image path
python engine.py "$IMAGE_PATH"

echo "========================================"
echo "Processing complete!"
echo "========================================"