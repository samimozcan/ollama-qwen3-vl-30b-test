import requests
import json
import base64
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:30b-a3b-instruct-q8_0"

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def query_ollama_vision(prompt, image_path, model=MODEL_NAME):
    """Query Ollama with a vision model using an image."""
    
    # Encode the image to base64
    image_base64 = encode_image_to_base64(image_path)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.RequestException as e:
        return f"Error: {e}"

def save_output(result, image_path, elapsed_time):
    """Save the output to the out/ folder with timing information."""
    # Create out directory if it doesn't exist
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    
    # Generate output filename based on image name and timestamp
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{image_name}_{timestamp}.json"
    output_path = out_dir / output_filename
    
    # Prepare output data with timing
    output_data = {
        "image_path": str(image_path),
        "model": MODEL_NAME,
        "processing_time_seconds": round(elapsed_time, 2),
        "timestamp": datetime.now().isoformat(),
        "result": result
    }
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_path

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python engine.py <image_path>")
        print("Example: python engine.py invoice.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Prompt for invoice extraction
    prompt = """You are a helpful assistant. Extract all fields from the provided invoice image and return them in JSON format.
Do not add any text except JSON.
Extract fields such as: invoice number, date, vendor name, vendor address, customer name, customer address, line items (description, quantity, unit price, total), subtotal, tax, total amount, payment terms, etc."""

    print(f"Processing image: {image_path}")
    print(f"Using model: {MODEL_NAME}")
    print("Please wait...")
    
    # Start timing
    start_time = time.time()
    
    # Get the response from Ollama
    result = query_ollama_vision(prompt, image_path)
    
    # End timing
    elapsed_time = time.time() - start_time
    
    # Print the result
    print("\n" + "="*50)
    print("RESULT:")
    print("="*50)
    print(result)
    print("="*50)
    print(f"Processing time: {elapsed_time:.2f} seconds")
    
    # Save output
    output_path = save_output(result, image_path, elapsed_time)
    print(f"Output saved to: {output_path}")