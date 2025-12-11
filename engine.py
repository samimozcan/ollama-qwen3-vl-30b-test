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

import re

def parse_json_from_result(result):
    """Extract and parse JSON from LLM response, removing ```json``` markers if present."""
    cleaned = result.strip()
    
    # Remove ```json ... ``` or ``` ... ``` code block markers
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_block_pattern, cleaned)
    
    if match:
        cleaned = match.group(1).strip()
    
    # Try to parse as JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"parse_error": str(e), "raw_content": cleaned}

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
    
    # Parse JSON from result
    result_json = parse_json_from_result(result)
    
    # Prepare output data with timing
    output_data = {
        "image_path": str(image_path),
        "model": MODEL_NAME,
        "processing_time_seconds": round(elapsed_time, 2),
        "timestamp": datetime.now().isoformat(),
        "result": result,
        "result_JSON": result_json
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
    prompt = """Extract invoice data. Output ONLY valid JSON array with one object:

[{"invoice_currency":"str|null","invoice_customer_address":"str|null","invoice_customer_country":"str|null","invoice_customer_name":"str|null","invoice_date":"str|null","invoice_delivery_term":"str|null","invoice_id":"str|null","invoice_payment_term":"str|null","invoice_po_number":"str|null","invoice_shipment_country_of_origin":"str|null","invoice_supplier_address":"str|null","invoice_supplier_country":"str|null","invoice_supplier_name":"str|null","invoice_supplier_vkn":"str|null","invoice_total_amount":"str|null","invoice_total_package_quantity":"str|null","invoice_total_quantity":"str|null","invoice_total_gross_weight":"str|null","invoice_total_net_weight":"str|null","items":[{"invoice_item_commodity_code":"str|null","invoice_item_country_of_origin":"str|null","invoice_item_description":"str|null","invoice_item_no":"str|null","invoice_item_package_quantity":"str|null","invoice_item_product_id":"str|null","invoice_item_quantity":"str|null","invoice_item_total_amount":"str|null","invoice_item_unit_price":"str|null","invoice_item_unit_type":"str|null"}]}]

RULES:
- Output RAW JSON only, no markdown/explanations
- Use null for missing fields (never "N/A", "", "-")
- Preserve original number formats (e.g., 4.013.082,09)
- invoice_currency: primary transaction currency (EUR if items in EUR)
- invoice_total_amount: include currency (e.g., "4.013.082,09 TL")
- Countries as source (ALMANYA, TÜRKİYE)
- Weights: numeric kg string, no unit (e.g., "19050")

COMMODITY CODE (invoice_item_commodity_code):
- Labels: Esya Kodu, Gtip Kodu, GTIP, HS CODE, HS Kodu, Malzeme/Hizmet Kodu, Ürün Kodu
- Priority: GTIP > HS CODE > Esya Kodu > Malzeme/Hizmet Kodu > SKU
- If none found → null

COUNTRY OF ORIGIN:
- If "Eşyalar Türk menşeilidir" → all items "TÜRKİYE"
- Otherwise check per-item Ürün Menşei/Country of Origin"""

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