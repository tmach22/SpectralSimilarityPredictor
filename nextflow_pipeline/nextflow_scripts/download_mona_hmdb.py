import requests
import json
import os
import argparse
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def download_paginated_mona(base_url, output_dir, batch_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the URL to find the query parameters
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query)
    
    # Force the batch size
    params['size'] = [str(batch_size)]
    
    page = 0
    total_spectra = 0
    
    print(f"Starting Paginated Download from MoNA...")
    print(f"Query: {params.get('query', [''])[0]}")

    while True:
        # Update page number
        params['page'] = [str(page)]
        
        # Reconstruct URL
        new_query = urlencode(params, doseq=True)
        new_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
        
        try:
            print(f"  Downloading Page {page}...", end="", flush=True)
            response = requests.get(new_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(" Done (No more data).")
                break
                
            # Save batch
            batch_filename = os.path.join(output_dir, f"mona_batch_{page}.json")
            with open(batch_filename, 'w') as f:
                json.dump(data, f)
            
            count = len(data)
            total_spectra += count
            print(f" Got {count} spectra. (Total: {total_spectra})")
            
            if count < batch_size:
                print("  Reached last page.")
                break
                
            page += 1
            time.sleep(1) # Be polite to the API
            
        except Exception as e:
            print(f"\nError on page {page}: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="The full curl URL provided by the user")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    download_paginated_mona(args.url, args.output_dir)