import os
import tqdm
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf

def run_serial_pipeline(input_file, output_dir, batch_size=10000):
    # 1. Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Initialize the generator (streaming, not loading everything)
    spectra_gen = load_from_mgf(input_file)
    
    current_batch = []
    batch_id = 0
    total_spectra_saved = 0

    # 3. Process with a visual progress bar
    # Note: 'total' is unknown for generators, so the bar will show 'it/s'
    with tqdm.tqdm(desc="Processing Batches", unit="batch") as pbar:
        for spec in spectra_gen:
            current_batch.append(spec)
            
            # When the batch is full, save it
            if len(current_batch) >= batch_size:
                output_path = os.path.join(output_dir, f"batch_{batch_id}.mgf")
                
                # Optional: Apply your cleaning logic here
                # e.g., current_batch = [your_filter_func(s) for s in current_batch]
                
                save_as_mgf(current_batch, output_path)
                
                total_spectra_saved += len(current_batch)
                batch_id += 1
                pbar.set_postfix({"Total Spectra": total_spectra_saved})
                pbar.update(1)
                
                # Clear current batch to free up RAM
                current_batch = []

        # 4. Handle the final remaining spectra
        if current_batch:
            output_path = os.path.join(output_dir, f"batch_{batch_id}.mgf")
            save_as_mgf(current_batch, output_path)
            total_spectra_saved += len(current_batch)
            pbar.update(1)

    print(f"\nPipeline Complete!")
    print(f"Total batches created: {batch_id + 1}")
    print(f"Total spectra processed: {total_spectra_saved}")

if __name__ == "__main__":
    run_serial_pipeline("large_library.mgf", "serial_chunks", batch_size=10000)