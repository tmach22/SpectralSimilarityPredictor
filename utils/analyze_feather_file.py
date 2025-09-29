import pandas as pd
import numpy as np
import pyarrow.feather as feather
import argparse
import os
import sys

def analyze_feather_file(file_path):
    """Analyzes the structure and content of a feather file."""
    try:
        # Read the feather file using pandas
        df = pd.read_feather(file_path)

        print("\n" + "="*50)
        print(f"Analysis Report for: {file_path}")
        print("="*50)

        # 1. Total Number of Records
        print(f"Total Number of Records: {len(df)}")
        print("-" * 50)

        # 2. Columns and Data Types
        print("Column Names and Data Types:")
        # Create a DataFrame to display dtypes nicely
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ['Column Name', 'Data Type']
        print(dtype_df.to_markdown(index=False))
        print("-" * 50)

        # 3. Example Records
        print("First 5 Example Records:")
        if len(df) > 0:
            # Use .to_markdown() for clean table output
            print(df.head())
        else:
            print("The file contains no records.")
        print("="*50)

    except FileNotFoundError:
        print(f"\n[Error] File not found: {file_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"\n[Error] An error occurred during file analysis: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze the structure and contents of a .feather file.")
    # The 'nargs="?"' makes the argument optional
    parser.add_argument("--file_path", nargs='?', default=None, 
                        help="Path to the .feather file to analyze. If not provided, a dummy file is created.")
    
    args = parser.parse_args()
    
    analysis_file_path = args.file_path

    # Proceed with analysis
    analyze_feather_file(analysis_file_path)
