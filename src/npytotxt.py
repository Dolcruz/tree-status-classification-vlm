#!/usr/bin/env python3
"""
NumPy Array to Text Converter

This utility converts NumPy array files (.npy) to text files (.txt)
for point cloud data processing.

Usage:
    python npytotxt.py

Update the folder paths before running.
"""

import numpy as np
import os
from pathlib import Path

def convert_npy_to_txt(input_folder, output_folder=None, delimiter=" "):
    """
    Convert .npy files to .txt files.
    
    Args:
        input_folder: Directory containing .npy files
        output_folder: Directory to save .txt files (optional)
        delimiter: Delimiter for text output
    """
    
    input_path = Path(input_folder)
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Get all .npy files
    npy_files = list(input_path.glob("*.npy"))
    
    if not npy_files:
        print("No .npy files found in the input folder.")
        return
    
    print(f"Found {len(npy_files)} .npy files to convert")
    
    converted_count = 0
    for npy_file in npy_files:
        try:
            # Load NumPy array
            data = np.load(npy_file)
            
            # Generate output filename
            txt_filename = npy_file.stem + ".txt"
            txt_path = output_path / txt_filename
            
            # Save as text file
            np.savetxt(txt_path, data, delimiter=delimiter, fmt='%.6f')
            
            print(f"  {npy_file.name} -> {txt_filename} (shape: {data.shape})")
            converted_count += 1
            
        except Exception as e:
            print(f"  Error converting {npy_file.name}: {e}")
    
    print(f"\nConversion completed! {converted_count} files converted.")

def main():
    """Main function with configuration."""
    
    # Configuration - Update these paths
    INPUT_FOLDER = "Your path to/npy_files"  # Update this path
    OUTPUT_FOLDER = "Your path to/txt_files"  # Update this path (optional)
    
    # Settings
    DELIMITER = " "  # Space delimiter, change to "," for CSV, ";" for semicolon
    
    print("NumPy to Text Converter")
    print("=" * 25)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Delimiter: '{DELIMITER}'")
    print()
    
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist!")
        print("Please update the INPUT_FOLDER path in the script.")
        return
    
    try:
        convert_npy_to_txt(INPUT_FOLDER, OUTPUT_FOLDER, DELIMITER)
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()