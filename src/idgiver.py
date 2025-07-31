#!/usr/bin/env python3
"""
Tree ID Assignment Utility

This utility script assigns unique IDs to tree point cloud files.
It processes files in a directory and renames them with sequential IDs.

Usage:
    python idgiver.py

Update the folder paths and settings before running.
"""

import os
import shutil
from pathlib import Path

def assign_tree_ids(input_folder, output_folder=None, start_id=1, file_extension=".txt"):
    """
    Assign sequential IDs to tree files.
    
    Args:
        input_folder: Directory containing tree files
        output_folder: Directory to save renamed files (optional)
        start_id: Starting ID number
        file_extension: File extension to process
    """
    
    input_path = Path(input_folder)
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Get all files with specified extension
    files = list(input_path.glob(f"*{file_extension}"))
    files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(files)} files to process")
    
    current_id = start_id
    for file_path in files:
        new_name = f"{current_id}{file_extension}"
        new_path = output_path / new_name
        
        if output_folder:
            # Copy to new location with new name
            shutil.copy2(file_path, new_path)
        else:
            # Rename in place
            file_path.rename(new_path)
        
        print(f"  {file_path.name} -> {new_name}")
        current_id += 1
    
    print(f"Completed! Assigned IDs from {start_id} to {current_id - 1}")

def main():
    """Main function with configuration."""
    
    # Configuration - Update these paths
    INPUT_FOLDER = "Your path to/input_trees"  # Update this path
    OUTPUT_FOLDER = "Your path to/numbered_trees"  # Update this path (optional)
    
    # Settings
    START_ID = 1
    FILE_EXTENSION = ".txt"  # Change to .ply, .las, etc. as needed
    
    print("Tree ID Assignment Utility")
    print("=" * 30)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Starting ID: {START_ID}")
    print(f"File extension: {FILE_EXTENSION}")
    print()
    
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist!")
        print("Please update the INPUT_FOLDER path in the script.")
        return
    
    # Confirm before proceeding
    response = input("Proceed with ID assignment? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    try:
        assign_tree_ids(INPUT_FOLDER, OUTPUT_FOLDER, START_ID, FILE_EXTENSION)
    except Exception as e:
        print(f"Error during ID assignment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()