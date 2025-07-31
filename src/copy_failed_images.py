#!/usr/bin/env python3
"""
Failed Images Copy Utility

This utility helps manage failed or problematic tree images by copying them
to a separate directory for analysis or exclusion from processing.

Usage:
    python copy_failed_images.py

Update the configuration before running.
"""

import os
import shutil
from pathlib import Path

def copy_failed_images(failed_tree_ids, source_folder, destination_folder, image_extensions=None):
    """
    Copy images of failed/problematic trees to a separate folder.
    
    Args:
        failed_tree_ids: List of tree IDs that failed processing
        source_folder: Directory containing all tree images
        destination_folder: Directory to copy failed images
        image_extensions: List of image extensions to copy
    """
    
    if image_extensions is None:
        image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    for tree_id in failed_tree_ids:
        print(f"Looking for images of tree {tree_id}...")
        
        # Look for all images related to this tree ID
        for ext in image_extensions:
            # Pattern: tree_id followed by underscore and any suffix
            pattern = f"{tree_id}_*{ext}"
            matching_files = list(source_path.glob(pattern))
            
            # Also look for exact match (tree_id.ext)
            exact_match = source_path / f"{tree_id}{ext}"
            if exact_match.exists():
                matching_files.append(exact_match)
            
            for file_path in matching_files:
                dest_file = dest_path / file_path.name
                try:
                    shutil.copy2(file_path, dest_file)
                    print(f"  Copied: {file_path.name}")
                    copied_count += 1
                except Exception as e:
                    print(f"  Error copying {file_path.name}: {e}")
    
    print(f"\nCompleted! Copied {copied_count} files to {destination_folder}")

def read_failed_ids_from_file(file_path):
    """
    Read failed tree IDs from a text file.
    
    Expected format: One ID per line, or comma-separated IDs
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # Try comma-separated first
        if ',' in content:
            ids = [id.strip() for id in content.split(',')]
        else:
            # Try line-separated
            ids = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Filter out empty strings and convert to strings
        ids = [str(id) for id in ids if id]
        
        return ids
    except Exception as e:
        print(f"Error reading failed IDs file: {e}")
        return []

def main():
    """Main function with configuration."""
    
    # Configuration - Update these paths and settings
    SOURCE_FOLDER = "Your path to/TreeswithID"  # Update this path
    DESTINATION_FOLDER = "Your path to/Failed_Images"  # Update this path
    
    # Failed tree IDs - Update this list or provide a file
    FAILED_IDS_FILE = None  # Set to file path if reading from file
    FAILED_TREE_IDS = ["38", "307"]  # Manual list of failed tree IDs
    
    # Image extensions to copy
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    
    print("Failed Images Copy Utility")
    print("=" * 30)
    print(f"Source folder: {SOURCE_FOLDER}")
    print(f"Destination folder: {DESTINATION_FOLDER}")
    print()
    
    # Check if source folder exists
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: Source folder '{SOURCE_FOLDER}' does not exist!")
        print("Please update the SOURCE_FOLDER path in the script.")
        return
    
    # Get failed IDs
    if FAILED_IDS_FILE and os.path.exists(FAILED_IDS_FILE):
        print(f"Reading failed IDs from file: {FAILED_IDS_FILE}")
        failed_ids = read_failed_ids_from_file(FAILED_IDS_FILE)
    else:
        failed_ids = FAILED_TREE_IDS
    
    if not failed_ids:
        print("No failed tree IDs specified or found.")
        print("Please update FAILED_TREE_IDS list or provide FAILED_IDS_FILE.")
        return
    
    print(f"Failed tree IDs to process: {failed_ids}")
    print(f"Image extensions: {IMAGE_EXTENSIONS}")
    print()
    
    # Confirm before proceeding
    response = input("Proceed with copying failed images? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    try:
        copy_failed_images(failed_ids, SOURCE_FOLDER, DESTINATION_FOLDER, IMAGE_EXTENSIONS)
    except Exception as e:
        print(f"Error during copying: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()