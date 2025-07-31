#!/usr/bin/env python3
"""
2D Rasterization Point Cloud Processor

This script processes 3D point cloud files and generates 2D rasterized images
through projection onto different planes. It creates 6 different perspective views
by rotating the point cloud and projecting onto XZ and XY planes.

Usage:
    python 2drastertest.py

Requirements:
    - NumPy for numerical operations
    - Matplotlib for visualization and image generation
    - Pillow (PIL) for image processing

Update the INPUT_FOLDER and OUTPUT_FOLDER paths before running.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image

# Configuration - Update these paths to match your setup
INPUT_FOLDER  = "Your path to/pointclouds"  # Update this path to your pointclouds folder
OUTPUT_FOLDER = "Your path to/TreeswithID"  # Update this path to your output folder

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported file extensions for point cloud data
ALLOWED_EXTS = (".ply", ".las", ".txt", ".asc", ".xyz", ".csv")

def read_ascii_cloud(fp: str) -> dict:
    """
    Read ASCII point cloud file and return as dictionary with XYZ and RGB data.
    
    Expected format: X Y Z R G B (6 columns)
    Where R, G, B are color values (either 0-1 float range or 0-255 integer range)
    """
    with open(fp, "r") as f:
        first = f.readline()
    
    # Detect delimiter (semicolon or whitespace)
    delim = ";" if ";" in first else None
    data = np.loadtxt(fp, delimiter=delim)
    
    # Extract XYZ coordinates (first 3 columns)
    xyz = data[:, :3]
    
    result = {
        'xyz': xyz,
        'rgb': None
    }
    
    # Check if RGB values are present (assuming format: X Y Z R G B)
    if data.shape[1] >= 6:
        # Extract RGB values (columns 3, 4, 5)
        rgb_values = data[:, 3:6]
        
        # Detect RGB format and normalize accordingly
        max_rgb = np.max(rgb_values)
        min_rgb = np.min(rgb_values)
        
        if max_rgb > 1.0:
            # Integer format (0-255): ensure proper scaling to 0-255 for visualization
            print(f"    Detected integer RGB values (range: {min_rgb:.0f}-{max_rgb:.0f})")
            rgb_values = (rgb_values / 255.0) * 255.0  # Ensure proper scaling
        else:
            # Float format (0.0-1.0): scale to 0-255 for visualization
            print(f"    Detected float RGB values (range: {min_rgb:.3f}-{max_rgb:.3f})")
            rgb_values = rgb_values * 255.0
        
        # Ensure RGB values are in valid range [0, 255] and convert to uint8
        rgb_values = np.clip(rgb_values, 0, 255).astype(np.uint8)
        result['rgb'] = rgb_values
        print(f"    Successfully loaded {rgb_values.shape[0]} RGB color values")
    else:
        print(f"    ASCII file has only {data.shape[1]} columns, no RGB data found")
        # Set default gray color
        result['rgb'] = np.full((xyz.shape[0], 3), 179, dtype=np.uint8)  # Light gray
    
    return result

def read_pointcloud(fp: str) -> dict:
    """
    Read point cloud file and return XYZ coordinates and RGB colors.
    
    Currently supports ASCII formats. PLY and LAS support would require
    additional libraries like open3d or laspy.
    """
    ext = os.path.splitext(fp)[1].lower()
    if ext in [".ply", ".las"]:
        # For .ply and .las files, we would need open3d or laspy
        print(f"    {ext} format not yet supported in rasterization mode, skipping...")
        return None
    else:
        # ASCII formats
        return read_ascii_cloud(fp)

def rotate_points_z(xyz, angle_degrees):
    """
    Rotate points around Z-axis by given angle in degrees.
    
    Args:
        xyz: Point coordinates array (N, 3)
        angle_degrees: Rotation angle in degrees
    
    Returns:
        Rotated point coordinates array (N, 3)
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around Z-axis
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    
    return xyz @ rotation_matrix.T

def rasterize_pointcloud(xyz, rgb, resolution=0.05, axis1=0, axis2=2, rotation_angle=0):
    """
    Rasterize 3D point cloud to 2D image by projecting onto specified plane.
    
    Args:
        xyz: Point coordinates array (N, 3)
        rgb: RGB colors array (N, 3) in range 0-255
        resolution: Grid resolution for rasterization
        axis1, axis2: Which axes to use for projection (0=X, 1=Y, 2=Z)
        rotation_angle: Angle in degrees to rotate around Z-axis before projection
    
    Returns:
        dict with rasterized image and metadata
    """
    # Apply rotation if specified
    if rotation_angle != 0:
        xyz = rotate_points_z(xyz, rotation_angle)
    
    # Project 3D points onto 2D plane
    view_coords = xyz[:, [axis1, axis2]]
    min_vals = view_coords.min(axis=0)
    max_vals = view_coords.max(axis=0)
    
    # Calculate grid size
    grid_size = ((max_vals - min_vals) / resolution).astype(int) + 1
    
    # Create dictionary to store colors per cell
    cell_colors = defaultdict(list)
    
    # Map points to grid cells
    indices = ((view_coords - min_vals) / resolution).astype(int)
    
    for idx, (i, j) in enumerate(indices):
        # Ensure indices are within bounds
        if 0 <= i < grid_size[0] and 0 <= j < grid_size[1]:
            cell_colors[(i, j)].append(rgb[idx])
    
    # Initialize output image
    rgb_image = np.zeros((*grid_size, 3), dtype=np.uint8)
    
    # Average colors per cell
    for (i, j), colors_in_cell in cell_colors.items():
        avg_color = np.mean(colors_in_cell, axis=0).astype(np.uint8)
        rgb_image[i, j] = avg_color
    
    return {
        'rgb_image': rgb_image,
        'min_vals': min_vals,
        'max_vals': max_vals,
        'grid_size': grid_size,
        'non_zero_pixels': np.count_nonzero(np.sum(rgb_image, axis=2))
    }

def create_rasterized_views(xyz, rgb, resolution=0.05):
    """
    Create 6 different perspective views of the point cloud through rasterization.
    
    Args:
        xyz: Point coordinates array (N, 3)
        rgb: RGB colors array (N, 3)
        resolution: Grid resolution for rasterization
    
    Returns:
        Dictionary containing 6 rasterized perspective views
    """
    print(f"    Creating 6 rasterized views with resolution {resolution}...")
    
    results = {}
    
    # Perspective 1: XZ view at 0° (front view)
    print(f"    Generating perspective 1 (0° front view)...")
    results['perspective_1'] = rasterize_pointcloud(xyz, rgb, resolution, axis1=0, axis2=2, rotation_angle=0)
    
    # Perspective 2: XZ view at 72° 
    print(f"    Generating perspective 2 (72° rotated view)...")
    results['perspective_2'] = rasterize_pointcloud(xyz, rgb, resolution, axis1=0, axis2=2, rotation_angle=72)
    
    # Perspective 3: XZ view at 144°
    print(f"    Generating perspective 3 (144° rotated view)...")
    results['perspective_3'] = rasterize_pointcloud(xyz, rgb, resolution, axis1=0, axis2=2, rotation_angle=144)
    
    # Perspective 4: XZ view at 216°
    print(f"    Generating perspective 4 (216° rotated view)...")
    results['perspective_4'] = rasterize_pointcloud(xyz, rgb, resolution, axis1=0, axis2=2, rotation_angle=216)
    
    # Perspective 5: XZ view at 288°
    print(f"    Generating perspective 5 (288° rotated view)...")
    results['perspective_5'] = rasterize_pointcloud(xyz, rgb, resolution, axis1=0, axis2=2, rotation_angle=288)
    
    # Perspective 6: XY view (top view)
    print(f"    Generating perspective 6 (top view)...")
    results['perspective_6'] = rasterize_pointcloud(xyz, rgb, resolution, axis1=0, axis2=1, rotation_angle=0)
    
    return results

def save_rasterized_images(results, base_name, output_folder):
    """
    Save the rasterized views as both individual and combined images.
    
    Args:
        results: Dictionary containing rasterized perspective views
        base_name: Base filename for output images
        output_folder: Directory to save images
    
    Returns:
        Path to the combined image file
    """
    # Create a combined figure with all six views (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"6 Perspective Views: {base_name}", fontsize=16, color="black")
    
    view_names = ['Perspective 1 (0°)', 'Perspective 2 (72°)', 'Perspective 3 (144°)', 
                  'Perspective 4 (216°)', 'Perspective 5 (288°)', 'Perspective 6 (Top View)']
    view_keys = ['perspective_1', 'perspective_2', 'perspective_3', 
                 'perspective_4', 'perspective_5', 'perspective_6']
    
    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()
    
    for idx, (ax, view_name, view_key) in enumerate(zip(axes_flat, view_names, view_keys)):
        rgb_image = results[view_key]['rgb_image']
        non_zero = results[view_key]['non_zero_pixels']
        total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
        
        # Display the image (different orientation for top view)
        if view_key == 'perspective_6':  # Top view (XY projection)
            ax.imshow(rgb_image, origin='lower')
        else:  # Side views (XZ projections)
            ax.imshow(np.moveaxis(rgb_image, 1, 0), origin='lower')
        
        ax.set_title(f"{view_name}\n{non_zero}/{total_pixels} pixels", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save combined image
    combined_path = os.path.join(output_folder, f"{base_name}_combined.png")
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual views with numbered filenames for tree analysis script
    for idx, (view_key, view_name) in enumerate(zip(view_keys, view_names), 1):
        rgb_image = results[view_key]['rgb_image']
        
        # Create individual image
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Display the image (different orientation for top view)
        if view_key == 'perspective_6':  # Top view (XY projection)
            ax.imshow(rgb_image, origin='lower')
        else:  # Side views (XZ projections)
            ax.imshow(np.moveaxis(rgb_image, 1, 0), origin='lower')
        
        ax.set_title(f"{base_name} - {view_name}", fontsize=14)
        ax.axis('off')
        
        # Save with numbered suffix (_1, _2, _3, _4, _5, _6) for tree analysis script
        individual_path = os.path.join(output_folder, f"{base_name}_{idx}.png")
        fig.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return combined_path

def main():
    """Main processing function."""
    print("Starting 2D Rasterization Processing!")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    processed_count = 0
    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(ALLOWED_EXTS):
            continue
        
        fp = os.path.join(INPUT_FOLDER, fname)
        print(f"\nProcessing {fname}...")
        
        try:
            # Read point cloud
            point_data = read_pointcloud(fp)
            if point_data is None:
                print(f"  Skipped {fname} (unsupported format)")
                continue
            
            xyz = point_data['xyz']
            rgb = point_data['rgb']
            
            if len(xyz) == 0:
                print("  No points found.")
                continue
            
            base = os.path.splitext(fname)[0]
            print(f"  Loaded {len(xyz)} points")
            
            # Create rasterized views
            results = create_rasterized_views(xyz, rgb, resolution=0.05)
            
            # Save images
            combined_path = save_rasterized_images(results, base, OUTPUT_FOLDER)
            print(f"  Saved rasterized views: {combined_path}")
            
            processed_count += 1
            print(f"Completed processing {fname}")
            
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n2D Rasterization completed! Processed {processed_count} files.")
    print(f"Results saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()