#!/usr/bin/env python3
"""
Point Cloud to Images Converter

This script processes 3D point cloud files and generates multiple perspective images
using Open3D for 3D visualization. It creates 6 side-view images (at different rotation
angles) and 1 top-down view for each tree point cloud.

Usage:
    python pointcloud_to_images.py

Requirements:
    - Open3D for 3D point cloud processing and visualization
    - NumPy for numerical operations
    - Pillow (PIL) for image processing

Update the INPUT_FOLDER and OUTPUT_FOLDER paths before running.
"""

import open3d as o3d
import numpy as np
import os

# Configuration - Update these paths to match your setup
INPUT_FOLDER  = "Your path to/pointclouds"  # Update this path to your pointclouds folder
OUTPUT_FOLDER = "Your path to/TreeswithID"  # Update this path to your output folder

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported file extensions for point cloud data
ALLOWED_EXTS = (".ply", ".las", ".txt", ".asc", ".xyz", ".csv")

# Global renderer to reuse for better performance
global_renderer = None

def initialize_renderer(width=1920, height=1440):
    """Initialize the global OffscreenRenderer for image generation."""
    global global_renderer
    if global_renderer is None:
        try:
            print("Creating global OffscreenRenderer...")
            global_renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            print("Global OffscreenRenderer created successfully!")
            return True
        except Exception as e:
            print(f"Failed to create OffscreenRenderer: {e}")
            return False
    return True

def read_ascii_cloud(fp: str) -> o3d.geometry.PointCloud:
    """
    Read ASCII point cloud file and return Open3D PointCloud object.
    
    Expected format: X Y Z R G B (6 columns)
    Where R, G, B are color values (either 0-1 float range or 0-255 integer range)
    """
    with open(fp, "r") as f:
        first = f.readline()
    
    # Detect delimiter (semicolon or whitespace)
    delim = ";" if ";" in first else None
    data = np.loadtxt(fp, delimiter=delim)
    
    # Extract XYZ coordinates (first 3 columns)
    pts = data[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    
    # Check if RGB values are present (assuming format: X Y Z R G B)
    if data.shape[1] >= 6:
        # Extract RGB values (columns 3, 4, 5)
        rgb_values = data[:, 3:6]
        
        # Detect RGB format and normalize accordingly
        max_rgb = np.max(rgb_values)
        min_rgb = np.min(rgb_values)
        
        if max_rgb > 1.0:
            # Integer format (0-255): normalize to 0-1 range
            print(f"    Detected integer RGB values (range: {min_rgb:.0f}-{max_rgb:.0f})")
            rgb_values = rgb_values / 255.0
            print(f"    Normalized integer RGB to float range (0.0-1.0)")
        else:
            # Float format (0.0-1.0): use values as they are
            print(f"    Detected float RGB values (range: {min_rgb:.3f}-{max_rgb:.3f})")
            print(f"    Using float RGB values directly")
        
        # Ensure RGB values are in valid range [0, 1] (safety check)
        rgb_values = np.clip(rgb_values, 0.0, 1.0)
        
        # Assign colors to point cloud
        pcd.colors = o3d.utility.Vector3dVector(rgb_values)
        print(f"    Successfully loaded {rgb_values.shape[0]} RGB color values")
    else:
        print(f"    ASCII file has only {data.shape[1]} columns, no RGB data found")
    
    return pcd

def read_pointcloud(fp: str) -> o3d.geometry.PointCloud:
    """
    Read point cloud file using appropriate method based on file extension.
    
    Supports:
    - .ply files (using Open3D)
    - .las files (using Open3D)
    - ASCII files (.txt, .asc, .xyz, .csv)
    """
    ext = os.path.splitext(fp)[1].lower()
    
    if ext == ".ply":
        p = o3d.io.read_point_cloud(fp)
    elif ext == ".las":
        p = o3d.io.read_point_cloud(fp, format="las")
    else:
        p = read_ascii_cloud(fp)
    
    # Set default color if point cloud has no colors
    if not p.has_colors() or len(p.colors) == 0:
        print("    No colors found in point cloud, setting default color...")
        # Set a neutral gray color as default
        p.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
    else:
        print(f"    Using original colors from point cloud ({len(p.colors)} color values)")
    
    return p

def snapshot_pointcloud(pcd, out_path, rotation_angle=0, top_down=False, width=1920, height=1440):
    """
    Generate a snapshot of the point cloud from specified viewpoint.
    
    Args:
        pcd: Open3D PointCloud object
        out_path: Output file path for the image
        rotation_angle: Rotation angle around Z-axis (degrees)
        top_down: If True, generate top-down view; if False, generate side view
        width, height: Image dimensions
    """
    global global_renderer
    
    # Try using the global OffscreenRenderer first
    if global_renderer is not None:
        try:
            print(f"    Using global OffscreenRenderer for {out_path}...")
            scene = global_renderer.scene
            
            # Clear previous geometry
            scene.clear_geometry()
            
            # Set background color (RGBA)
            scene.set_background(np.array([0.0, 0.15, 0.2, 1.0]), None)
            
            # Create unlit material with larger point size for better coverage
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 8.0  # Increase point size for better coverage
            scene.add_geometry("pcd", pcd, material)
            
            # Compute bounding box center and radius for framing
            bbox = scene.bounding_box
            center = bbox.get_center()
            extent = bbox.get_extent()
            radius = max(extent) / 2.0
            
            # Calculate camera eye position - either top-down or rotated around Z axis
            if top_down:
                # Position camera directly above the center looking down
                eye = center + np.array([0.0, 0.0, radius * 1.5])  # Closer for better detail
                up = np.array([0.0, 1.0, 0.0])  # Y-axis as up vector for top-down view
                print(f"    Top-down view: camera at {eye}, looking at {center}")
            else:
                # Standard side view with rotation around Z axis - get closer for better coverage
                ang = np.radians(rotation_angle)
                eye = center + np.array([radius * 0.8 * np.cos(ang), radius * 0.8 * np.sin(ang), 0.0])
                up = np.array([0.0, 0.0, 1.0])  # Z-axis as up vector for side views
                print(f"    Side view: rotation {rotation_angle}°, camera at {eye}")
            
            # Position camera and set orthographic projection
            cam = scene.camera
            cam.look_at(center, eye, up)
            
            # Extended orthographic bounds to prevent edge cutting
            left, right = -radius * 1.4, radius * 1.4
            bottom, top = -radius * 1.4, radius * 1.4
            near, far = -radius * 3.0, radius * 3.0
            cam.set_projection(o3d.visualization.rendering.Camera.Projection.Ortho,
                               left, right, bottom, top, near, far)
            
            # Render to image and save
            img = global_renderer.render_to_image()
            o3d.io.write_image(out_path, img)
            print(f"    Global OffscreenRenderer SUCCESS for {out_path}")
            return
            
        except Exception as e:
            print(f"    Global OffscreenRenderer failed: {e}")
    
    # Fallback to hidden legacy Visualizer
    print(f"    Using fallback Visualizer for {out_path}...")
    try:
        vis = o3d.visualization.Visualizer()
        if not vis.create_window(width=width, height=height, visible=False):
            raise Exception("Failed to create visualization window")
        
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        
        if top_down:
            # Set top-down view for fallback visualizer
            ctr.set_front([0, 0, -1])  # Looking down (negative Z direction)
            ctr.set_lookat(pcd.get_center())
            ctr.set_up([0, 1, 0])  # Y-axis as up vector
            print(f"    Fallback top-down view")
        else:
            # Rotate camera around Z axis for side views
            ang = np.radians(rotation_angle)
            ctr.set_front([np.cos(ang), np.sin(ang), 0])
            ctr.set_lookat(pcd.get_center())
            ctr.set_up([0, 0, 1])  # Z-axis as up vector
            print(f"    Fallback side view: rotation {rotation_angle}°")
        
        ctr.set_zoom(1.2)  # Get closer for better detail
        opt = vis.get_render_option()
        opt.background_color = np.array([0, 0.15, 0.2])
        opt.point_size = 10.0  # Increased point size
        
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(out_path, do_render=True)
        vis.destroy_window()
        print(f"    Fallback Visualizer SUCCESS for {out_path}")
        
    except Exception as fallback_error:
        print(f"    Both rendering methods failed: {fallback_error}")
        raise fallback_error

def main():
    """Main processing function."""
    # Initialize the global renderer once
    if initialize_renderer():
        print("Using OffscreenRenderer with orthographic projection!")
    else:
        print("Will use fallback Visualizer with perspective projection")

    # Process all point cloud files in the input folder
    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(ALLOWED_EXTS):
            continue
            
        fp = os.path.join(INPUT_FOLDER, fname)
        print(f"Processing {fname} ...")
        
        try:
            pcd = read_pointcloud(fp)
            if len(pcd.points) == 0:
                print("  No points found.")
                continue
                
            base = os.path.splitext(fname)[0]
            print(f"  Loaded {len(pcd.points)} points")
            
            # Generate 3 side-view snapshots with different rotation angles
            for i, angle in enumerate([30, 150, 270], 1):
                out = os.path.join(OUTPUT_FOLDER, f"{base}_{i}.png")
                print(f"  Creating side-view snapshot {i} with angle {angle}°...")
                snapshot_pointcloud(pcd, out, rotation_angle=angle)
                print(f"  Snapshot {i}: {out}")
            
            # Generate top-down snapshot as the 4th image
            out = os.path.join(OUTPUT_FOLDER, f"{base}_4.png")
            print(f"  Creating top-down snapshot 4...")
            snapshot_pointcloud(pcd, out, top_down=True)
            print(f"  Snapshot 4 (top-down): {out}")
            
            print(f"Completed processing {fname}")
            
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()

    print("Script completed!")

if __name__ == "__main__":
    main()