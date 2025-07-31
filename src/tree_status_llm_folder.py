import os
from pathlib import Path
import google.genai as genai
from pydantic import BaseModel
from collections import defaultdict
import re
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import time

class TreeStatus(BaseModel):
    reasoning: str
    id: str
    status: str
    probability: float  # Probability that the tree is dead (0.0 = definitely alive, 1.0 = definitely dead)

# Google Gemini API configuration
API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY_HERE"  # Replace with your actual API key

# Models to test in parallel
MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it"
]

# Number of parallel runs per model
NUM_RUNS = 2 # Reduced to avoid hitting rate limits too quickly

def process_single_tree(client, base_id, image_paths, output_file, model_name, display_name=None, retry_count=0):
    """
    Process a single tree with its multiple perspective images using Google Gemini API.
    Returns True if successful, False if failed.
    """
    if display_name is None:
        display_name = model_name
        
    try:
        # Sort images to ensure consistent order (_1 to _6, then _combined)
        def sort_key(p):
            if p.stem.endswith('_combined'):
                return 7  # Combined image comes last
            else:
                match = re.match(r'.+_([1-6])$', p.stem)
                return int(match.group(1)) if match else 0
        
        image_paths.sort(key=sort_key)
        
        # Only process if we have all seven images (6 perspectives + 1 combined)
        if len(image_paths) != 7:
            print(f"{display_name}: Warning: Tree {base_id} has {len(image_paths)} images instead of 7, skipping.")
            return False
        
        # Prepare image parts for the API request
        image_parts = []
        for img_path in image_paths:
            mime_type = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            
            with img_path.open("rb") as f:
                img_bytes = f.read()
            
            # Determine the label for each image
            if img_path.stem.endswith('_combined'):
                label = "Combined view (all perspectives):"
            else:
                match = re.match(r'.+_([1-6])$', img_path.stem)
                perspective_num = match.group(1) if match else "unknown"
                label = f"Perspective {perspective_num}:"
            
            # Add text label for each perspective
            image_parts.append(
                {"text": label}
            )
            
            # Add the image data
            image_parts.append(
                {"inline_data": {
                    "mime_type": mime_type,
                    "data": img_bytes,
                }}
            )

        # Create the prompt text with all perspectives
        prompt = (
            f"You are analyzing a tree with ID: {base_id}. "
            f"""You are given seven different images of the same tree: five side views (perspectives 1-5, taken 72 degrees apart), one top view (perspective 6), and one combined view showing all perspectives together.

            CRITICAL ANALYSIS GUIDELINES:
            1. AERIAL PHOTOGRAMMETRY CONTEXT: These point clouds were captured from aerial view in early June 2024 (peak growing season).

            2. TREE vs BUSH IDENTIFICATION:
            - If green points appear at the apparent "top" with nothing above them, that IS the tree crown (even if it appears low - it's simply a small tree)
            - If bare branches extend above green foliage, the green foliage is not the tree crown but probably bushes
            - Ground-level vegetation around the trunk should NOT be confused with tree foliage

            3. CANOPY DENSITY INTERPRETATION:
            - Dense canopies may show green at top and bottom but little visible in the middle
            - This is NORMAL for healthy trees and indicates thick foliage blocking interior view
            - Be careful with interpreting missing middle points as death indicators

            4. DEATH INDICATORS:
            - Absence of green points in the upper canopy area
            - Structural branches visible with no foliage coverage

            5. ALIVE INDICATORS:
            - Green foliage visible in expected crown area
            - Dense green clusters indicating active growth
            - June timing means trees should show full leaf coverage if alive in either the top or middle part of the tree
            - If the tree is dead, the green points should be absent or very sparse in the upper canopy area

            Using all seven images for context, classify whether the tree is Alive or Dead. 
            Respond in JSON with fields: 
            - id (tree ID as string)
            - reasoning (detailed step-by-step analysis of key factors observed)
            - status (Alive/Dead) 
            - probability (0.0 = definitely alive, 1.0 = definitely dead, be conservative with high probabilities)

            Think step by step"""
        )

        # Send all perspectives to the LLM in one request
        response = client.models.generate_content(
            model=model_name,
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        *image_parts
                    ]
                }
            ],
            config={
                "temperature": 0,
            }
        )

        # Parse the response text manually since Gemma models don't support structured output
        response_text = response.text
        print(f"{display_name}: Raw response: {response_text}")

        # Try to parse JSON from the response
        try:
            # Look for JSON in the response (might be wrapped in markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                # Find the JSON object in the response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            # Parse the JSON
            result_dict = json.loads(json_text)
            
            # Create TreeStatus object manually with robust type conversion
            result = TreeStatus(
                reasoning=str(result_dict.get("reasoning", "")),
                id=str(result_dict.get("id", base_id)),  # Convert ID to string
                status=str(result_dict.get("status", "Unknown")),
                probability=float(result_dict.get("probability", 0.5))
            )
            
        except Exception as json_error:
            print(f"{display_name}: Error parsing JSON: {json_error}")
            # Create a fallback result if JSON parsing fails
            result = TreeStatus(
                reasoning=f"Failed to parse response: {response_text[:200]}...",
                id=str(base_id),
                status="Unknown",
                probability=0.5
            )

        print(f"{display_name}: Tree {result.id}: {result.status} (P(dead) = {result.probability:.2f})")
        
        # Save results to file (with thread-safe writing)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Tree ID: {base_id}\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Raw LLM Response: {response_text}\n")
            f.write(f"Parsed Result: Tree {result.id} - Status: {result.status} - P(dead): {result.probability:.2f}\n")
            if hasattr(result, 'reasoning') and result.reasoning:
                f.write(f"Reasoning: {result.reasoning}\n")
            f.write("-" * 40 + "\n\n")
        
        return True
        
    except Exception as e:
        error_str = str(e)
        print(f"{display_name}: Error processing tree {base_id}: {error_str}")
        
        # Check if it's a rate limit error and retry
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            if retry_count < 3:  # Max 3 retries
                # Extract retry delay from error message if available
                wait_time = 60  # Default wait time
                if "retryDelay" in error_str:
                    try:
                        # Try to extract the retry delay
                        delay_match = re.search(r"'retryDelay': '(\d+)s'", error_str)
                        if delay_match:
                            wait_time = int(delay_match.group(1)) + 5  # Add 5 seconds buffer
                    except:
                        pass
                
                print(f"{display_name}: Rate limit hit, waiting {wait_time}s before retry {retry_count + 1}/3...")
                time.sleep(wait_time)
                return process_single_tree(client, base_id, image_paths, output_file, model_name, display_name, retry_count + 1)
            else:
                print(f"{display_name}: Max retries reached for tree {base_id}")
        
        # Log the error to the output file (with thread-safe writing)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"ERROR processing Tree ID: {base_id}\n")
            f.write(f"Error Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error Message: {error_str}\n")
            f.write(f"Retry Count: {retry_count}\n")
            f.write("-" * 40 + "\n\n")
        return False

def analyze_trees_with_model_and_run(model_name, run_number):
    """
    Analyze trees using a specific model and run number, processing one tree at a time sequentially.
    """
    print(f"Starting analysis with model: {model_name}, Run: {run_number}")
    
    # Create client for this thread
    client = genai.Client(api_key=API_KEY)

    # Path to the folder containing tree images
    folder = Path("Your path to/TreeswithID")  # Update this path to your TreeswithID folder

    # Path to the output file for this specific model and run
    output_file = Path(f"Your path to/Output/{model_name}_Output{run_number}.txt")  # Update this path to your output folder

    # Initialize or clear the output file for this model and run
    print(f"Initializing output file for {model_name} Run {run_number}: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Tree Health Analysis Results - {model_name} - Run {run_number}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Group images by their base ID (before the _1, _2, _3 suffix or _combined)
    tree_images = defaultdict(list)
    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        
        img_stem = img_path.stem
        
        # Check for individual perspective images (_1 to _6)
        match = re.match(r'(.+)_([1-6])$', img_stem)
        if match:
            base_id = match.group(1)
            tree_images[base_id].append(img_path)
        
        # Check for combined image (_combined)
        elif img_stem.endswith('_combined'):
            base_id = img_stem[:-9]  # Remove '_combined' suffix
            tree_images[base_id].append(img_path)

    print(f"{model_name} Run {run_number}: Found {len(tree_images)} trees to analyze")

    # Convert to sorted list to ensure consistent order across models and runs
    tree_items = sorted(tree_images.items(), key=lambda x: x[0])  # Sort by tree ID
    total_trees = len(tree_items)
    processed_count = 0
    
    # Process trees sequentially one at a time
    for i, (base_id, image_paths) in enumerate(tree_items, 1):
        print(f"{model_name} Run {run_number}: Processing tree {i}/{total_trees} (ID: {base_id})")
        
        try:
            success = process_single_tree(client, base_id, image_paths, output_file, model_name, f"{model_name} Run {run_number}")
            if success:
                processed_count += 1
                print(f"{model_name} Run {run_number}: Tree {base_id} processed successfully. Progress: {processed_count}/{total_trees}")
            else:
                print(f"{model_name} Run {run_number}: Tree {base_id} failed processing.")
                
            # Add delay between requests to respect rate limits
            # With 30 requests/minute limit and 6 parallel jobs, need ~12 second delays
            time.sleep(12)  # 12 second delay to stay well under 30 requests/minute
            
        except Exception as e:
            print(f"{model_name} Run {run_number}: Error processing tree {base_id}: {e}")
            time.sleep(15)  # Longer delay after errors

    # Write completion message to output file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\nAnalysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total trees processed: {processed_count}/{total_trees}\n")
    
    print(f"{model_name} Run {run_number}: Analysis completed. Processed {processed_count}/{total_trees} trees.")

def main():
    """
    Main function to start parallel analysis with multiple Gemma models and multiple runs per model.
    """
    total_jobs = len(MODELS) * NUM_RUNS
    print(f"Starting parallel tree analysis with {len(MODELS)} Gemma models, {NUM_RUNS} runs each...")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Total jobs: {total_jobs}")
    print(f"Output files will be saved as: {', '.join([f'{model}_Output{run}.txt' for model in MODELS for run in range(1, NUM_RUNS + 1)])}")
    
    # Use ThreadPoolExecutor to run model-run combinations with limited parallelism
    # Limit to 6 parallel jobs to better respect 30 requests/minute rate limit
    max_parallel_jobs = min(6, total_jobs)
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        # Submit all model-run combinations
        futures = []
        for model_name in MODELS:
            for run_number in range(1, NUM_RUNS + 1):
                future = executor.submit(analyze_trees_with_model_and_run, model_name, run_number)
                futures.append((f"{model_name} Run {run_number}", future))
        
        # Wait for all runs to complete
        for job_name, future in futures:
            try:
                future.result()  # This will raise any exception that occurred in the thread
                print(f"{job_name} analysis completed successfully!")
            except Exception as e:
                print(f"{job_name} analysis failed with error: {e}")
    
    print(f"All {total_jobs} analysis jobs completed!")
    print(f"Results saved in {len(MODELS) * NUM_RUNS} output files.")

if __name__ == "__main__":
    main()