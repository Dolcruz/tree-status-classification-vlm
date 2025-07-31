#!/usr/bin/env python3
"""
Tree Health Evaluation Script

Evaluate tree-health predictions from vision language models.

This script:
1. Loads ground truth labels for tree health status
2. Parses model predictions from output files  
3. Calculates comprehensive evaluation metrics
4. Generates confusion matrices and visualizations
5. Outputs detailed performance reports

Usage:
    python tree_eval.py

The script will prompt for model name and automatically find LLMOutput files.
"""

import json, re, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
from collections import Counter, defaultdict
from pathlib import Path

# Ground truth tree status labels (Alive/Dead)
# This dictionary contains the manually verified status for each tree ID
GROUND_TRUTH = {
    "1": "Alive", "2": "Alive", "3": "Alive", "4": "Alive", "5": "Alive",
    "6": "Alive", "7": "Alive", "8": "Alive", "9": "Alive", "10": "Alive",
    "11": "Alive", "12": "Alive", "13": "Alive", "14": "Alive", "15": "Dead",
    "16": "Alive", "17": "Alive", "18": "Alive", "19": "Alive", "20": "Alive",
    "21": "Alive", "22": "Alive", "23": "Alive", "24": "Alive", "25": "Alive",
    "26": "Alive", "27": "Alive", "28": "Alive", "29": "Alive", "30": "Alive",
    "31": "Alive", "32": "Alive", "33": "Alive", "34": "Alive", "35": "Alive",
    "36": "Alive", "37": "Alive",
    # "38": "Alive", # Very unclear hence excluded
    "39": "Alive", "40": "Alive", "41": "Alive", "42": "Alive", "43": "Alive",
    "44": "Alive", "45": "Alive", "46": "Alive", "47": "Alive", "48": "Alive",
    "49": "Alive", "50": "Alive", "51": "Alive", "52": "Alive", "53": "Alive",
    "54": "Alive", "55": "Alive", "56": "Alive", "57": "Alive", "58": "Alive",
    "59": "Alive", "60": "Alive", "61": "Alive", "62": "Alive", "63": "Alive",
    "64": "Alive", "65": "Alive", "66": "Alive", "67": "Alive", "68": "Alive",
    "69": "Alive", "70": "Alive", "71": "Alive", "72": "Alive", "73": "Alive",
    "74": "Alive", "75": "Alive", "76": "Alive", "77": "Alive", "78": "Alive",
    "79": "Alive", "80": "Alive", "81": "Alive", "82": "Alive", "83": "Alive",
    "84": "Alive", "85": "Alive", "86": "Alive", "87": "Alive", "88": "Alive",
    "89": "Alive", "90": "Alive", "91": "Alive", "92": "Alive", "93": "Alive",
    "94": "Alive", "95": "Alive", "96": "Alive", "97": "Alive", "98": "Alive",
    "99": "Alive", "100": "Alive", "101": "Alive", "102": "Alive", "103": "Alive",
    "104": "Alive", "105": "Alive", "106": "Alive", "107": "Alive", "108": "Alive",
    "109": "Alive", "110": "Alive", "111": "Alive", "112": "Alive", "113": "Alive",
    "114": "Alive", "115": "Alive", "116": "Alive", "117": "Alive", "118": "Alive",
    "119": "Alive", "120": "Alive", "121": "Alive", "122": "Alive", "123": "Alive",
    "124": "Alive", "125": "Alive", "126": "Alive", "127": "Alive", "128": "Alive",
    "129": "Alive", "130": "Alive", "131": "Alive", "132": "Alive", "133": "Alive",
    "134": "Alive", "135": "Alive", "136": "Alive", "137": "Alive", "138": "Alive",
    "139": "Alive", "140": "Alive", "141": "Alive", "142": "Alive", "143": "Alive",
    "144": "Alive", "145": "Alive", "146": "Alive", "147": "Alive", "148": "Alive",
    "149": "Alive", "150": "Alive", "151": "Alive", "152": "Alive", "153": "Alive",
    "154": "Alive", "155": "Alive", "156": "Alive", "157": "Alive", "158": "Alive",
    "159": "Alive", "160": "Alive", "161": "Alive", "162": "Alive", "163": "Alive",
    "164": "Alive", "165": "Alive", "166": "Alive", "167": "Alive", "168": "Alive",
    "169": "Alive", "170": "Alive", "171": "Alive", "172": "Alive", "173": "Alive",
    "174": "Alive", "175": "Alive", "176": "Alive", "177": "Alive", "178": "Alive",
    "179": "Alive", "180": "Alive", "181": "Alive", "182": "Alive", "183": "Alive",
    "184": "Alive", "185": "Alive", "186": "Alive", "187": "Alive", "188": "Alive",
    "189": "Alive", "190": "Alive", "191": "Alive", "192": "Alive", "193": "Alive",
    "194": "Alive", "195": "Alive", "196": "Alive", "197": "Alive", "198": "Alive",
    "199": "Alive", "200": "Alive", "201": "Alive", "202": "Alive", "203": "Alive",
    "204": "Alive", "205": "Alive", "206": "Alive", "207": "Alive", "208": "Alive",
    "209": "Alive", "210": "Alive", "211": "Alive", "212": "Alive", "213": "Alive",
    "214": "Alive", "215": "Alive", "216": "Alive", "217": "Alive", "218": "Alive",
    "219": "Alive", "220": "Alive", "221": "Alive", "222": "Alive", "223": "Alive",
    "224": "Alive", "225": "Alive", "226": "Alive", "227": "Alive", "228": "Alive",
    "229": "Alive", "230": "Alive", "231": "Alive", "232": "Alive", "233": "Alive",
    "234": "Alive", "235": "Alive", "236": "Alive", "237": "Alive", "238": "Alive",
    "239": "Alive", "240": "Alive", "241": "Alive", "242": "Alive", "243": "Alive",
    "244": "Alive", "245": "Alive", "246": "Alive", "247": "Alive", "248": "Alive",
    "249": "Alive", "250": "Alive", "251": "Alive", "252": "Alive", "253": "Alive",
    "254": "Alive", "255": "Alive", "256": "Alive", "257": "Alive", "258": "Alive",
    "259": "Alive", "260": "Alive", "261": "Alive", "262": "Alive", "263": "Alive",
    "264": "Alive", "265": "Alive", "266": "Alive", "267": "Alive", "268": "Alive",
    "269": "Alive", "270": "Alive", "271": "Alive", "272": "Alive", "273": "Alive",
    "274": "Alive", "275": "Alive", "276": "Alive", "277": "Alive", "278": "Alive",
    "279": "Alive", "280": "Alive", "281": "Alive", "282": "Alive", "283": "Alive",
    "284": "Alive", "285": "Alive", "286": "Alive", "287": "Alive", "288": "Alive",
    "289": "Alive", "290": "Alive", "291": "Alive", "292": "Alive", "293": "Alive",
    "294": "Alive", "295": "Alive", "296": "Alive", "297": "Alive", "298": "Alive",
    "299": "Alive", "300": "Alive", "301": "Alive", "302": "Alive", "303": "Alive",
    "304": "Alive", "305": "Alive", "306": "Alive",
    # "307": "Alive", # Very unclear hence excluded (Its a Bush)
    "308": "Alive", "309": "Alive", "310": "Alive", "311": "Alive", "312": "Alive",
    "313": "Alive", "314": "Alive", "315": "Alive", "316": "Alive", "317": "Alive",
    "318": "Alive", "319": "Alive", "320": "Alive", "321": "Alive", "322": "Alive",
    "323": "Alive", "324": "Alive", "325": "Alive", "326": "Alive", "327": "Alive",
    "328": "Alive", "329": "Alive", "330": "Alive", "331": "Alive", "332": "Alive",
    "333": "Alive", "334": "Alive", "335": "Alive", "336": "Alive", "337": "Alive",
    "338": "Alive", "339": "Alive", "340": "Alive", "341": "Alive", "342": "Alive",
    "343": "Alive", "344": "Alive", "345": "Alive", "346": "Alive", "347": "Alive",
    "348": "Alive", "349": "Alive", "350": "Alive", "351": "Alive", "352": "Alive",
    "353": "Alive", "354": "Alive", "355": "Alive", "356": "Alive", "357": "Alive",
    "358": "Alive", "359": "Alive", "360": "Alive", "361": "Alive", "362": "Alive",
    "363": "Alive", "364": "Alive", "365": "Alive", "366": "Alive", "367": "Alive",
    "368": "Alive", "369": "Alive", "370": "Alive", "371": "Alive", "372": "Alive",
    "373": "Alive", "374": "Alive", "375": "Alive", "376": "Alive", "377": "Alive",
    "378": "Alive", "379": "Alive", "380": "Alive", "381": "Alive", "382": "Alive",
    "383": "Alive", "384": "Alive", "385": "Alive", "386": "Alive", "387": "Alive",
    "388": "Alive", "389": "Alive", "390": "Alive", "391": "Alive", "392": "Alive",
    "393": "Alive", "394": "Alive", "395": "Alive", "396": "Alive", "397": "Alive",
    "398": "Alive", "399": "Alive", "400": "Alive", "401": "Alive", "402": "Alive",
    "403": "Alive", "404": "Alive", "405": "Alive", "406": "Alive", "407": "Alive",
    "408": "Alive", "409": "Alive", "410": "Alive", "411": "Alive", "412": "Alive",
    "413": "Alive", "414": "Alive", "415": "Alive", "416": "Alive", "417": "Alive",
    "418": "Alive", "419": "Alive", "420": "Alive", "421": "Alive", "422": "Alive",
    "423": "Alive", "424": "Alive", "425": "Alive", "426": "Alive", "427": "Alive",
    "428": "Alive", "429": "Alive", "430": "Alive", "431": "Alive", "432": "Alive",
    "433": "Alive", "434": "Alive", "435": "Alive", "436": "Alive", "437": "Alive",
    "438": "Alive", "439": "Alive", "440": "Alive", "441": "Dead", "442": "Alive",
    "443": "Alive", "444": "Alive", "445": "Alive", "446": "Alive", "447": "Alive",
    "448": "Alive", "449": "Alive", "450": "Alive", "451": "Alive", "452": "Alive",
    "453": "Alive", "454": "Alive", "455": "Alive", "456": "Alive", "457": "Alive",
    "458": "Alive", "459": "Alive", "460": "Alive", "461": "Alive", "462": "Alive",
    "463": "Alive", "464": "Alive", "465": "Alive", "466": "Alive", "467": "Alive",
    "468": "Alive", "469": "Alive", "470": "Alive", "471": "Alive", "472": "Alive",
    "473": "Alive", "474": "Alive", "475": "Alive", "476": "Alive", "477": "Alive",
    "478": "Alive", "479": "Alive", "480": "Alive", "481": "Alive", "482": "Alive",
    "483": "Alive", "484": "Alive", "485": "Alive", "486": "Alive", "487": "Alive",
    "488": "Alive", "489": "Alive", "490": "Alive", "491": "Alive", "492": "Alive",
    "493": "Alive", "494": "Alive", "495": "Alive", "496": "Alive", "497": "Alive",
    "498": "Alive", "499": "Alive", "500": "Alive", "501": "Alive", "502": "Alive",
    "503": "Alive", "504": "Alive", "505": "Alive", "506": "Alive", "507": "Alive",
    "508": "Alive", "509": "Alive", "510": "Alive", "511": "Alive", "512": "Alive",
    "513": "Alive", "514": "Alive", "515": "Alive", "516": "Alive", "517": "Alive",
    "518": "Alive", "519": "Dead", "520": "Dead", "521": "Dead", "522": "Dead",
    "523": "Dead", "524": "Dead", "525": "Dead", "526": "Dead", "527": "Dead",
    "528": "Dead", "529": "Dead", "530": "Dead", "531": "Dead", "532": "Dead",
    "533": "Dead", "534": "Dead", "535": "Dead", "536": "Dead", "537": "Dead",
    "538": "Dead", "539": "Dead", "540": "Dead", "541": "Dead", "542": "Dead",
    "543": "Dead", "544": "Dead", "545": "Dead", "546": "Dead", "547": "Dead",
    "548": "Dead", "549": "Dead", "550": "Dead", "551": "Dead", "552": "Dead",
    "553": "Dead", "554": "Dead", "555": "Dead", "556": "Dead", "557": "Dead",
    "558": "Dead", "559": "Dead", "560": "Dead", "561": "Dead", "562": "Dead",
}

def parse_predictions(raw: str, threshold=0.5):
    """Extract JSON blocks containing tree predictions and apply threshold."""
    preds = {}
    probabilities = {}
    full_json_data = {}
    
    for block in re.findall(r"\{[^{}]*\}", raw):
        try:
            obj = json.loads(block)
            _id = str(obj.get("id")).strip()
            status = str(obj.get("status")).strip().lower()
            prob = obj.get("probability", None)
            
            if _id and status:
                # Store original LLM decision and probability
                full_json_data[_id] = obj
                probabilities[_id] = prob if prob is not None else (1.0 if status == 'dead' else 0.0)
                
                # Apply threshold override: if P(dead) < threshold, classify as alive
                if prob is not None and prob < threshold:
                    final_status = 'alive'
                else:
                    final_status = status
                
                preds[_id] = final_status
                
        except json.JSONDecodeError:
            continue
    
    return preds, probabilities, full_json_data

def parse_predictions_from_file(file_path: Path):
    """Parse predictions from a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        return parse_predictions(raw, threshold=0.0)  # Don't apply threshold yet
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}, {}, {}

def average_predictions_across_runs(llm_output_files, threshold=0.5):
    """Read multiple LLMOutput files and average probability scores across runs."""
    all_runs_data = []
    
    # Parse each file
    for i, file_path in enumerate(llm_output_files, 1):
        print(f"Parsing {file_path.name}...")
        preds, probs, json_data = parse_predictions_from_file(file_path)
        all_runs_data.append({
            'predictions': preds,
            'probabilities': probs,
            'json_data': json_data,
            'file': file_path.name
        })
    
    # Collect all unique tree IDs across all runs
    all_tree_ids = set()
    for run_data in all_runs_data:
        all_tree_ids.update(run_data['probabilities'].keys())
    
    # Average probabilities across runs
    averaged_probabilities = {}
    averaged_predictions = {}
    averaged_json_data = {}
    run_counts = {}
    
    for tree_id in all_tree_ids:
        prob_sum = 0.0
        valid_runs = 0
        latest_json = {}
        
        for run_data in all_runs_data:
            if tree_id in run_data['probabilities']:
                prob_sum += run_data['probabilities'][tree_id]
                valid_runs += 1
                latest_json = run_data['json_data'].get(tree_id, {})
        
        if valid_runs > 0:
            # Average probability across valid runs
            avg_prob = prob_sum / valid_runs
            averaged_probabilities[tree_id] = avg_prob
            run_counts[tree_id] = valid_runs
            
            # Apply threshold to averaged probability
            if avg_prob < threshold:
                averaged_predictions[tree_id] = 'alive'
            else:
                averaged_predictions[tree_id] = 'dead'
            
            # Store JSON data with averaged probability
            averaged_json_data[tree_id] = latest_json.copy()
            averaged_json_data[tree_id]['probability'] = avg_prob
            averaged_json_data[tree_id]['runs_count'] = valid_runs
            averaged_json_data[tree_id]['status'] = averaged_predictions[tree_id]
    
    print(f"\\nAveraging complete:")
    print(f"  - Total unique trees: {len(all_tree_ids)}")
    print(f"  - Trees with averaged predictions: {len(averaged_predictions)}")
    
    # Print distribution of run counts
    run_count_dist = Counter(run_counts.values())
    print(f"  - Run count distribution:")
    for count, freq in sorted(run_count_dist.items()):
        print(f"    {count} runs: {freq} trees")
    
    return averaged_predictions, averaged_probabilities, averaged_json_data

def evaluate(preds, truth, probabilities=None):
    """Calculate comprehensive evaluation metrics."""
    classes = {v.lower() for v in truth.values()} | set(preds.values())
    conf = defaultdict(Counter)
    
    # Build confusion matrix
    for _id, t in truth.items():
        p = preds.get(_id, "missing")
        conf[t.lower()][p] += 1
    
    # Extra predictions with no ground truth
    for _id, p in preds.items():
        if _id not in truth:
            conf["unknown"][p] += 1
    
    total = sum(sum(row.values()) for row in conf.values())
    correct = sum(conf[c][c] for c in classes)
    overall_acc = correct / total if total else 0.0
    
    # Calculate per-class metrics
    per_acc, per_precision, per_f1, iou = {}, {}, {}, {}
    for c in classes:
        tp = conf[c][c]
        fp = sum(conf[t][c] for t in conf if t != c)
        fn = sum(conf[c][p] for p in conf[c] if p != c)
        
        # Recall (same as accuracy for per-class)
        per_acc[c] = tp / (tp + fn) if (tp + fn) else 0.0
        
        # Precision
        per_precision[c] = tp / (tp + fp) if (tp + fp) else 0.0
        
        # F1 Score
        precision = per_precision[c]
        recall = per_acc[c]
        per_f1[c] = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        
        # IoU
        iou[c] = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    
    mean_iou = sum(iou.values()) / len(iou) if iou else 0.0
    
    # Calculate AUPRC for binary classification
    auprc_scores = {}
    main_classes = [c for c in classes if c in ['alive', 'dead']]
    
    if len(main_classes) == 2 and probabilities:
        y_true = []
        y_scores = []
        
        for _id, true_label in truth.items():
            if _id in preds and _id in probabilities:
                true_binary = 1 if true_label.lower() == 'dead' else 0
                prob_dead = probabilities[_id] if probabilities[_id] is not None else 0.5
                
                y_true.append(true_binary)
                y_scores.append(prob_dead)
        
        if y_true and len(set(y_true)) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            auprc_dead = auc(recall_curve, precision_curve)
            auprc_scores['dead'] = auprc_dead
            
            y_true_alive = [1 - y for y in y_true]
            y_scores_alive = [1.0 - y for y in y_scores]
            precision_curve_alive, recall_curve_alive, _ = precision_recall_curve(y_true_alive, y_scores_alive)
            auprc_alive = auc(recall_curve_alive, precision_curve_alive)
            auprc_scores['alive'] = auprc_alive
        else:
            auprc_scores = {'alive': 0.0, 'dead': 0.0}
    else:
        auprc_scores = {}
    
    return overall_acc, per_acc, per_precision, per_f1, iou, mean_iou, auprc_scores, conf

def write_false_classifications(predictions, probabilities, full_json_data, truth, script_dir, threshold=0.5):
    """Write false classifications to a separate text file with detailed analysis."""
    false_classifications = []
    
    for tree_id, predicted_status in predictions.items():
        if tree_id in truth:
            true_status = truth[tree_id].lower()
            if predicted_status != true_status:
                original_llm_status = full_json_data.get(tree_id, {}).get('status', 'Unknown').lower()
                prob_dead = probabilities.get(tree_id, None)
                
                false_entry = {
                    "tree_id": tree_id,
                    "predicted": predicted_status.capitalize(),
                    "correct": truth[tree_id],
                    "probability_dead": prob_dead,
                    "original_llm_status": original_llm_status.capitalize(),
                    "threshold_applied": prob_dead is not None and prob_dead < threshold and original_llm_status == 'dead',
                    "llm_response": full_json_data.get(tree_id, {})
                }
                false_classifications.append(false_entry)
    
    # Write detailed false classification report
    false_file = script_dir / "false_classification.txt"
    with open(false_file, "w", encoding="utf-8") as f:
        f.write("FALSE CLASSIFICATIONS REPORT\\n")
        f.write("="*50 + "\\n")
        f.write(f"Total false classifications: {len(false_classifications)}\\n")
        f.write(f"Threshold applied: P(dead) < {threshold} → classify as Alive\\n\\n")
        
        for i, entry in enumerate(false_classifications, 1):
            f.write(f"FALSE CLASSIFICATION #{i}\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"Tree ID: {entry['tree_id']}\\n")
            f.write(f"Final Prediction: {entry['predicted']}\\n")
            f.write(f"Correct Label: {entry['correct']}\\n")
            f.write(f"LLM Original Status: {entry['original_llm_status']}\\n")
            if entry['probability_dead'] is not None:
                f.write(f"P(dead): {entry['probability_dead']:.3f}\\n")
            else:
                f.write("P(dead): Not available\\n")
            
            if entry['threshold_applied']:
                f.write(f"THRESHOLD OVERRIDE: LLM said '{entry['original_llm_status']}' but P(dead)={entry['probability_dead']:.3f} < {threshold}, so classified as 'Alive'\\n")
            
            f.write(f"Error Type: ")
            if entry['predicted'].lower() == 'alive' and entry['correct'].lower() == 'dead':
                f.write("False Negative (predicted Alive, actually Dead)\\n")
            elif entry['predicted'].lower() == 'dead' and entry['correct'].lower() == 'alive':
                f.write("False Positive (predicted Dead, actually Alive)\\n")
            else:
                f.write("Other classification error\\n")
            
            f.write("\\nOriginal LLM JSON Response:\\n")
            f.write(json.dumps(entry['llm_response'], indent=2))
            f.write("\\n")
            
            reasoning = entry['llm_response'].get('reasoning', entry['llm_response'].get('reasoning', 'No reasoning provided'))
            f.write(f"\\nLLM Reasoning:\\n{reasoning}\\n")
            f.write("\\n" + "="*80 + "\\n\\n")
    
    print(f"False classifications report saved as: {false_file}")
    return len(false_classifications)

def main():
    """Main evaluation function."""
    print("Tree Health Evaluation Script")
    print("=" * 40)
    model_name = input("Please enter the model name: ").strip()
    if not model_name:
        print("Error: Model name cannot be empty!")
        sys.exit(1)
    
    # Configuration: Set threshold for P(dead) below which trees are classified as alive
    THRESHOLD = 0.8  # Change this value to experiment with different thresholds
    
    # Set up directories
    script_dir = Path(__file__).parent
    results_dir = Path("Your path to/EvalsResults")  # Update this path to your results folder
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all LLMOutput files in the same directory as this script
    llm_output_files = []
    for i in range(1, 10):  # Check for LLMOutput1.txt to LLMOutput9.txt
        llm_file = script_dir / f"LLMOutput{i}.txt"
        if llm_file.exists():
            llm_output_files.append(llm_file)
    
    # Also check for the original LLMOutput.txt
    original_file = script_dir / "LLMOutput.txt"
    if original_file.exists():
        llm_output_files.append(original_file)
    
    if not llm_output_files:
        print("Error: No LLMOutput files found!")
        print("Looking for files like: LLMOutput1.txt, LLMOutput2.txt, LLMOutput3.txt, or LLMOutput.txt")
        print("Please make sure the files exist in the same directory as this script and try again.")
        sys.exit(1)
    
    print(f"Found {len(llm_output_files)} LLMOutput files:")
    for file in llm_output_files:
        print(f"  - {file.name}")
    print()
    
    # Average predictions across all runs
    predictions, probabilities, full_json_data = average_predictions_across_runs(llm_output_files, threshold=THRESHOLD)
    acc, per_acc, per_precision, per_f1, iou, miou, auprc_scores, conf = evaluate(predictions, GROUND_TRUTH, probabilities)
    
    # Generate and display results
    results_text = []
    results_text.append(f"{'='*60}")
    results_text.append(f"TREE HEALTH EVALUATION RESULTS (AVG@{len(llm_output_files)})")
    results_text.append(f"Model: {model_name}")
    results_text.append(f"{'='*60}")
    results_text.append(f"Input files: {', '.join([f.name for f in llm_output_files])}")
    results_text.append(f"Evaluation method: Averaged probabilities across {len(llm_output_files)} runs")
    results_text.append(f"Threshold: P(dead) < {THRESHOLD} → classify as Alive")
    results_text.append(f"Total trees evaluated: {len([t for t in predictions.keys() if t in GROUND_TRUTH])}")
    results_text.append(f"Overall accuracy: {acc:.3%}")
    results_text.append("Per-class accuracy (recall):")
    for cls, v in per_acc.items():
        results_text.append(f"  {cls.capitalize():>5}: {v:.3%}")
    results_text.append("Per-class precision:")
    for cls, v in per_precision.items():
        results_text.append(f"  {cls.capitalize():>5}: {v:.3%}")
    results_text.append("Per-class F1 score:")
    for cls, v in per_f1.items():
        results_text.append(f"  {cls.capitalize():>5}: {v:.3%}")
    results_text.append("Per-class IoU:")
    for cls, v in iou.items():
        results_text.append(f"  {cls.capitalize():>5}: {v:.3%}")
    results_text.append(f"Mean IoU: {miou:.3%}")
    
    # Display AUPRC scores
    if auprc_scores:
        results_text.append("Area Under Precision-Recall Curve (AUPRC):")
        for cls, score in auprc_scores.items():
            results_text.append(f"  {cls.capitalize():>5}: {score:.3f}")
        mean_auprc = sum(auprc_scores.values()) / len(auprc_scores)
        results_text.append(f"  Mean AUPRC: {mean_auprc:.3f}")
    results_text.append("")
    
    # Print results to console
    for line in results_text:
        print(line)
    
    # Generate confusion matrix visualization
    all_classes = sorted(set(conf.keys()) | {p for row in conf.values() for p in row.keys()})
    conf_matrix = np.zeros((len(all_classes), len(all_classes)))
    for i, true_class in enumerate(all_classes):
        for j, pred_class in enumerate(all_classes):
            conf_matrix[i, j] = conf[true_class][pred_class]
    
    # Create normalized confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
    conf_matrix_percent = conf_matrix_norm * 100
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Absolute counts
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='g', 
                cmap='Blues',
                xticklabels=[cls.capitalize() for cls in all_classes],
                yticklabels=[cls.capitalize() for cls in all_classes],
                cbar_kws={'label': 'Count'},
                ax=ax1)
    ax1.set_title('Confusion Matrix (Absolute Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('Reference Class', fontsize=12)
    
    # Normalized percentages
    sns.heatmap(conf_matrix_percent, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                xticklabels=[cls.capitalize() for cls in all_classes],
                yticklabels=[cls.capitalize() for cls in all_classes],
                cbar_kws={'label': 'Percentage (%)'},
                ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Class', fontsize=12)
    ax2.set_ylabel('Reference Class', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = results_dir / f"{model_name}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as: {plot_file}")
    plt.show()
    
    # Write false classifications analysis
    num_false = write_false_classifications(predictions, probabilities, full_json_data, GROUND_TRUTH, script_dir, threshold=THRESHOLD)
    results_text.append(f"\\nFalse classifications: {num_false}")
    
    # Save comprehensive results
    accuracy_percent = int(acc * 100)
    miou_percent = int(miou * 100)
    results_filename = f"{model_name}_AVG@{len(llm_output_files)}_{accuracy_percent}acc_{miou_percent}miou.txt"
    results_file = results_dir / results_filename
    
    with open(results_file, 'w', encoding='utf-8') as f:
        for line in results_text:
            f.write(line + '\\n')
    
    print(f"Results text saved as: {results_file}")

if __name__ == "__main__":
    main()