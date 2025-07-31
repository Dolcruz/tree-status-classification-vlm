# Tree Status Classification Using Vision Language Models

A Comparative Study in Sandhausen Forest

## Abstract

This repository contains the code and methodology for a novel approach using state-of-the-art vision language models to classify tree health status in the Sandhausen Forest, Germany. We collected 3D point cloud data using drone technology and processed it through various segmentation and classification pipelines. Our methodology involved extracting individual tree point clouds using segmentation algorithms, generating multi-view images, and testing multiple vision language models including Gemini 2.5 Flash, o4-mini, and various Gemma models.

From an initial dataset of 560 labeled trees (516 alive, 44 dead), we generated rasterized views through custom Python scripts, creating seven images per tree (six perspectives plus one combined view) for a total of 3,920 images. The best performing model, Gemini 2.5 Flash, achieved an overall accuracy of 97.86% with a mean IoU of 87.73% when evaluated with multiple inference runs.

## Authors

- Falk Pfisterer - University of Heidelberg
- Runan Duan - University of Heidelberg

## Research Highlights

- **97.86% accuracy** achieved with Gemini 2.5 Flash
- **87.73% mean IoU** for tree health classification
- **3,920 multi-perspective images** generated from 560 trees
- **Comparative evaluation** of 6 different vision language models
- **Novel rasterization approach** for point cloud to image conversion

## Repository Structure

```
tree-status-classification-vlm/
├── README.md                      # This file
├── requirements.txt              # Python dependencies
├── src/
│   ├── pointcloud_to_images.py   # 3D point cloud to image conversion
│   ├── 2drastertest.py          # 2D rasterization approach
│   ├── tree_status_llm_folder.py # Vision language model inference
│   ├── tree_eval.py             # Model evaluation and metrics
│   ├── idgiver.py               # Tree ID assignment utility
│   ├── npytotxt.py              # NumPy to text conversion
│   └── copy_failed_images.py    # Utility for handling failed cases
├── data/
│   ├── pointclouds/             # Raw point cloud data (ASCII format)
│   ├── TreeswithID/             # Generated tree images
│   └── EvalsResults/            # Evaluation results and metrics
├── docs/
│   └── methodology.md           # Detailed methodology documentation
└── examples/
    └── sample_outputs/          # Example outputs and visualizations
```

## Methodology

### Data Collection

Data were collected in June 2024 under leaf-on conditions, covering approximately 11.18 hectares (260m × 430m) section of Sandhausen Forest in Baden-Württemberg, Germany (coordinates: 49.3388°N, 8.6413°E). The dataset consisted of high-resolution point clouds from UAV-borne laser scanning (ULS).

### Point Cloud Processing

1. **Ground Classification**: Simple Morphological Filter (SMRF) from PDAL
2. **Tree Segmentation**: Local Minimal Filter with 5m window size
3. **Individual Tree Delineation**: Region growing algorithm based on Dalponte method
4. **Manual Validation**: Visual inspection and cross-validation between team members

### Image Generation

Each ASCII-formatted point cloud file containing XYZ coordinates and RGB color values was processed to generate:

- **6 perspective views**: Rotated around Z-axis at 72-degree intervals (0°, 72°, 144°, 216°, 288°)
- **1 top-down view**: XY projection
- **1 combined view**: Showing all six perspectives together

The rasterization algorithm uses a grid resolution of 0.05 units and applies color averaging for multiple points within the same grid cell.

### Vision Language Models Evaluated

1. **Gemini 2.5 Flash**: Google's efficient multimodal model (Best performer)
2. **o4-mini**: OpenAI's compact reasoning-vision-language model
3. **Gemma-3-4b, 12b, 27b**: Open-source models with scaled language models
4. **Gemini 2.5 Flash Lite**: Lightweight variant

## Key Results

| Model | Overall Accuracy (%) | Mean IoU (%) | F1-Score (Alive/Dead) |
|-------|---------------------|--------------|----------------------|
| Gemini 2.5 Flash (AVG@3) | 97.86 | 87.73 | 98.83 / 87.50 |
| o4-mini (AVG@1) | 96.61 | 83.17 | 98.13 / 82.35 |
| Gemini 2.5 Flash Lite (AVG@4) | 92.68 | 51.74 | 96.17 / 19.61 |
| Gemma-3-12b (AVG@3) | 92.50 | 50.57 | 96.08 / 16.00 |
| Gemma-3-27b (AVG@3) | 92.14 | 48.23 | 95.90 / 8.33 |
| Gemma-3-4b (AVG@3) | 12.14 | 6.46 | 11.51 / 12.77 |

## Installation and Usage

### Prerequisites

- Python 3.8+
- Google Gemini API access
- PDAL (Point Data Abstraction Library)
- Open3D for 3D visualization

### Installation

```bash
git clone https://github.com/Dolcruz/tree-status-classification-vlm.git
cd tree-status-classification-vlm
pip install -r requirements.txt
```

### Configuration

1. **Update paths** in all Python files:
   - Replace "Your path to/..." placeholders with actual paths
   - Set up your folder structure as shown in Repository Structure

2. **API Configuration**:
   - Add your Google Gemini API key in `tree_status_llm_folder.py`
   - Replace `YOUR_GOOGLE_GEMINI_API_KEY_HERE` with your actual key

### Usage

1. **Generate Images from Point Clouds**:
```bash
python src/pointcloud_to_images.py
```

2. **Run Vision Language Model Inference**:
```bash
python src/tree_status_llm_folder.py
```

3. **Evaluate Results**:
```bash
python src/tree_eval.py
```

## Data Format

### Input Point Clouds

ASCII format with 6 columns:
```
X Y Z R G B
474046.41 5465122.73 156.35 0.545 0.690 0.447
```

Where:
- X, Y, Z: Spatial coordinates
- R, G, B: RGB color values (0.0-1.0 range)

### Ground Truth Labels

Trees are labeled as either "Alive" or "Dead" based on visual assessment of point cloud characteristics with cross-validation between team members.

## Evaluation Metrics

- Overall Accuracy
- Per-class Accuracy (Recall)
- Per-class Precision
- F1 Scores
- Intersection over Union (IoU)
- Mean IoU
- Area Under Precision-Recall Curve (AUPRC)
- Confusion Matrices

## Key Findings

1. **Vision-language models can effectively perform tree health classification** without domain-specific architectures
2. **Larger-scale models significantly outperform smaller variants** in specialized visual tasks
3. **Multi-perspective image generation** enhances classification accuracy
4. **Open-source models show limitations** in identifying dead trees, exhibiting bias toward majority class

## Limitations

- Ground truth labels created by non-forestry experts
- Binary classification simplifies continuous nature of tree health
- Limited to single forest type and geographic location
- Evaluation during optimal conditions (June, leaf-on season)
- Budget constraints limited evaluation to cost-efficient models

## Future Work

- Evaluation on diverse forest types and conditions
- Multi-class health assessments
- Integration of newer model architectures
- Professional forestry validation of ground truth labels
- Extension to other ecological monitoring tasks

## Citation

If you use this work in your research, please cite:

```
Tree Status Classification Using Vision Language Models: A Comparative Study in Sandhausen Forest
Falk Pfisterer, Runan Duan
University of Heidelberg, 2024
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank William for his valuable guidance throughout this research project and for providing the photogrammetric dataset. His weekly consultations and insights on conducting research were instrumental to the project's success. We also acknowledge the University of Heidelberg for providing the educational framework for this research.

## Keywords

Tree Status Classification, Vision Language Models, Forest Monitoring, Point Cloud Rasterization, Photogrammetry, LiDAR, UAV, Ecological Monitoring, Computer Vision, Machine Learning