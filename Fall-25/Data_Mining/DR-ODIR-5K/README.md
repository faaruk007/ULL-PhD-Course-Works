# Diabetic Retinopathy Detection from ODIR-5K Dataset

This project implements transfer learning models for detecting diabetic retinopathy (DR) from fundus images using the ODIR-5K dataset. The approach employs multiple pre-trained architectures with configurable parameter freezing for fine-tuning.

## Overview

The pipeline extracts SLO fundus images from NPZ-formatted ODIR data and trains several deep learning models with class weight balancing to handle data imbalance. The training includes early stopping, learning rate scheduling, and comprehensive evaluation metrics including sensitivity, specificity, and gender-stratified AUC analysis.

## Features

- **Data Processing**: Extracts and preprocesses fundus images from NPZ format
- **Multiple Architectures**: Supports VGG16, ResNet50, DenseNet121, EfficientNet-B0, and Vision Transformer
- **Transfer Learning**: Flexible parameter freezing strategy (configurable percentage)
- **Class Balancing**: Weighted sampling and loss weighting for imbalanced datasets
- **Augmentation**: Random flips, rotations, and color jittering for train set
- **Evaluation**: Comprehensive metrics including AUC, sensitivity, specificity, and gender-stratified analysis
- **Visualization**: Training curves and confusion matrices for result analysis
- **GPU Acceleration**: Full CUDA support for faster training

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd DR-Detection-ODIR

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Jupyter notebook support (if running as .ipynb)
pip install jupyter ipywidgets
```

## Configuration

Key hyperparameters in the CONFIG dictionary:
- `DATA_DIR`: Path to ODIR dataset directory
- `SAVE_DIR`: Output directory for model checkpoints
- `IMG_SIZE`: 224 (image resolution)
- `BATCH_SIZE`: 64
- `NUM_EPOCHS`: 100
- `LEARNING_RATE`: 3e-4
- `FREEZE_PCT`: 0.5 (freeze 50% of backbone parameters)
- `EARLY_STOPPING_PATIENCE`: 8 epochs
- `LR_PATIENCE`: 5 epochs

## Dataset

The ODIR-5K dataset contains approximately 7,000 fundus images split into:
- Train: 4,476 samples (74.5% no DR, 25.5% DR)
- Validation: 641 samples
- Test: 1,914 samples

**Binary Classification**: DR present (1) vs. no DR (0)

Expected format: NPZ files containing 'slo_fundus' (image), 'dr_class' (label), and optional 'male' (gender)

## Directory Structure

```
DR-Detection-ODIR/
├── Full_Python_Scripts.ipynb    # Main training notebook
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── outputs/
    ├── model_checkpoints/       # Saved best models
    ├── results/                 # Evaluation metrics and visualizations
    └── logs/                    # Training logs
```

## Supported Models

1. **VGG16** - Deep convolutional network with 16 layers
2. **ResNet50** - Residual network with skip connections
3. **DenseNet121** - Dense connections for feature reuse
4. **EfficientNet-B0** - Efficient scaled architecture
5. **Vision Transformer (ViT)** - Transformer-based architecture

## Training Process

1. Data extraction from NPZ files
2. Class weight computation for imbalanced dataset
3. Data augmentation and normalization
4. Model initialization with ImageNet pretrained weights
5. Selective layer freezing for transfer learning
6. Training with weighted cross-entropy loss
7. Validation-based model checkpointing
8. Early stopping and learning rate scheduling
9. Test set evaluation with detailed metrics

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **Gender-Stratified AUC**: Fairness assessment across male/female subgroups
- **Confusion Matrix**: Visual representation of prediction patterns
- **Classification Report**: Per-class precision, recall, and F1-score

## Usage

### Running the Notebook

```bash
jupyter notebook Full_Python_Scripts.ipynb
```

### Key Steps

1. Update `CONFIG` dictionary with your data paths
2. Adjust hyperparameters as needed
3. Run all cells sequentially
4. Models will be saved to `SAVE_DIR` automatically
5. Review generated visualizations and metrics

### Example Output

```
Epoch 100: Train Loss=0.3245, Acc=0.8912, AUC=0.9234 | 
           Val Loss=0.3512, Acc=0.8756, AUC=0.9145
           
ResNet50 - Test Results
Accuracy: 0.8834
AUC: 0.9256
Sensitivity: 0.8745
Specificity: 0.8901
Male AUC: 0.9312 (n=956)
Female AUC: 0.9198 (n=958)
```

## System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: Multi-core processor for data loading
- **RAM**: 16GB+ system memory
- **Storage**: ~50GB for dataset and model checkpoints

## Performance Tips

- Use GPU for training (automatically detected)
- Adjust `BATCH_SIZE` if OOM errors occur
- Reduce `NUM_WORKERS` if data loading is slow
- Pre-compute augmented images for faster training
- Monitor validation metrics to tune `FREEZE_PCT`



## References

- ODIR-5K Dataset: https://www.kaggle.com/datasets/sir05/odir-5k
- ImageNet Pretrained Models: https://pytorch.org/vision/main/models.html


## License

This project is licensed under the MIT License - see LICENSE file for details.

## Troubleshooting

**CUDA Out of Memory (OOM)**
- Reduce `BATCH_SIZE` (e.g., 32 or 16)
- Use smaller model (e.g., EfficientNet-B0 instead of ResNet50)

**Slow Data Loading**
- Reduce `NUM_WORKERS` or set to 0
- Check disk I/O performance

**Model Not Improving**
- Increase `FREEZE_PCT` for more extensive fine-tuning
- Adjust `LEARNING_RATE` (try 1e-4 or 5e-4)
- Check data quality and label distribution


## Acknowledgments

- ODIR-5K dataset curators
- PyTorch and torchvision teams
