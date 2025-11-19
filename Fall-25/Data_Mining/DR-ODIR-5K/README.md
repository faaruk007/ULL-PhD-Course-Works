# Automated Diabetic Retinopathy Detection from Color Fundus Images Using Transfer Learning: A Multi-Architecture Evaluation

This project implements transfer learning models for detecting diabetic retinopathy (DR) from fundus images using the ODIR-5K dataset. The approach employs multiple pre-trained architectures with configurable parameter freezing for fine-tuning.

## Overview

The pipeline extracts SLO fundus images from NPZ-formatted ODIR data and trains several deep learning models with class weight balancing to handle data imbalance. The training includes early stopping, learning rate scheduling, and comprehensive evaluation metrics including sensitivity, specificity and gender-stratified AUC analysis.

## Features

- **Data Processing**: Extracts and preprocesses fundus images from NPZ format.
- **Multiple Architectures**: Supports VGG-16, ResNet-50, DenseNet-121, EfficientNet-B0, and Vision Transformer.
- **Transfer Learning**: Flexible parameter freezing strategy (configurable percentage).
- **Class Balancing**: Weighted loss for imbalanced datasets.
- **Augmentation**: Random flips, rotations, and color jittering for train set.
- **Evaluation**: Comprehensive metrics including AUC, sensitivity, specificity, and gender-stratified analysis.
- **Visualization**: Training curves and AUC matrices for result analysis.
- **GPU Acceleration**: Full CUDA support for faster training.

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

### Setup Option 1: Kaggle Notebooks (Recommended)

This is the primary environment where the model has been developed and tested on Tesla P100-PCIE-16GB GPU.

**Steps:**

1. **Go to Kaggle** and create a new notebook
2. **Add the dataset** (ODIR-5K):
   - Download the original datasets and upload it to Kaggle
   - Click **Add Data** → Select ODIR-5K

3. **Upload the notebook** or create cells with:
   ```python
   # Install dependencies (run once)
   !pip install -q torch torchvision timm scikit-learn pandas matplotlib seaborn tqdm Pillow
   ```

4. **In Kaggle notebook, update paths:**
   ```python
   CONFIG = {
       'DATA_DIR': '/kaggle/input/odir-dataset/ODIR_Data',  # Kaggle dataset path
       'SAVE_DIR': '/kaggle/working',  # Kaggle output folder
       # ... rest of config
   }
   ```

5. **Run all cells** - Kaggle handles GPU allocation automatically

**Kaggle Environment Specs:**
- GPU: Tesla P100-PCIE-16GB (16GB VRAM)
- CPU: Dual-core (4 vCPU available)
- RAM: 30GB system memory
- No installation required - all libraries pre-installed
- Batch size: 64 (optimal for P100 memory)

---

### Setup Option 2: Local Machine (Similar Config)

For running on your local machine with comparable GPU specifications.

#### **2a. Ubuntu/Linux Setup:**

```bash
# Clone the repository
git clone https://github.com/faaruk007/ULL-PhD-Course-Works.git
cd ULL-PhD-Course-Works/Fall-25/Data_Mining/DR-ODIR-5K

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU setup
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Launch Jupyter
jupyter notebook Full_Python_Scripts.ipynb
```

#### **2b. Windows Setup:**

```bash
# Clone the repository
git clone https://github.com/faaruk007/ULL-PhD-Course-Works.git
cd ULL-PhD-Course-Works\Fall-25\Data_Mining\DR-ODIR-5K

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU setup
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Launch Jupyter
jupyter notebook Full_Python_Scripts.ipynb
```

#### **2c. macOS Setup:**

```bash
# Clone the repository
git clone https://github.com/faaruk007/ULL-PhD-Course-Works.git
cd ULL-PhD-Course-Works/Fall-25/Data_Mining/DR-ODIR-5K

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CPU or Apple Silicon GPU)
pip install --upgrade pip
pip install -r requirements.txt

# Note: For M1/M2 Macs, consider using: pip install torch::mps

# Launch Jupyter
jupyter notebook Full_Python_Scripts.ipynb
```

---

### Local Machine Requirements (Similar to Kaggle)

For optimal performance matching Kaggle environment:

| Component | Kaggle | Local (Recommended) | Local (Minimum) |
|-----------|--------|-------------------|-----------------|
| GPU | Tesla P100 (16GB) | RTX 3080/3090/4080 (10GB+) | RTX 2060 (6GB) |
| CUDA | 12.0+ | 12.1+ | 11.8+ |
| cuDNN | 8.0+ | 8.0+ | 8.0+ |
| RAM | 30GB | 16GB+ | 8GB |
| Storage | 50GB+ | 50GB+ | 30GB+ |
| Python | 3.11 | 3.9-3.11 | 3.8-3.11 |

---

### Configuration for Local Machine

Update `CONFIG` in the notebook for your local paths:

```python
CONFIG = {
    'DATA_DIR': '/path/to/ODIR_Data',  # Linux/Mac: ~/datasets/ODIR_Data
                                        # Windows: C:\\datasets\\ODIR_Data
    'SAVE_DIR': './checkpoints',       # Local checkpoints folder
    'IMG_SIZE': 224,
    'BATCH_SIZE': 64,  # Adjust if GPU OOM (try 32 or 16)
    'NUM_EPOCHS': 100,
    'LEARNING_RATE': 3e-4,
    'WEIGHT_DECAY': 1e-3,
    'FREEZE_PCT': 0.5,
    'NUM_WORKERS': 4,  # Adjust based on CPU cores
    'EARLY_STOPPING_PATIENCE': 8,
    'LR_PATIENCE': 5,
    'LR_FACTOR': 0.5,
}
```

---

### GPU Setup Verification

After installation, verify your GPU setup:

```bash
# Check PyTorch GPU availability
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}")
EOF
```

Expected output:
```
PyTorch version: 2.0.0
CUDA available: True
GPU: Tesla P100-PCIE-16GB  (or your local GPU)
GPU Memory: 16.00 GB
CUDA version: 12.1
```

---

### Troubleshooting Installation

**If CUDA is not detected:**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**If dependencies conflict:**
```bash
# Create fresh virtual environment
deactivate
rm -rf venv  # or rmdir venv on Windows
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
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

**Dataset Format:** NPZ files containing:
- `slo_fundus`: Fundus image array
- `dr_class`: Diabetic retinopathy classification label (0 or 1)
- `male`: Gender information (optional, 0=female, 1=male)
- And other metadata information
**Kaggle Dataset Path:**
```
/kaggle/input/odir-dataset/ODIR_Data/
├── train/
├── val/
└── test/
```

**Local Dataset Path:**
```
~/datasets/ODIR_Data/  (or your custom path)
├── train/
├── val/
└── test/
```

## Directory Structure

```
ULL-PhD-Course-Works/
└── Fall-25/
    └── Data_Mining/
        └── DR-ODIR-5K/
            ├── Full_Python_Scripts.ipynb    # Main training notebook
            ├── README.md                     # This file
            ├── requirements.txt              # Python dependencies
```

**On Kaggle:** Files are saved to `/kaggle/working/` and available for download after notebook execution.

**On Local Machine:** Create an `outputs/` folder in the project directory for checkpoints and results.

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

### Kaggle Notebook Workflow (Primary)

This is the recommended and tested environment.

**Quick Start on Kaggle:**

1. Create a new Kaggle Notebook
2. Add the ODIR-5K dataset (Search: "ODIR-5K" → Add Data)
3. Copy the notebook code to Kaggle cells
4. Update the CONFIG:
   ```python
   CONFIG = {
       'DATA_DIR': '/kaggle/input/odir-dataset/ODIR_Data',
       'SAVE_DIR': '/kaggle/working',
       # ... rest remains same
   }
   ```
5. Run all cells (Kaggle automatically allocates Tesla P100 GPU)
6. Download results from `/kaggle/working/`

**Kaggle Tips:**
- No GPU memory issues - P100 has 16GB VRAM
- Data is mounted automatically
- Use `!pip install` in cells for any missing packages
- Results saved to `/kaggle/working/` are downloadable
- Session timeout: 9 hours (sufficient for 100 epochs)

**Expected Training Time on Kaggle P100:**
- ResNet50: ~2-3 hours for 100 epochs
- EfficientNet-B0: ~1.5-2 hours
- ViT: ~3-4 hours

---

### Local Machine Workflow

#### **Running the Notebook**

```bash
jupyter notebook Full_Python_Scripts.ipynb
```

#### **Key Steps**

1. Update `CONFIG` dictionary with your data paths
2. Adjust hyperparameters as needed
3. Run all cells sequentially
4. Models will be saved to `SAVE_DIR` automatically
5. Review generated visualizations and metrics

#### **For Different Operating Systems:**

**Linux/macOS:**
```bash
# Activate environment
source venv/bin/activate

# Run notebook
jupyter notebook Full_Python_Scripts.ipynb

# Or run as Python script
python Full_Python_Scripts.ipynb --to script --output full_script.py
python full_script.py
```

**Windows PowerShell:**
```bash
# Activate environment
venv\Scripts\Activate.ps1

# Run notebook
jupyter notebook Full_Python_Scripts.ipynb
```

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

### Kaggle Environment (Current/Tested)

- **GPU**: Tesla P100-PCIE-16GB (16GB VRAM) ✅ Verified working
- **CPU**: Dual-core (4 vCPU available)
- **RAM**: 30GB system memory
- **Storage**: 50GB+ for dataset
- **Python**: 3.11 (pre-installed)
- **CUDA/cuDNN**: Pre-configured
- **Training Time**: 2-4 hours for 100 epochs

### Local Machine Recommendations

| Spec | Recommended | Minimum | Note |
|------|-------------|---------|------|
| **GPU** | RTX 3080/3090/4080 (10GB+) | RTX 2060 (6GB) | Matches Kaggle P100 performance |
| **VRAM** | 10GB+ | 6GB | Adjust batch size if lower |
| **CPU** | Multi-core (8+ cores) | 4-core | For data loading |
| **System RAM** | 16GB+ | 8GB | For smooth operation |
| **Storage** | 50GB+ | 30GB | Dataset + checkpoints |
| **CUDA** | 12.1+ | 11.8+ | Must match PyTorch |
| **cuDNN** | 8.0+ | 8.0+ | For GPU acceleration |
| **Python** | 3.9-3.11 | 3.8+ | Tested on 3.11 |
| **OS** | Any (Linux/Windows/macOS) | Any | Linux recommended for performance |

### GPU Memory Considerations

**With different GPUs:**
- **P100 (16GB)**: Batch size 64 optimal, full training supported ✅
- **RTX 3090 (24GB)**: Batch size 64-128 possible
- **RTX 3080 (10GB)**: Batch size 32-64 recommended
- **RTX 2060 (6GB)**: Batch size 16-32, longer training time

**If you get OOM errors, adjust CONFIG:**
```python
CONFIG = {
    'BATCH_SIZE': 32,  # or 16 if still OOM
    'NUM_WORKERS': 2,  # reduce from 4
}
```

## Performance Tips

- Use GPU for training (automatically detected)
- Adjust `BATCH_SIZE` if OOM errors occur
- Reduce `NUM_WORKERS` if data loading is slow
- Pre-compute augmented images for faster training
- Monitor validation metrics to tune `FREEZE_PCT`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dr_detection_2025,
  title={Diabetic Retinopathy Detection using Transfer Learning on ODIR-5K},
  author={Faruk},
  year={2025},
  url={https://github.com/faaruk007/ULL-PhD-Course-Works/tree/main/Fall-25/Data_Mining/DR-ODIR-5K}
}
```

Or in APA format:
```
Faruk. (2025). Diabetic Retinopathy Detection using Transfer Learning on ODIR-5K. 
Retrieved from https://github.com/faaruk007/ULL-PhD-Course-Works/tree/main/Fall-25/Data_Mining/DR-ODIR-5K
```

## References

- ODIR-5K Dataset: https://www.kaggle.com/datasets/sir05/odir-5k
- ImageNet Pretrained Models: https://pytorch.org/vision/main/models.html
- Transfer Learning Best Practices: https://arxiv.org/abs/1411.1792

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

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

## Acknowledgments

- ODIR-5K dataset curators
- PyTorch and torchvision teams
- TIMM (Hugging Face) for efficient model implementations
