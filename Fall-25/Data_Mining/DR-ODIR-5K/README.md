# Automated Diabetic Retinopathy Detection from Color Fundus Images Using Transfer Learning: A Multi-Architecture Evaluation

This project implements transfer learning models for detecting diabetic retinopathy (DR) from fundus images using the ODIR-5K dataset. The approach employs multiple pre-trained architectures with configurable parameter freezing for fine-tuning.

## Overview

The pipeline extracts SLO fundus images from NPZ-formatted ODIR data and trains several deep learning models with class weight balancing to handle data imbalance. The training includes early stopping, learning rate scheduling, and comprehensive evaluation metrics including sensitivity, specificity and gender-stratified AUC analysis.


Complete Transfer Learning Pipeline: <img width="1525" height="652" alt="to git" src="https://github.com/user-attachments/assets/4295bdae-d4b2-4eb6-9c9a-ee989aa2acc3" />

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

3. **Upload the notebook** [Full_Python_Scripts.ipynb]:  **[Full_Python_Scripts.ipynb](https://github.com/faaruk007/ULL-PhD-Course-Works/blob/main/Fall-25/Data_Mining/DR-ODIR-5K/Full_Python_Scripts.ipynb)**


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

  
**Kaggle Dataset Path:**
```
/kaggle/input/odir-dataset/ODIR_Data/
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

1. **VGG16** - Deep convolutional network with 16 layers!

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
    
![Main Architecture] 



## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **Gender-Stratified AUC**: Fairness assessment across male/female subgroups
- **Classification Report**: Per-class precision, recall, and F1-score


## Performance Tips

- Use GPU for training (automatically detected)
- Adjust `BATCH_SIZE` if OOM errors occur
- Pre-compute augmented images for faster training
- Monitor validation metrics to tune `FREEZE_PCT`


## References

1.  ODIR-5K Dataset: https://odir2019.grand-challenge.org/dataset/
2.  ImageNet Pretrained Models (Pytorch): https://pytorch.org/vision/main/models.html
3.  K. Simonyan and A. Zisserman, “Very deep convolutional networks for
large-scale image recognition,” in International Conference on Learning
Representations, 2015
4.  K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 2016, pp. 770–778. 
5. G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely
connected convolutional networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 4700–4708.
6.  Efficientnet:M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in International Conference on Machine
Learning. PMLR, 2019, pp. 6105–6114.

7. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al.,
“An image is worth 16x16 words: Transformers for image recognition
at scale,” arXiv preprint arXiv:2010.11929, 2020.

## License

This project is licensed under the MIT License - see LICENSE file for details.


## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

## Acknowledgments

- ODIR-5K dataset curators
- PyTorch and torchvision


