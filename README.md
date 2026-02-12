# Mini Project 4: Deep Learning Classifier

Course: COMP 9130 - Applied Artificial Intelligence
Team Members: Savina Cai, Michael Persson

## Business Problem

**StyleSort** is an online fashion retailer processing over 100,000 product listings per month with a 32% return rate — significantly higher than the industry average of 20%. Customer surveys reveal that 40% of returns happen because "the item wasn't what I expected," often due to products being miscategorized in the catalog.

This project builds a **product image classification system** using deep learning to:
1. Automatically categorize new product images into one of 10 clothing categories
2. Flag low-confidence predictions for human review before publishing
3. Identify problematic category pairs to improve product descriptions and photography guidelines

## Approach and Methodology

### Framework
- **PyTorch** with custom `nn.Module` class and manual training loop

### Dataset
- **Fashion-MNIST**: 60,000 training images, 10,000 test images, 28×28 grayscale, 10 classes
- Split: 54,000 train / 6,000 validation / 10,000 test

### Model Architecture

We define a flexible `FashionClassifier` class using `nn.Module` that supports configurable depth, width, activation function, batch normalization, and dropout. This allows systematic comparison of different configurations.

### Experiments (5 Configurations)

| Classifier | Layers | Activation | Hidden Size | Batch Norm | Optimizer |
|-----------|--------|-----------|-------------|-----------|-----------|
| 1 (Best) | 3 | ELU | 512 | Yes | Adam |
| 2 | 3 | ReLU | 256 | No | Adam |
| 3 | 2 | ReLU | 256 | No | Adam |
| 4 | 3 | ReLU | 128 | Yes | Adam |
| 5 | 2 | ELU | 256 | Yes | AdamW |

These configurations were selected based on a prior hyperparameter grid search over hidden layers, activation functions, hidden sizes, optimizers, batch normalization, and dropout rates.

### Key Findings
- **Classifier 1 achieves >85% test accuracy**, meeting the business requirement
- Batch normalization significantly improved training stability and final accuracy
- ELU activation provided a slight edge over ReLU in deeper architectures
- Larger hidden sizes (512) outperformed smaller ones when combined with batch normalization

## Results Summary

### Confusion Analysis
- **Most confused pairs**: Shirt ↔ T-shirt/top, Coat ↔ Pullover
- These align with the business context: visually similar categories with different customer expectations

### Cost-Weighted Accuracy
- Standard accuracy and cost-weighted accuracy are both reported
- High-cost errors (Shirt↔T-shirt, Coat↔Pullover) are prioritized in the analysis

### Confidence Threshold
- At 80% confidence threshold, the model achieves higher accuracy on accepted predictions
- Remaining items are flagged for human review

## Business Recommendations

1. **Deploy best model** with 80% confidence threshold for auto-classification
2. **Improve photography guidelines** for Shirt vs T-shirt (require collar/button visibility) and Coat vs Pullover (require side-view photos)
3. **Prioritize human review** for high-cost category pairs
4. **Future work**: CNN architectures, higher-resolution images, data augmentation, transfer learning

## Setup and Running Instructions

### Option 1: Google Colab (Recommended)
1. Open `notebooks/fashion_classifier.ipynb` in Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells — the notebook is fully self-contained

### Option 2: Local Setup
```bash
# Clone the repository
git clone <repo-url>
cd mini-project-4

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/fashion_classifier.ipynb
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```

## Project Structure

```
mini-project-4/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                          # .gitignored (downloaded at runtime)
├── notebooks/
│   └── fashion_classifier.ipynb   # Main Colab notebook (run this)
├── src/
│   ├── __init__.py
│   ├── model.py                   # nn.Module model definition
│   ├── train.py                   # Custom training loop
│   └── utils.py                   # Helper/utility functions
└── results/                       # Generated after running notebook
├── training_curves.png
├── confusion_matrix.png
├── confidence_threshold.png
└── misclassified_examples.png
```

## Team Member Contributions

- **[Michael Persson]:** Model architecture design, hyperparameter grid search, training pipeline, confusion matrix visualization, cost matrix and cost-weighted accuracy analysis, misclassification visualization
- **[Savina Cai]:** Confidence threshold analysis and dual-axis plot, all markdown documentation and methodology narrative, per-class precision/recall/F1 metrics, business recommendations for StyleSort, confidence scores on misclassification visualization, figure exports to results directory, Colab integration
