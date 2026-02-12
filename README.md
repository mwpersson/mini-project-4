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
- **PyTorch** with custom `nn.Module` classes and manual training loops

### Dataset
- **Fashion-MNIST**: 60,000 training images, 10,000 test images, 28×28 grayscale, 10 classes
- Split: 50,000 train / 10,000 validation / 10,000 test

### Model Architectures (3 Experiments)

| Model | Architecture | Activation | Regularization | Optimizer |
|-------|-------------|------------|----------------|-----------|
| V1 (Baseline) | 784→256→10 | ReLU | None | Adam (lr=0.001) |
| V2 (Improved) | 784→512→256→10 | ReLU | Dropout(0.3) | Adam (lr=0.001) |
| V3 (Best) | 784→512→256→128→10 | LeakyReLU | BatchNorm + Dropout(0.3) | AdamW (lr=0.001) + StepLR |

### Key Findings
- **V3 achieves >85% test accuracy**, meeting the business requirement
- Batch normalization + dropout combination provides the best regularization
- LeakyReLU provides marginal improvement over ReLU for deeper networks
- AdamW with learning rate scheduling helps convergence

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

1. **Deploy V3 model** with 80% confidence threshold for auto-classification
2. **Improve photography guidelines** for Shirt vs T-shirt (require collar/button visibility) and Coat vs Pullover (require side-view photos)
3. **Prioritize human review** for high-cost category pairs
4. **Future work**: CNN architectures, higher-resolution images, data augmentation, transfer learning

## Setup and Running Instructions

### Option 1: Google Colab (Recommended)
1. Open `notebooks/fashion_classifier.ipynb` in Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells

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
│   ├── model.py                   # nn.Module model definitions
│   ├── train.py                   # Custom training loop
│   └── utils.py                   # Helper/utility functions
└── results/                       # Generated after running notebook
├── training_curves.png
├── confusion_matrix.png
├── confidence_threshold.png
└── misclassified_examples.png
```
## Team Member Contributions

| Team Member | Contributions |
|-------------|---------------|
| Luying Cai | Confidence threshold analysis, all written analysis and documentation, business recommendations, per-class metrics, figure exports, and Colab integration. |
| Michael Persson | core framework (model, training, confusion matrix, cost analysis, misclassification visualization) and achieved >85% accuracy.  |

## License

This project is for educational purposes as part of COMP 9130 coursework.
