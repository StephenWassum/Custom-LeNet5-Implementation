# Custom-LeNet5-Implementation

### Author:
- **Stephen Wassum**  

## Overview

This project implements LeNet5, a Neural Network Architecture built to classify handwritten digits 
using the MNIST dataset. The project includes:

- A full implementation of the original LeNet5 architecture.
- A custom loss function and optimizer (MAP loss and Stochastic Diagonal Levenberg-Marquardt).
- A second model modified to handle unseen, augmented MNIST data.

## How to Run

1. Make sure `test1.py`, `test2.py`, and `data.py` are in the same directory.
2. Run either model with:

   ```
   python test1.py
   python test2.py
   ```

3. Output:
   - Console displays train/test accuracy per epoch.
   - PNG files saved to `P1output/` and `P2output/`:
     - Error rate plots
     - Confusion matrices
     - Most confidently misclassified digits

## Model Files

The Pretrained Models are included as follows:

- Original LeNet5 (`LeNet5_1.pth`):  
- Modified LeNet5 (`LeNet5_2.pth`):  

## Original LeNet5 functions

Key components:
- `MNIST` class: Loads and resizes MNIST images to 32×32.
- `generateDigitBitmaps()`: Creates 7×12 bitmaps for each digit.
- `LeNet5Model`: Implements LeNet5 architecture as per the original paper.
- `MAPLossFunction`: Custom loss from the paper.
- `StochasticDiagonalLevenbergMarquardt`: Custom optimizer from Appendix C.
- `train()`: Runs training, tracks and plots performance.

## Modified LeNet5 structure

Changes made:
- **Data Augmentation**:
  - Random rotation
  - Shifting
  - Flipping
  - Normalization

- **Architecture Updates**:
  - Max pooling instead of average pooling
  - ReLU activation
  - Dropout layers for regularization

- **Training Updates**:
  - Used PyTorch’s CrossEntropyLoss and Adam optimizer
  - Tested custom vs. built-in methods
  - Final version runs 8 epochs to reduce runtime
