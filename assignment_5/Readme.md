# MNIST CNN Optimized Architecture

This project demonstrates a highly optimized CNN for the MNIST handwritten digits dataset, achieving **99.45% test accuracy** within 20 Epochs and 20k parameters.

## Key Improvements \& Changes

- **Applied the same normalization** to the test set to ensure consistency.
- **Batch size decreased**: 128
- **Number of workers increased**: 4 for faster data loading on GPU
- **Optimizer**: Adam
- **Removed StepLR scheduler** to simplify learning rate adjustments
- **Total trainable parameters**: 19,532


## Final Architecture

The model follows a **C-C-C-M-C-C-M-C-C-GAP** pattern:

### Block 1 (C-C-C-M)

- **Conv2d**: 1 → 10 → 14 → 20 channels, 3x3 kernels
- **BatchNorm** after each conv
- **MaxPool2d**(2,2)
- **Dropout2d**(0.05)
- **Receptive Field** after this block: **8x8**


### Block 2 (C-C-M)

- **Conv2d**: 20 → 24 → 20 channels, 3x3 kernels
- **BatchNorm** after each conv
- **MaxPool2d**(2,2)
- **Dropout2d**(0.1)
- **Receptive Field** after this block: **18x18**


### Block 3 (C-C-D)

- **Conv2d**: 20 → 16 channels, 3x3 kernels
- **BatchNorm** after each conv
- **Dropout2d**(0.1)
- **Receptive Field** after this block: **34x34**


### Global Average Pooling + Classifier

- **AdaptiveAvgPool2d**(1)
- **1x1 Conv2d** for 10 classes
- **Flattened output** → log softmax


## Goals Achieved

- ✅ **Test Accuracy**: 99.45% (achieved by 15th epoch)
- ✅ **Total Parameters**: 19,532


## Architecture Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetOptimized(nn.Module):
    def __init__(self):
        super(NetOptimized, self).__init__()

        # --- BLOCK 1 ---
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 14, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(14)
        self.conv3 = nn.Conv2d(14, 20, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout2d(0.05)

        # --- BLOCK 2 ---
        self.conv4 = nn.Conv2d(20, 24, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(24, 20, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout2d(0.1)

        # --- BLOCK 3 ---
        self.conv6 = nn.Conv2d(20, 20, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(20)
        self.conv7 = nn.Conv2d(20, 16, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.drop3 = nn.Dropout2d(0.1)

        # --- GAP + Classifier ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(16, 10, 1)

    def forward(self, x):
        # BLOCK 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # BLOCK 2
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # BLOCK 3
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.drop3(x)

        # GAP + Classifier
        x = self.gap(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return F.log_softmax(x, dim=1)
```


## Conclusion

The network demonstrates how careful design using **batch normalization**, **dropout**, **GAP**, and a compact architecture (**C-C-C-M-C-C-M-C-C-GAP**) can yield near state-of-the-art results on MNIST with minimal parameters and fast training.
