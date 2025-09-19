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

## Logs
```
Epoch 1
Train: Loss=0.1225 Batch_id=468 Accuracy=89.23: 100%|██████████| 469/469 [00:20<00:00, 22.74it/s]
Test set: Average loss: 0.0007, Accuracy: 9715/10000 (97.15%)

Epoch 2
Train: Loss=0.0647 Batch_id=468 Accuracy=96.42: 100%|██████████| 469/469 [00:21<00:00, 22.31it/s]
Test set: Average loss: 0.0004, Accuracy: 9852/10000 (98.52%)

Epoch 3
Train: Loss=0.0659 Batch_id=468 Accuracy=97.33: 100%|██████████| 469/469 [00:21<00:00, 22.26it/s]
Test set: Average loss: 0.0003, Accuracy: 9888/10000 (98.88%)

Epoch 4
Train: Loss=0.0311 Batch_id=468 Accuracy=97.61: 100%|██████████| 469/469 [00:19<00:00, 23.95it/s]
Test set: Average loss: 0.0003, Accuracy: 9867/10000 (98.67%)

Epoch 5
Train: Loss=0.1303 Batch_id=468 Accuracy=97.86: 100%|██████████| 469/469 [00:19<00:00, 23.59it/s]
Test set: Average loss: 0.0003, Accuracy: 9875/10000 (98.75%)

Epoch 6
Train: Loss=0.0428 Batch_id=468 Accuracy=98.14: 100%|██████████| 469/469 [00:20<00:00, 22.51it/s]
Test set: Average loss: 0.0003, Accuracy: 9892/10000 (98.92%)

Epoch 7
Train: Loss=0.0116 Batch_id=468 Accuracy=98.20: 100%|██████████| 469/469 [00:20<00:00, 22.39it/s]
Test set: Average loss: 0.0002, Accuracy: 9915/10000 (99.15%)

Epoch 8
Train: Loss=0.0347 Batch_id=468 Accuracy=98.29: 100%|██████████| 469/469 [00:21<00:00, 22.09it/s]
Test set: Average loss: 0.0002, Accuracy: 9913/10000 (99.13%)

Epoch 9
Train: Loss=0.0410 Batch_id=468 Accuracy=98.50: 100%|██████████| 469/469 [00:21<00:00, 22.20it/s]
Test set: Average loss: 0.0002, Accuracy: 9911/10000 (99.11%)

Epoch 10
Train: Loss=0.0657 Batch_id=468 Accuracy=98.39: 100%|██████████| 469/469 [00:20<00:00, 23.09it/s]
Test set: Average loss: 0.0002, Accuracy: 9910/10000 (99.10%)

Epoch 11
Train: Loss=0.0127 Batch_id=468 Accuracy=98.54: 100%|██████████| 469/469 [00:20<00:00, 23.06it/s]
Test set: Average loss: 0.0002, Accuracy: 9923/10000 (99.23%)

Epoch 12
Train: Loss=0.0414 Batch_id=468 Accuracy=98.52: 100%|██████████| 469/469 [00:20<00:00, 23.16it/s]
Test set: Average loss: 0.0003, Accuracy: 9887/10000 (98.87%)

Epoch 13
Train: Loss=0.0355 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:20<00:00, 22.49it/s]
Test set: Average loss: 0.0002, Accuracy: 9922/10000 (99.22%)

Epoch 14
Train: Loss=0.0526 Batch_id=468 Accuracy=98.64: 100%|██████████| 469/469 [00:21<00:00, 22.22it/s]
Test set: Average loss: 0.0002, Accuracy: 9929/10000 (99.29%)

Epoch 15
Train: Loss=0.0440 Batch_id=468 Accuracy=98.68: 100%|██████████| 469/469 [00:20<00:00, 22.44it/s]
Test set: Average loss: 0.0002, Accuracy: 9945/10000 (99.45%)

Epoch 16
Train: Loss=0.1297 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:20<00:00, 22.63it/s]
Test set: Average loss: 0.0002, Accuracy: 9909/10000 (99.09%)

Epoch 17
Train: Loss=0.0638 Batch_id=468 Accuracy=98.71: 100%|██████████| 469/469 [00:20<00:00, 23.28it/s]
Test set: Average loss: 0.0002, Accuracy: 9917/10000 (99.17%)

Epoch 18
Train: Loss=0.0748 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:19<00:00, 23.55it/s]
Test set: Average loss: 0.0002, Accuracy: 9930/10000 (99.30%)

Epoch 19
Train: Loss=0.0222 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:20<00:00, 23.23it/s]
Test set: Average loss: 0.0002, Accuracy: 9915/10000 (99.15%)

Epoch 20
Train: Loss=0.0316 Batch_id=468 Accuracy=98.93: 100%|██████████| 469/469 [00:21<00:00, 22.32it/s]
Test set: Average loss: 0.0001, Accuracy: 9947/10000 (99.47%)
```

## Conclusion

The network demonstrates how careful design using **batch normalization**, **dropout**, **GAP**, and a compact architecture (**C-C-C-M-C-C-M-C-C-GAP**) can yield near state-of-the-art results on MNIST with minimal parameters and fast training.
