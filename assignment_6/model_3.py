import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

"""CODE BLOCK: 2"""

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

"""CODE BLOCK: 3"""

# Train data transformations
train_transforms = transforms.Compose([
    # transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    # transforms.Resize((28, 28)),
    transforms.RandomRotation((-7., 7.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

"""CODE BLOCK: 4"""

train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

"""CODE BLOCK: 5"""

batch_size = 64

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

"""CODE BLOCK: 6"""

import matplotlib.pyplot as plt

batch_data, batch_label = next(iter(train_loader))

fig = plt.figure()

for i in range(12):
  plt.subplot(3,4,i+1)
  plt.tight_layout()
  plt.imshow(batch_data[i].squeeze(0), cmap='gray')
  plt.title(batch_label[i].item())
  plt.xticks([])
  plt.yticks([])

"""CODE BLOCK: 7"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # INPUT BLOCK
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # CONV BLOCK 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # POOLING
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 → 14x14

        # CONV BLOCK 2 (lateral growth)
        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv7b = nn.Sequential(
            nn.Conv2d(8, 8, 1, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # 1x1 conv to increase channels before GAP
        self.conv8 = nn.Sequential(
            nn.Conv2d(8, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # GAP + output
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv9 = nn.Conv2d(16, 10, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv7b(x)
        x = self.conv8(x)
        x = self.gap(x)
        x = self.conv9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

"""CODE BLOCK: 8"""

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

"""CODE BLOCK: 9"""

from tqdm import tqdm
import torch
import torch.nn.functional as F

def GetCorrectPredCount(pPrediction, pLabels):
    """
    pPrediction: Tensor of shape [batch_size, num_classes]
        Example: [[2.5, 0.3, -1.0],   # sample 1 logits
                  [-1.0, 0.3, 2.5]]   # sample 2 logits

    pLabels: Tensor of shape [batch_size]
        Example: [0, 2]   # true labels (class indices)

    Process:
      1. pPrediction.argmax(dim=1) → picks class with max logit per sample
         Example: [0, 2]
      2. Compare with pLabels → [True, True]
      3. Sum → number of correct predictions (2 in this case)

    Returns:
      Integer: number of correct predictions in this batch
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
    model.train()  # Enable training mode (dropout/batchnorm active)
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        """
        batch_idx: integer, index of the current batch (0, 1, 2, ...)
        data: tensor of shape [batch_size, channels, height, width]
              Example: [64, 1, 28, 28] for MNIST batch of 64 images
              data[0] = first image in batch, shape [1,28,28]
        target: tensor of shape [batch_size]
                Example: [5, 0, 3, ...] labels for each image in the batch
        This line fetches a **single batch** from the DataLoader each iteration.
        """

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset gradients from previous step

        # 1️⃣ Forward pass: model transforms input → logits
        # Internal call: model(data) → triggers model.forward(data)
        pred = model(data)
        # Example shapes:
        # data: [2,1,28,28], pred: [2,10] (2 images, 10 classes)
        # Each row in pred = raw scores (logits) for each class

        # 2️⃣ Compute loss
        # criterion = nn.CrossEntropyLoss()
        # Internally:
        #    - softmax(pred) → probabilities
        #    - negative log likelihood of true class
        # Example:
        # pred[0] = [2.5, 0.3, -1.0, ...] (logits)
        # target[0] = 0
        # -log(softmax(pred[0])[0]) = contribution to loss check about softmax here: https://www.notion.so/Vision-a80ad2dc4f88489bb3ecc45eac005ea2?source=copy_link#2692ba0a615b808b87a1d7d9d5d0da0c
        # loss = criterion(pred, target)
        loss = F.nll_loss(pred, target)
        train_loss += loss.item()  # accumulate scalar loss

        # 3️⃣ Backward pass: compute gradients
        loss.backward()

        # 4️⃣ Optimizer step: update weights
        optimizer.step()

        # 5️⃣ Track training accuracy
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        # Update progress bar
        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} '
                                   f'Batch_id={batch_idx} '
                                   f'Accuracy={100*correct/processed:0.2f}')

    # Store epoch-level metrics
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()  # Evaluation mode (dropout/batchnorm frozen)

    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (data, target) in enumerate(test_loader):
            """
            Forward pass only (no backward):
              - Compute output logits for the batch
              - Do not compute gradients (no memory overhead)
            batch_idx: index of batch
            data: [batch_size, channels, height, width]
            target: [batch_size] true labels
            """

            data, target = data.to(device), target.to(device)

            # 1️⃣ Forward pass only (no weight updates)
            output = model(data)
            # output shape: [batch_size, num_classes]
            # Each row = logits for that image

            # 2️⃣ Compute batch loss
            test_loss += criterion(output, target).item()

            # 3️⃣ Count correct predictions
            correct += GetCorrectPredCount(output, target)

    # Average test loss across all samples
    test_loss /= len(test_loader.dataset)

    # Test accuracy
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

"""CODE BLOCK: 10"""

device = 'cuda'
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

criterion = nn.CrossEntropyLoss()
num_epochs = 15

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train(model, device, train_loader, optimizer, criterion)
  test(model, device, test_loader, criterion)
  scheduler.step()

"""CODE BLOCK: 11"""

fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")