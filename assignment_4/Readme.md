# How I Improved the Network and Got \~97% Accuracy

### Original Network (Given)

* 4 convolutional layers (`conv1 → conv2 → conv3 → conv4`) with large channel sizes (32, 64, 128, 256).
* 2 fully connected layers (`fc1=320→50`, `fc2=50→10`).
* Used **ReLU + MaxPool**, but structure was heavy and fully connected part was restrictive.
* Optimizer: **SGD** (slower convergence).
* Batch order was fixed (no shuffle).

Accuracy: lower (not optimized well).

---

### My Modified Network

1. **Simplified CNN Structure**

   * Instead of very large channel jumps (32→256), we started small (1→8→16→32).
   * This reduced **overfitting** and made training more stable.
   * Still deep enough to learn features, but more efficient.

2. **Added Pooling at the Right Places**

   * Used `MaxPool2d(2,2)` after `conv2` and `conv4`.
   * Gradually reduced spatial dimensions (28×28 → 14×14 → 7×7).
   * Keeps computation low and improves translation invariance.

3. **Calculated Parameters Properly**

   * Conv parameters = `in_channels × out_channels × kernel_size × kernel_size`.
   * FC layer = `input_features × output_features`.
   * Our design kept total params \~23k (vs. original much larger).
   * Smaller network → faster training, less overfitting.

4. **Single Fully Connected Layer**

   * Flattened 32×7×7 = 1568 features → directly mapped to 10 classes.
   * Removed the unnecessary bottleneck (`320→50→10`).
   * Improved gradient flow and reduced wasted parameters.

5. **Better Optimizer (Adam instead of SGD)**

   * Adam adapts learning rate per parameter.
   * Faster convergence, less manual tuning.
   * Helped reach higher accuracy quickly.

6. **Shuffled Training Data**

   * Ensures batches are **different each epoch**.
   * Prevents the model from memorizing order of digits.
   * Improves generalization.

---

### Why Accuracy Improved to \~97%

* **Smaller but deeper CNN** → better feature extraction with fewer parameters.
* **Pooling at right stages** → robustness to shifts and scale.
* **Flattening to 10 directly** → efficient classification.
* **Adam optimizer + shuffling** → faster, better convergence.

Net result: A **leaner, better-optimized CNN** that generalizes well and reached \~97% test accuracy.

---

