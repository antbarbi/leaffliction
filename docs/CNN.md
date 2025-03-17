**Convolutional Neural Network (CNN)**
A type of deep learning model specifically designed for processing structured grid-like data, such as images. CNNs are highly effective for tasks like image classification, object detection, and facial recognition. They work by automatically learning spatial hierarchies of features from input data through a combination of convolutional layers, pooling layers, and fully connected layers.

---

## **How a CNN Works in Detail**
CNNs process input images through a series of layers that extract important features while reducing computational complexity. The key components of a CNN include:

### **1. Input Layer**
- The input to a CNN is typically an image.
  - An RGB image has three channels (Red, Green, Blue).
  - A grayscale image has only one channel.

---

### **2. Convolutional Layer**
The **convolutional layer** is the core building block of a CNN. It applies **filters (kernels)** to the input image to extract features.

#### **How Convolution Works:**
1. A small **filter (kernel)** (e.g., 3×3 or 5×5 matrix) slides over the input image.
2. The filter performs an **element-wise multiplication** with the part of the image it overlaps.
3. The results are summed to obtain a **single pixel** in the output feature map.
4. The filter moves (or "slides") by a certain step size called **stride**.
5. Multiple filters are used to detect different patterns (edges, textures, colors, etc.).
6. The output is known as a **feature map** (activation map).

##### **Example Calculation:**
If we have a **5×5 image** and apply a **3×3 filter** with a **stride of 1**, the output size is:
\[
\frac{(5 - 3)}{1} + 1 = 3
\]
Thus, the output feature map is **3×3**.

#### **Activation Function (ReLU)**
After convolution, we apply an activation function, typically **ReLU (Rectified Linear Unit)**, which introduces non-linearity:
\[
ReLU(x) = max(0, x)
\]
This helps CNNs learn complex patterns.

---

### **3. Pooling Layer**
Pooling reduces the spatial size of the feature maps, making the network computationally efficient and robust to minor distortions.

#### **Types of Pooling**
- **Max Pooling**: Takes the maximum value from each region.
- **Average Pooling**: Takes the average value from each region.

##### **Example:**
A **2×2 Max Pooling** operation with a **stride of 2** on a **4×4 feature map** reduces it to **2×2**.

---

### **4. Fully Connected Layer (FC Layer)**
After multiple convolutional and pooling layers, the feature maps are **flattened** into a 1D vector and fed into a fully connected layer.

- This layer acts as a **classifier** by assigning probabilities to different classes.
- The final layer uses **Softmax Activation** (for multi-class classification) or **Sigmoid Activation** (for binary classification).

---

### **5. Output Layer**
- Produces the final class probabilities.
- The neuron with the highest probability determines the predicted class.

---

## **How CNN Learns: Backpropagation & Optimization**
CNNs **learn through backpropagation**, using optimization techniques like **Stochastic Gradient Descent (SGD) and Adam**.

1. **Forward Pass**:
   - Input is passed through the layers to get a prediction.
2. **Loss Calculation**:
   - The difference between the predicted and actual values is measured using a **loss function** (e.g., Cross-Entropy Loss).
3. **Backward Pass (Backpropagation)**:
   - The gradients of the loss with respect to the weights are calculated.
   - Using **Gradient Descent**, the weights of filters are updated to minimize the loss.

---

## **Example CNN Architecture**
A simple CNN architecture for handwritten digit recognition:

1. **Input**: 32×32 grayscale image.
2. **Conv Layer 1**: 6 filters of size 5×5 → Output: 28×28×6.
3. **Pooling Layer 1**: 2×2 max pooling → Output: 14×14×6.
4. **Conv Layer 2**: 16 filters of size 5×5 → Output: 10×10×16.
5. **Pooling Layer 2**: 2×2 max pooling → Output: 5×5×16.
6. **Flatten**: Converts feature maps to a 1D vector.
7. **Fully Connected Layer**: 120 neurons.
8. **Output Layer**: 10 neurons (for digit classes 0-9).
