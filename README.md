# Common Convolutional Neural Network (CNN) Architectures

## 1. [LeNet (1998)](LeNet/LeNet.ipynb)

- __Developed by__: [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun)
- __Key Features__:
	- One of the earliest CNNs, designed for digit recognition (e.g., MNIST dataset).
	- Simple architecture with convolutional layers, pooling layers, and fully connected layers.
- __Use Case__: Handwritten digit classification.
## 2. AlexNet (2012)

- __Developed by__: [Alex Krizhevsky](https://en.wikipedia.org/wiki/Alex_Krizhevsky), [Ilya Sutskever](https://en.wikipedia.org/wiki/Ilya_Sutskever), and [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton)
- __Key Features__:
	- First CNN to achieve breakthrough performance in the ImageNet competition.
	- Introduced ReLU activation and dropout for regularization.
	- 5 convolutional layers, max-pooling, and 3 fully connected layers.
- Use Case: General image classification.
<!--

## 3. VGGNet (2014)

	•	Developed by: Visual Geometry Group (VGG), University of Oxford
	•	Key Features:
	•	Simple architecture using 3x3 convolutional filters and 2x2 max-pooling.
	•	Variants like VGG-16 and VGG-19 with 16 and 19 layers, respectively.
	•	High computational cost due to a large number of parameters.
	•	Use Case: Image classification and feature extraction.

## 4. GoogLeNet/Inception (2014)

	•	Developed by: Google
	•	Key Features:
	•	Introduced the “Inception module,” which uses filters of multiple sizes in parallel.
	•	Deep network (22 layers) but computationally efficient due to fewer parameters.
	•	Includes batch normalization, dropout, and RMSProp optimization.
	•	Use Case: Image classification (ImageNet winner).

## 5. ResNet (2015)

	•	Developed by: Microsoft Research
	•	Key Features:
	•	Introduced residual connections (skip connections) to address vanishing gradients.
	•	Variants like ResNet-50, ResNet-101, and ResNet-152 with different depths.
	•	Deep architectures with better training efficiency.
	•	Use Case: Image classification and object detection.

## 6. DenseNet (2017)

	•	Developed by: Gao Huang et al.
	•	Key Features:
	•	Uses dense connections where each layer is connected to every other layer.
	•	Encourages feature reuse and reduces the number of parameters.
	•	Variants include DenseNet-121, DenseNet-169, etc.
	•	Use Case: Classification, segmentation, and feature extraction.

## 7. MobileNet (2017)

	•	Developed by: Google
	•	Key Features:
	•	Designed for mobile and embedded devices with limited computational resources.
	•	Uses depthwise separable convolutions to reduce complexity.
	•	Variants like MobileNetV1, V2, and V3.
	•	Use Case: Mobile vision tasks like face recognition and object detection.

## 8. EfficientNet (2019)

	•	Developed by: Google
	•	Key Features:
	•	Scalable architecture that balances depth, width, and resolution systematically.
	•	Highly efficient, achieving state-of-the-art performance with fewer parameters.
	•	Variants like EfficientNet-B0 to B7.
	•	Use Case: Image classification and transfer learning.

## 9. YOLO (You Only Look Once)

	•	Developed by: Joseph Redmon et al.
	•	Key Features:
	•	Designed for real-time object detection.
	•	Predicts bounding boxes and class probabilities directly from images.
	•	Variants like YOLOv3, YOLOv4, and YOLOv5.
	•	Use Case: Object detection in real-time applications.
-->