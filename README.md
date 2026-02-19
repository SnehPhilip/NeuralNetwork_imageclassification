# NeuralNetwork_imageclassification
Deep Learning models to classify images - ANN,CNN,transferLearning

1. Handwritten Digit Recognition (MNIST)

This project is the "Hello World" of deep learning. I built a Convolutional Neural Network (CNN) to classify grayscale images of handwritten digits (0–9).

    The Tech: Utilized the MNIST dataset, implementing layers for feature extraction (Convolutions) and spatial reduction (Pooling).

    The Result: Achieved high accuracy by optimizing cross-entropy loss, demonstrating a fundamental understanding of how neural networks "see" and interpret simple patterns.

2. Fashion Items Classification (Fashion-MNIST)

Moving beyond simple digits, this project focused on distinguishing between 10 categories of clothing and accessories (e.g., sneakers, coats, shirts).

    The Challenge: Unlike digits, fashion items have more complex textures and shapes, requiring a more robust network architecture to prevent overfitting.

    The Result: Successfully trained a model to generalize across diverse clothing styles, proving the scalability of CNNs to more nuanced image data.

3. Advanced Image Classification via Transfer Learning

For this project, I tackled complex, high-resolution data by leveraging Transfer Learning. Instead of training a model from scratch, I utilized pre-trained architectures like ResNet50 or VGG16, which were previously trained on millions of images (ImageNet).

    The Tech: I performed "fine-tuning" by freezing the initial layers of the pre-trained model and training a custom classifier on top to suit a specific, complex dataset.

    The Result: This approach significantly reduced training time while achieving state-of-the-art accuracy on sophisticated data that would otherwise require massive computational resources.
