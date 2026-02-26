# NeuralNetwork_imageclassification
Deep Learning models to classify images - ANN,CNN,Deep CNN(from scratch ),Transfer Learning

| Architecture | Best Used For... | Example Task |Dataset|
| :--- | :--- | :--- | :--- |
| **ANN**(Flattened) | Tabular/simple Images Flattened data | Credit scoring, basic classification,grayscale images|DigitRecognizer
| **CNN (Convolutional)** | Spatial data / Images | color channel images |CIFAR-10
| **Deep CNN** | Spatial data / Images | From scratch lightweight models on sensors,Pathogen Detection etc |CIFAR-100
| **Pretrained CNN** | Advanced Spatial data / Images classification | Facial recognition,E-commerce: Visual Search & Recommendations,Identify land use patterns (Forest, Water, Urban, Agriculture) from satellite tiles | DeepFashion Dataset,googleEarth

# Progress Trail

1. Handwritten Digit Recognition (MNIST)

        A Convolutional Neural Network (CNN) to classify grayscale images of handwritten digits (0–9).

        The Tech: Utilized the MNIST dataset, implementing layers for feature extraction (Convolutions) and spatial reduction (Pooling).

        The Result: Achieved high accuracy by optimizing cross-entropy loss, demonstrating a fundamental understanding of how neural networks "see" and interpret simple patterns.

2. Fashion Items Classification (Fashion-MNIST)

        Moving beyond simple digits, this project focused on distinguishing between 10 categories of clothing and accessories (e.g., sneakers, coats, shirts).

        The Challenge: Unlike digits, fashion items have more complex textures and shapes, requiring a more robust network architecture to prevent overfitting.

        The Result: Successfully trained a model to generalize across diverse clothing styles, proving the scalability of CNNs to more nuanced image data.

3. Advanced Image Classification via Transfer Learning

        Using ResNet50 and VGG16 with pre-trained weights ,complex and high-resolution data was tackled by leveraging Transfer Learning. Instead of training a model from scratch,  pre-trained architectures  previously trained on millions of images (ImageNet) enhances training.
   
        The Tech: Performed "fine-tuning" by freezing the initial layers of the pre-trained model and training a custom classifier on top to suit a specific, complex dataset.

        The Result: This approach significantly reduced training time while achieving state-of-the-art accuracy on sophisticated data that would otherwise require massive computational resources.
