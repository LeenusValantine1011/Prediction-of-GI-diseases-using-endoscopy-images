# Prediction-of-GI-diseases-using-endoscopy-images

Overview

This project leverages deep learning models—ResNet50, MobileNetV2, and VGG16—for the classification of gastrointestinal (GI) diseases using endoscopy images. The models utilize convolutional neural networks (CNNs) to extract features and classify diseases efficiently. Various preprocessing techniques, including contrast adjustment, denoising, and image augmentation, are applied to enhance image quality and model performance.

Dataset

The dataset consists of endoscopy images used for training and evaluating the deep learning models. The dataset is partitioned into training, validation, and testing subsets to ensure robust evaluation.

You can access the dataset here: Google Drive Link

Models Used

ResNet50: A deep residual network with 50 layers, effective for capturing intricate patterns in medical images.

MobileNetV2: An efficient architecture optimized for mobile and embedded applications, utilizing depthwise separable convolutions.

VGG16: A classic CNN with 16 layers, known for its simplicity and effectiveness in image classification.

Preprocessing Techniques

Contrast adjustment and brightness normalization

Noise reduction using Gaussian blur and median filtering

Image resizing and cropping for standardization

Data augmentation (rotation, flipping, scaling)

Region of interest (ROI) extraction

Color normalization for consistency

Model Evaluation

The models are assessed using key performance metrics:

Accuracy: Measures overall prediction correctness.

Sensitivity: Evaluates the ability to detect diseased cases (true positive rate).

Specificity: Assesses the ability to exclude non-diseased cases (true negative rate).

Training and Testing

The dataset is divided into training, validation, and testing sets. Models are trained using techniques like stochastic gradient descent (SGD) and evaluated on unseen test data. Performance comparisons are made using statistical analysis, confusion matrices, and ROC curves.

Conclusion

This project demonstrates the application of deep learning in medical image classification, providing a framework for GI disease prediction using CNN-based architectures. Future enhancements may include additional preprocessing techniques and hyperparameter tuning to further improve model performance.

