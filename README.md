# Fashion Classifier

Fashion Classifier is a predictor that classifies clothing items into 10 different classes. It has been trained on the Fashion MNIST dataset using a classification model. The project is implemented as a React app, with the trained model stored in the public folder.

## Overview

The Fashion Classifier utilizes deep learning techniques to classify clothing items based on their images. It follows the following steps to make predictions:

1. Image Processing: The input image is processed to convert it into grayscale and resize it to a standard size of 28 by 28 pixels.
2. Normalization: The processed image is normalized to ensure consistent input across the model.
3. Model Prediction: The normalized image is fed into the trained classification model, which predicts the class label of the clothing item.
4. Result Visualization: The predicted class label and associated probability are displayed to provide insights into the classification outcome.

## Training the Model

The model used in this project was trained using TensorFlow, a popular deep learning framework. The Fashion MNIST dataset, consisting of 60,000 training images and 10,000 test images, was used for training and evaluation. The model was trained to recognize the following 10 classes of clothing items:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Training a deep learning model involves multiple iterations of feeding the training data to the model, adjusting the model's parameters, and optimizing its performance. The exact architecture and training details may vary based on the implementation and experimentation.

## Getting Started

To run the Fashion Classifier locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/Fashion-Classifier.git`
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
4. Access the application in your browser at `http://localhost:3000`.

Please note that you will need to have Node.js and npm (Node Package Manager) installed on your machine to run the application.

## Acknowledgements

The Fashion MNIST dataset used for training the model was originally created by Zalando Research and is widely used for benchmarking machine learning algorithms. TensorFlow, an open-source deep learning framework, was instrumental in training and deploying the model.

## License

This project is licensed under the [MIT License](LICENSE).
