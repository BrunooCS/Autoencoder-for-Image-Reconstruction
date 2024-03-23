# Autoencoder for Image Reconstruction

This project aims to implement an autoencoder using TensorFlow and Keras for image reconstruction. An autoencoder is a type of neural network that learns to encode and then decode input data, with the objective of reconstructing the original input as accurately as possible. The encoder part of the network learns a compact representation of the input data, while the decoder part reconstructs the input from this representation.

## Dataset
The project utilizes two datasets for experimentation:
- **MNIST**: A dataset of hand-written digits consisting of 28x28 grayscale images.
- **CIFAR-10**: A dataset of 32x32 color images across 10 classes, including various objects and animals.

## Architecture
The architecture of the autoencoder consists of two main components: the encoder and the decoder.

### Encoder
The encoder part of the autoencoder is responsible for mapping the input data to a lower-dimensional latent space. It typically consists of convolutional layers followed by max-pooling operations to reduce the spatial dimensions of the input data. The final layer of the encoder outputs the mean ($\mu$) and standard deviation ($\sigma$) of the latent space distribution.

### Decoder
The decoder part of the autoencoder reconstructs the input data from the latent space representation obtained from the encoder. It consists of convolutional layers followed by upsampling operations to gradually increase the spatial dimensions of the data until it matches the original input size.

## Training
The autoencoder is trained using a custom training step that incorporates both the reconstruction loss and the KL-divergence loss. The reconstruction loss measures the difference between the original input and the reconstructed output, while the KL-divergence loss regularizes the latent space distribution to follow a unit Gaussian distribution.

## Usage
To use the autoencoder model:
1. Define the architecture of the encoder and decoder.
2. Instantiate the AutoEncoder class, passing the encoder and decoder models as arguments.
3. Compile the model with an appropriate optimizer and loss function.
4. Train the model using the `train_step` method.

## Experimentation
The project includes experimentation with both the MNIST and CIFAR-10 datasets. After training the autoencoder, we visualize the reconstruction of sample images from the dataset to evaluate the performance of the model.

## Requirements
- TensorFlow
- Keras
- Matplotlib

## License
This project is licensed under the MIT License. See the LICENSE file for details.