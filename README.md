# captcha-recognizer
# Introduction
This project is a captcha recognizer that uses the LACC (Label Combination Classifier) deep learning model to perform image-to-text translation tasks. The LACC model is implemented using PyTorch and the EfficientNetV2-S backbone architecture for feature extraction from the input images. The converter combines the extracted features into a smaller representation that is suitable for text prediction. The converted features are then passed through three fully connected layers to predict the final text output. The output of the LACC model is a tensor that represents a probability distribution over the set of possible characters at each position in the predicted text.

# Datasets
The project uses a combined dataset of six different captcha datasets, which include various font styles and character sets. The dataset is divided into training, validation, and testing sets, with a ratio of 80:10:10, respectively.

# Architecture

![Untitled Diagram drawio-10](https://user-images.githubusercontent.com/64341057/230733270-89fb0e95-8517-4a2e-b927-6b1854ac11be.png)



The LACC (Label Combination Classifier) architecture is a PyTorch neural network used for image-to-text translation tasks. The architecture consists of an encoder, a converter, and several fully-connected layers.

The encoder is implemented using the EfficientNetV2-S model from the torchvision library. The EfficientNetV2-S is a lightweight and efficient model that is suitable for mobile and embedded devices. The encoder extracts feature maps from the input image tensor, which are then used for character recognition.

The converter is a trainable parameter matrix that maps the encoded feature maps to character embeddings. The converter combines the feature information into a smaller representation suitable for text prediction. In this implementation, the converter is defined as a nn.parameter.Parameter object initialized to a tensor of size (64, CHAR_NUM), where CHAR_NUM is the number of possible characters in the captcha.

The SiLU (Sigmoid-Weighted Linear Unit) activation function is used between the fully-connected layers to introduce non-linearity to the model. SiLU is a popular activation function that has been shown to improve the performance of deep neural networks.

The fully-connected layers consist of three linear layers with 512, 64, and MAX_LEN neurons, respectively. The MAX_LEN is the maximum length of the predicted text. The first layer takes the character embeddings as input, followed by a SiLU activation function. The second layer maps the output of the first layer to a 64-dimensional space. The third layer maps the output of the second layer to a tensor of shape (batch_size, MAX_LEN), where each element in the tensor represents a probability distribution over the set of possible characters at the corresponding position in the predicted text.

The forward method of the LACC class performs a forward pass of an input image tensor through the encoder, followed by the converter and the fully-connected layers, to obtain the predicted character embeddings. The output is a tensor of shape (batch_size, max_length), where max_length is the maximum length of the predicted text. The LACC architecture is suitable for captcha recognition tasks and has shown to be very accurate in recognizing captchas from the combined dataset.

# Results
The LACC model has shown to be very accurate in recognizing captchas from the combined dataset. The accuracy on the test set is 99.8%, which is a significant improvement compared to previous captcha recognition methods.

<img src="https://user-images.githubusercontent.com/64341057/230731893-d013d033-06a0-4a20-a88b-32996e03ca0a.png" height="200" width="300">

<img src="https://user-images.githubusercontent.com/64341057/230731942-29bba56b-11ce-44d4-b450-747773a608cc.png" height="200" width="300">

<img src="https://user-images.githubusercontent.com/64341057/230731944-2adfdb06-88fe-4dbf-95cd-91d80209adf5.png" height="200" width="300">


# Future Plans
In the future, the project can be improved by adding more captcha datasets to make the model more robust and generalize better to various captcha styles. Additionally, data augmentation techniques can be applied to the existing dataset to increase the size of the training set and improve the model's performance. Finally, the project can be integrated into a web application to provide captcha recognition as a service.

