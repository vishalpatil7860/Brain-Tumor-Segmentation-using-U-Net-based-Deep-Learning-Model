
# Brain Tumor Segmentation using UNet

The project aims to develop a deep learning model for brain tumor segmentation using MRI scans.
It is crucial to solve this problem as accurate segmentation aids in diagnosis and treatment planning
for brain tumor patients, potentially improving patient outcomes and quality of life.

## Authors

- [@vishalpatil7860](https://www.github.com/vishalpatil7860)


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)




## Dataset

Dataset Description
The public dataset comprises MRI scans of brain tumor patients, consisting of images and corresponding
masks for tumor segmentation. It is publicly available and was obtained from Kaggle. The
dataset’s origin is attributed to various medical research institutions and organizations involved in
collecting and curating medical imaging data. This dataset is the Figshare Brain Tumor dataset.

Dataset available on Kaggle:
https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation/data

Link to the original dataset:
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
3 Model


## Model Description

The chosen architecture for this project is UNet, a convolutional neural network (CNN) specifically designed for biomedical image segmentation tasks. UNet was selected for its proven effectiveness in segmenting medical images, particularly in scenarios with limited data. It consists of an encoder-decoder architecture with skip connections to preserve spatial information.

The model comprises:

- Convolutional layers
- Pooling layers
- Batch normalization layers
- Activation layers

The total number of parameters in the model is approximately 31 million, indicating moderate complexity.

The UNet architecture consists of an encoder path and a decoder path. The encoder path captures the context of the image, while the decoder path enables precise localization.

- The `conv_block` function defines a convolutional block, which consists of two convolutional layers with batch normalization and ReLU activation.
- The `encoder_block` function applies the convolutional block followed by max pooling to downsample the feature maps.
- The `decoder_block` function upsamples the feature maps using transposed convolution, concatenates them with the corresponding skip connections from the encoder path, and applies the convolutional block.
- The `build_unet` function constructs the complete UNet model by organizing the encoder and decoder blocks.

The UNet architecture follows a symmetric structure, with the number of filters doubling at each encoder block and halving at each decoder block. Skip connections are used to concatenate the feature maps from the encoder path with the corresponding feature maps in the decoder path, allowing the model to retain spatial information and improve localization accuracy.

The code defines an input shape of **(256, 256, 3)**, indicating that the input images have a spatial resolution of 256x256 pixels with 3 color channels (RGB).


## Model Training and Validation Description and Experiments

The training and validation process for the deep learning model involved the following:
- **Activation Function**: 
  - The code uses the `ReLU (Rectified Linear Unit)` activation function in the convolutional blocks, which introduces non-linearity and helps the model learn complex patterns.
  - The output layer of the UNet model uses the `sigmoid` activation function, which squashes the output values between 0 and 1, making it suitable for binary segmentation tasks where each pixel is classified as either belonging to the foreground (tumor) or background.
  
- **Optimizer**: Adam optimizer was used with the following settings:
  - Learning rate: 1e-4
  - Batch size: 16
  These choices were made based on experimentation and considering the model's complexity and available computational resources.
  
- **Dataset Division**: The dataset was split into:
  - Training set
  - Validation set
  - Test set
  Appropriate ratios and shuffling were applied to ensure unbiased evaluation and generalization.

- **Loss Function**: Dice loss function was employed due to its suitability for segmentation tasks, as it directly optimizes the overlap between predicted and ground truth masks.

- **Evaluation Metrics**: The following metrics were used to comprehensively assess the model's performance:
  - F1 score
  - Jaccard score
  - Recall
  - Precision
  These metrics provide insights into pixel-level accuracy and segmentation quality.

Experimental results demonstrated promising segmentation accuracy, with an F1 score of **0.72958** achieved on the test set, indicating the model's effectiveness in brain tumor segmentation.
