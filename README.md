## MIMIC-2024: Misogyny Identification in Memes Using Image Classification

This repository contains the code and data for a project focusing on identifying misogyny in memes using image classification techniques.

**Project Overview**

This project aims to address the pervasive issue of online misogyny, specifically within the realm of memes. While memes are often humorous and entertaining, they can also perpetuate harmful stereotypes and perpetuate misogynistic views. 

The goal of this project is to develop an image classification model capable of automatically identifying misogynistic content in memes. We utilize a Resnet152 model pre-trained on ImageNet and fine-tuned on a dataset of memes labeled for misogynistic content.

**Dataset**

The dataset used in this project, named MIMIC-2024, consists of over +5000 Indian memes collected from various social media platforms. Each meme is labeled for the presence of four types of misogynistic content:

* **Misogyny:** 
* **Objectification:** 
* **Prejudice:**
* **Humiliation:** 

The dataset is split into train, validation, and test sets to facilitate model training and evaluation. The csv file `MIMIC2024.csv` contains the filename, extracted text, and labels for each meme.

**Code Structure**

The repository is structured as follows:

* `meme_img_cls_resnet152.py`: This file contains the Python script for training and evaluating the image classification model. It includes the following steps:
    * Loading the dataset and applying data augmentation.
    * Defining a custom dataset class and data loaders.
    * Loading a pre-trained Resnet152 model and modifying its final layer for multi-label classification.
    * Defining the loss function, optimizer, and training loop.
    * Training the model and saving the trained weights.
    * Evaluating the model on the test set and generating predictions.
* `split_images.py`: This file contains the Python script for splitting the images into train, validation, and test sets. It takes the original dataset directory as input and creates three output directories for each split.

**Dependencies**

The following Python libraries are required to run the code:

* `torch`
* `torchvision`
* `pandas`
* `PIL`
* `tqdm`
* `numpy`

**How to Use**

1. **Prepare the Dataset:**
    * Download the dataset from the provided source.
    * Run the `split_images.py` script to split the images into train, validation, and test sets.
2. **Train and Evaluate the Model:**
    * Run the `meme_img_cls_resnet152.py` script to train and evaluate the image classification model.
3. **Predict on New Images:**
    * Load the trained model weights.
    * Pre-process the new images using the same transformations used for training.
    * Pass the pre-processed images through the model to obtain predictions.

**Future Work**

* Expanding the dataset with more diverse and nuanced examples of misogyny.
* Exploring different model architectures and training techniques.
* Developing a user interface for interacting with the model and predicting on new memes.

**Contributing**

Contributions to this project are welcome. Please feel free to open issues or submit pull requests for any improvements or bug fixes.
