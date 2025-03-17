# Leaffliction

## Overview

Leaffliction is a computer vision project focused on image classification for disease recognition in leaves. It involves analyzing a dataset of leaf images, augmenting the dataset to improve balance, applying image transformations, and developing a classification model to identify leaf diseases.

## Features

- **Data Analysis**: Extracts and visualizes information about the dataset using pie and bar charts.
- **Data Augmentation**: Enhances dataset balance by applying transformations like flipping, rotating, skewing, and cropping.
- **Image Transformation**: Processes images with techniques such as Gaussian blur, masking, and object analysis.
- **Classification Model**: Trains a model to recognize leaf diseases and predicts diseases from new images.

## Installation

1. Ensure you have Python installed.
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Clone the repository and navigate to the project folder.
   ```sh
   git clone <repo_url>
   cd leaffliction
   ```

## Usage

### Data Analysis

Run the dataset analysis script to visualize plant and disease distribution:

```sh
python Distribution.py ./path_to_dataset
```

### Data Augmentation

Generate augmented images for dataset balancing:

```sh
python Augmentation.py ./path_to_image
```

### Image Transformation

Apply transformations to images for feature extraction:

```sh
python Transformation.py -src ./source_dir -dst ./destination_dir
```

### Model Training

Train a disease classification model:

```sh
python train.py ./path_to_dataset
```

### Prediction

Predict the disease of a leaf image:

```sh
python predict.py ./path_to_image
```

## Evaluation

- The dataset should be divided into training and validation sets.
- Validation accuracy should be at least 90%.
- A `.zip` file containing the trained model and dataset signature is required.
- Use `sha1sum` to verify the dataset signature:
  ```sh
  sha1sum dataset.zip
  ```
