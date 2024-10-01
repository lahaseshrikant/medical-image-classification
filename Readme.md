
# Medical Image Classification Project

## Overview
This project is focused on developing a deep learning model for classifying medical images, specifically chest X-rays, to identify conditions like pneumonia. We built and trained two models: a baseline Convolutional Neural Network (CNN) and a second, more optimized CNN using advanced techniques such as data augmentation, early stopping, and model checkpointing.

The project is organized to enable easy reproducibility, with scripts to train, evaluate, and preprocess data. The goal is to provide a robust and scalable solution for medical image classification tasks.

---

## Project Structure

```plaintext
my_project/
├── Data/
│   ├── train_data/   # Training data directory
│   ├── test_data/    # Testing data directory
│   └── val_data/     # Validation data directory
├── src/
|   ├── advance/
|   │   ├── train.py          # Main script for training the model
|   │   ├── preprocess.py     # Script for data preprocessing and augmentation
|   │   ├── model.py          # Definition of the CNN architectures
|   |   └── evaluate.py       # Script for evaluating the trained model
|   └── baseline/
|       ├── train.py          # Main script for training the model
|       ├── preprocess.py     # Script for data preprocessing and augmentation
|       ├── model.py          # Definition of the CNN architectures
|       └── evaluate.py       # Script for evaluating the trained model
├── static/
|   └── styles.css
├── templates/
|       ├── index.html
|       └── result.html
├── app.py
└── requirements.txt

```

### Data Structure

The data should be organized in the `data/` directory as follows:

```
data/chest_xray
├── train_data/  # Subfolders for each class (e.g., Normal, Pneumonia)
├── test_data/   # Subfolders for each class
└── val_data/    # Subfolders for each class
```

---

## Installation

### Prerequisites
- **Python 3.x**
- **TensorFlow 2.x**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Pillow**

### Setup Steps

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   ```
   
2. **Activate the virtual environment**:
   - For Windows:
     ```bash
     venv\Scriptsctivate
     ```
   - For macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install the required libraries**:
   ```bash
   pip install tensorflow keras numpy matplotlib Pillow
   ```

---

## Data Preprocessing

Data preprocessing and augmentation are handled by the `preprocess.py` script. This script includes steps for:

- **Image rescaling**
- **Data augmentation**: Random rotations, flips, zooms to diversify the training set.
- **Image normalization**: Ensuring pixel values are scaled between 0 and 1.

The `create_generators` function in this script is responsible for generating batches of augmented data for training, validation, and testing.

---

## Models

### 1. **Baseline Model**

The initial model is a standard **Convolutional Neural Network (CNN)** with the following structure:

- **Conv2D layers**: Feature extraction with convolutional filters.
- **MaxPooling layers**: Reducing the spatial dimensions.
- **Dropout layers**: Preventing overfitting by randomly deactivating some neurons during training.
- **Dense layers**: Final classification.

The baseline model is used to establish a performance benchmark.

### 2. **Tuned Model**

For the second, more advanced model, we use a pre-trained model (VGG16) to take advantage of transfer learning. The pre-trained convolutional base is loaded without the top classification layers, and we add custom fully connected layers on top to adapt it for pneumonia detection.

- **Model Architecture**:
   - ***Pre-trained convolutional base***: Extracts features from the images (VGG16).
   - ***Custom layers***: Dense layers added on top of the pre-trained model.
      - A fully connected layer with ReLU activation
      - Dropout to prevent overfitting
      - Final layer using sigmoid activation for binary classification

we further optimized the model by:

- **Data Augmentation**: Applied techniques like random flips, rotations, and zooms to generate more diverse training data.
- **Early Stopping**: Monitors validation loss and stops training if no improvement is seen over multiple epochs.
- **Model Checkpointing**: Saves the model with the best performance (lowest validation loss) during training.

The tuned model demonstrated significant improvement in accuracy over the baseline model.

---

## Usage

### 1. **Training the Models**

To train the models, run the following command:

```bash
python src/advance/train.py
```

The script will read data from the `data/` directory, preprocess it, and train the model. Model checkpoints will be saved to the `saved_models/` directory.

### 2. **Evaluating the Models**

Once training is complete, evaluate the model's performance on the test set:

```bash
python src/advance/evaluate.py
```

This will output the model's accuracy, loss, and other relevant metrics on the test dataset.

---

## Callbacks

- **Early Stopping**: Stops the training process once the validation loss stops improving.
- **Model Checkpointing**: Saves the best model based on validation performance.

### Callbacks in Code

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
```

---

## Results

### Performance Metrics

Both models were evaluated on accuracy, loss, and confusion matrix metrics. The results showed that the **tuned model** significantly outperforms the baseline model.

### Results (for tuned model)

- **Training Accuracy**: 93.67 %
- **Validation Accuracy**: 87.50%
- **Test Accuracy**: 90.54%

### Plots

You can visualize the training and validation performance using the plots generated during training:

![Training and Validation Accuracy](path/to/accuracy_plot.png)
![Training and Validation Loss](path/to/loss_plot.png)

---
## Web Application
This project includes a Flask web application where users can upload chest X-ray images and receive a prediction of whether the patient has pneumonia.

**Steps to Run the Web Application**
1. Run the Flask app:
```python
python app.py
```
2. Access the web app: Open your browser and go to http://127.0.0.1:5000/. You will see a web interface where you can upload an X-ray image, and the model will classify the image as either "Pneumonia" or "Normal".


---

## Acknowledgments

- The dataset was sourced from publicly available repositories.
- Special thanks to the open-source community for tools like [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/).

---

## Contact

For any inquiries or collaboration opportunities, feel free to reach out:
- Email: [shrikantlahse143@gmail.com]
- GitHUB: [https://github.com/lahaseshrikant]
