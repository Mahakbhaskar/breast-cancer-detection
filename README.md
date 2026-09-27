# Breast Cancer Detection using a Neural Network

## Overview
This project demonstrates a machine learning-based classification workflow
for educational purposes. A feedforward neural network is trained to
classify a breast tumor as **Benign** or **Malignant** based on diagnostic
measurements, and the trained model is served through an interactive
Streamlit web application.

## Objective
Given a set of measurements computed from a digitized image of a fine
needle aspirate (FNA) of a breast mass (cell nucleus size, shape, and
texture statistics), predict whether the mass is benign or malignant. This
is framed as a binary classification problem.

## Dataset
- File: `data.csv`
- Breast Cancer Wisconsin (Diagnostic) dataset
- 569 samples
- Target column: `diagnosis` (`M` = Malignant, `B` = Benign), encoded as 1/0
- Class distribution: 357 Benign, 212 Malignant

## Features
30 real diagnostic features are used for training and prediction: 10
measurements (radius, texture, perimeter, area, smoothness, compactness,
concavity, concave points, symmetry, fractal dimension), each reported as a
`mean`, `standard error (se)`, and `worst` (largest) value across the cell
nuclei in the image. The `id` column (a row identifier) and an empty
placeholder column present in the raw CSV are not used as model features.

## Technology Stack
- Python
- TensorFlow / Keras — model definition and training
- scikit-learn — preprocessing, train/test split, evaluation metrics
- pandas / NumPy — data handling
- Streamlit — web application interface
- Matplotlib — training curve visualization

## Data Preprocessing
1. Removed non-predictive columns (`id`) and an empty placeholder column
   from the raw dataset, retaining exactly 30 diagnostic feature columns.
2. Verified there are no missing values in the feature columns.
3. Encoded the `diagnosis` target with `LabelEncoder` (B → 0, M → 1).
4. Split the data into training and test sets (80/20) using a stratified
   split to preserve the Benign/Malignant class ratio in both sets.
5. Scaled all features with `StandardScaler`, fit only on the training set
   and then applied to both the training and test sets. The fitted scaler
   is saved as `scaler.pkl` so the Streamlit app uses identical
   preprocessing at inference time.

## Model Architecture
A compact, fully-connected feedforward network:

```
Input (30 features)
    -> Dense(32, activation='relu')
    -> Dense(16, activation='relu')
    -> Dense(1,  activation='sigmoid')
```

- **ReLU** is used in the hidden layers for efficient computation and to
  let the network learn non-linear decision boundaries.
- **Sigmoid** is used on the output layer to produce a probability in the
  range `[0, 1]`, interpreted as the estimated likelihood that the tumor is
  malignant.

## Training Approach
- Loss function: `binary_crossentropy`
- Optimizer: `Adam`
- Epochs: 50
- An internal validation split (20% of the training data) was used during
  training to monitor generalization, separate from the held-out test set.

## Evaluation Results
Evaluated on the held-out test set (114 samples, not used during training
or scaling):

| Metric | Value |
|---|---|
| Test Accuracy | **97.37%** |
| Precision (Malignant) | 0.95 |
| Recall (Malignant) | 0.98 |
| F1-score (Malignant) | 0.96 |
| Precision (Benign) | 0.99 |
| Recall (Benign) | 0.97 |
| F1-score (Benign) | 0.98 |

### Confusion Matrix
(rows = true label, columns = predicted label, order = [Benign, Malignant])

```
[[70  2]
 [ 1 41]]
```

### Classification Report
```
              precision    recall  f1-score   support

      Benign       0.99      0.97      0.98        72
   Malignant       0.95      0.98      0.96        42

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

Training and validation accuracy/loss curves across the 50 training epochs
are included as `accuracy_curve.png` and `loss_curve.png`.

## Streamlit Application
`app.py` loads the trained model (`breast_cancer_model.h5`) and the fitted
scaler (`scaler.pkl`), and provides sliders for all 30 diagnostic features,
grouped into Mean, Standard Error, and Worst sections with ranges based on
the dataset's real value ranges. When the user clicks **Run Diagnosis**,
the app:

1. Assembles the 30 input values in the exact order used during training.
2. Applies the fitted `StandardScaler` to the input.
3. Passes the scaled input to the trained model.
4. Reads the sigmoid output as the malignant probability and classifies the
   result as Malignant if the probability exceeds 0.5, otherwise Benign.

The app displays a clear disclaimer that it is an educational project and
not a medical diagnostic tool.

## Project Structure
```
Breast-Cancer-Detection/
├── app.py
├── Breast_Cancer_Detection.ipynb
├── breast_cancer_model.h5
├── scaler.pkl
├── data.csv
├── requirements.txt
├── README.md
├── .gitignore
├── accuracy_curve.png
└── loss_curve.png
```

## Installation

```bash
git clone <your-repo-url>
cd Breast-Cancer-Detection

python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## How to Run the App

```bash
streamlit run app.py
```

To retrain the model, open `Breast_Cancer_Detection.ipynb` and run all
cells; this regenerates `breast_cancer_model.h5` and `scaler.pkl`.

## Limitations / Future Improvements
- No k-fold cross-validation or hyperparameter search was performed.
- No dropout or additional regularization is used in the network.
- Evaluation is limited to a single train/test split rather than an
  independent external validation set.
- Future work could explore cross-validation, regularization, model
  comparison against classical algorithms (e.g. logistic regression, SVM),
  and calibration of the output probability.

## Disclaimer
This project demonstrates a machine learning-based classification workflow
for educational purposes. It is **not** a validated medical device or
diagnostic tool, has not undergone clinical evaluation, and must not be
used to make real medical decisions.
