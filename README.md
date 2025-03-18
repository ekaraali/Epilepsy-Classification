# Epilepsy-Classification

This repository contains a modular implementation for the classification of EEG signals that includes 500 rows of patient samples with 5094 features, and there are 2 classes to represent classes: 
  - **Class 3** -> Mild Epilepsy Level
  - **Class 4** -> Severe Epilepsy Level

The EEG dataset is assumed to have a label column `y` (with values 3 and 4) and is preprocessed using a series of steps, including outlier elimination, normalization, and oversampling with SMOTE.

## Introduction

This project implements an end-to-end EEG classification pipeline using a Support Vector Machine (SVM) model. The pipeline comprises:

- **Data Preprocessing:**
  - **Stratified Train-Test Split:** Ensuring a balanced split for both classes.
  - **Outlier Elimination:** Performed on the raw training data for both classes based on z-score thresholds.
  - **Normalization:** Applied on the filtered training data and separately on the test data.
  - **SMOTE Oversampling:** Applied on the training set to address class imbalance.
 
- **Feature Selection:**
  - Utilizes `SelectKBest` with mutual information to select the top 50 features.
 
- **Model Training:**
  - An SVM classifier is trained with grid search using 5-fold cross-validation.
  - **Default SVM Hyperparameters:**
    - `C`: [0.1, 1, 10, 100]
    - `kernel`: ['linear', 'poly', 'rbf', 'sigmoid']
   
- **Evaluation:**
  - The classifier is evaluated using metrics such as accuracy, precision, recall, and F1 score.
  - Evaluation plots include the confusion matrix, precision-recall curve, and ROC curve.

Parameters and settings can be easily adjusted by modifying the corresponding modules.

---

## Repository Structure

- **preprocessing.py**  
  Contains the `EEGPreprocessor` class that:
  - Performs stratified train-test split.
  - Applies outlier elimination on raw training data followed by normalization.
  - Applies SMOTE on the normalized training set.
  - Normalizes the test set (without outlier elimination).

- **feature_selection.py**  
  Contains the `FeatureSelector` class that selects the top features using `SelectKBest`.

- **svm_trainer.py**  
  Contains the `SVMTrainer` class for training the SVM classifier using grid search and cross-validation.

- **evaluation.py**  
  Contains evaluation functions to compute metrics and generate plots:
  - Confusion Matrix
  - Precision-Recall Curve
  - ROC Curve

- **main.py**  
  The main script that ties all the modules together to execute the complete pipeline.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/eeg-svm-classifier.git
   cd eeg-svm-classifier

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate

3. **Install Dependencies:**

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

# Results

Results of the classifier are assessed by a few evaluation metrics: 
  - Accuracy: 87.5%
  - Precision: 90.0%
  - Recall: 86.0%
  - F1 Score: 88.0%

  The Precision-Recall Curve can be seen in below:

  <a href="url">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/train_loss_curve.png?raw=true">
  </a>

  The ROC Curve can be seen in below:

  <a href="url">
  <img src="https://github.com/ekaraali/Lemon_Classification_Model/blob/main/images/train_loss_curve.png?raw=true">
  </a>
