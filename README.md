Breast Cancer Diagnosis Machine Learning Model
==============================================

This project involved developing a machine learning model to predict whether cancer is malignant or benign based on physical characteristics. The dataset used in this project contain various features that describe the characteristics of cell nuclei present in the digitized images of a fine needle aspirate (FNA) of a breast mass.

Introduction
------------
The primary objective of this project is to classify cancer as either malignant or benign based on various features obtained from digitized images of FNA of breast masses. The data comes from the Breast Cancer Wisconsin dataset. This dataset was obtained from UCI Machine Learning on Kaggle and containes 569 instances of different breast cancer cases.

Data Preprocessing
------------------
The data preprocessing steps for this project include:
  1. Loading the dataset.
  2. Dropping the irrelevant columns.
  3. Encoding the diagnosis column to binary values (1 for malignant, 0 for benign).
  4. Checking for missing values.

Exploratory Data Analysis
-------------------------
To understand the data distribution and characteristics, histograms and boxplots are created for all features. Additionally, a correlation heatmap is generated to visualize the relationships between features.

### Histograms
Histograms are used to display the distribution of each feature in the dataset.

### Boxplots
Boxplots help in indentifying the presence of outliers in the dataset.

### Correlation Heatmap
A correlation heatmap is generated to show the correlation between different features.

Model Training and Evaluation
=============================
The following machine learning models are trained and evaluated:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Decision Tree Classifier
  4. Random Forest Classifier
  5. K-Nearest Neighbors (KNN)
  6. Neural Network (MLP Classifier)

Each model is tuned using GridSearchCV from the Scikit-Learn python module to find the best hyperparameters. The models are evaluated using k-fold cross-validation (Stratified K-Fold) and the following metrics are measured:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
