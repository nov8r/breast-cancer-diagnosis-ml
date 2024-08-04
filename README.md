Breast Cancer Diagnosis Machine Learning Model
==============================================

This project involved developing a machine learning model to predict whether cancer is malignant or benign based on physical characteristics. The dataset used in this project contains various features that describe the characteristics of cell nuclei present in the digitized images of a fine needle aspirate (FNA) of a breast mass.

Introduction
------------
The primary objective of this project is to classify cancer as either malignant or benign based on various features obtained from digitized images of FNA of breast masses. The data comes from the Breast Cancer Wisconsin dataset. This dataset was obtained from UCI Machine Learning on Kaggle and contains 569 instances of different breast cancer cases.

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

![alt text](https://github.com/nov8r/FP/blob/main/histogram.png "Distribution Histogram")

### Boxplots
Boxplots help in identifying the presence of outliers in the dataset.

![alt text](https://github.com/nov8r/FP/blob/main/boxplot.png "Boxplot")

When it comes to machine learning, you would usually remove outliers from your data so as not to skew it too much especially if your boxplots look anything like mine. That said, I did not remove outliers because, in the context of this data, you don't want to miss outliers. You want to be still able to identify those that are out of the average range of the rest of the data.

### Correlation Heatmap
A correlation heatmap is generated to show the correlation between different features.

![alt text](https://github.com/nov8r/FP/blob/main/correlation_heatmap.png "Correlation Heatmap")

Model Training and Evaluation
=============================
The following machine learning models are trained and evaluated:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Decision Tree Classifier
  4. Random Forest Classifier
  5. K-Nearest Neighbors (KNN)
  6. Neural Network (MLP Classifier)

Each model is tuned using GridSearchCV from the Scikit-Learn Python module to find the best hyperparameters. The models are evaluated using k-fold cross-validation (Stratified K-Fold) and the following metrics are measured:
  - Accuracy
  - F1 Score
  - Precision
  - Recall

Results
-------
The performance of each model is summarized in the table below:
| Model                        | Accuracy | fScore | Precision | Recall |
| ---------------------------- | -------: | -----: | --------: | -----: |
| Neural Network               | 0.97     | 0.95   | 1.00      | 0.91   |
| Support Vector Machine (SVM) | 1.00     | 1.00   | 1.00      | 1.00   |
| K-Nearest Neighbors (KNN)    | 0.93     | 0.90   | 1.00      | 0.82   |
| Decision Tree                | 0.86     | 0.78   | 1.00      | 0.64   |
| Random Forest                | 0.90     | 0.84   | 1.00      | 0.73   |
| Logistic Regression          | 0.97     | 0.95   | 1.00      | 0.91   |

Conclusion
==========
This project is my first significant deep dive into machine learning and while the results are promising, there are certainly areas to improve on. For instance, the results from the SVM model suggest perfect performance in prediction which is highly unlikely. Additionally, the precision scores provided are also seemingly perfect, indicating that there may be an issue in the evaluation process or data handling.

Despite these flaws, the overall performance of the models is comparable to those found in other projects and academic papers which is encouraging. I will continue working on this project to hopefully resolve the flaws detailed earlier.

