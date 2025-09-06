# COVID-19 Risk Prediction Using Synthetic Data

## Project Overview

This project analyzes a synthetic dataset of 10,000 patient medical records to predict COVID-19 survivability based on health features. The research explores machine learning models, feature importance, clustering patterns, and dimensionality reduction techniques to identify key factors in COVID-19 outcomes.

## Key Features

* **Data Analysis:** Comprehensive preprocessing of synthetic medical data
* **Machine Learning:** Implementation of four classification models (Logistic Regression, Naive Bayes, SVM, ANN)
* **Feature Importance:** Identification of key health predictors for COVID-19 survival
* **Clustering Analysis:** K-means clustering to identify patient health patterns
* **Dimensionality Reduction:** PCA implementation for efficient feature reduction

## Dataset

The synthetic dataset was derived from three peer-reviewed clinical papers and contains:

* 10,000 patient records (8,759 survivors, 2,352 deaths)
* 164 original features reduced to 13 key health indicators after preprocessing
* Medical observations, patient demographics, and encounter data from after January 20, 2020

**Key variables include:**

* QALY (Quality-Adjusted Life Year)
* DALY (Disability-Adjusted Life Year)
* QOLS (Quality of Life Scale)
* Vital signs (respiratory rate, heart rate, blood pressure, etc.)
* Body measurements (height, weight, BMI)

## Methodology

### Preprocessing

* **Missing Value Handling:** Features with >60% missing values were removed; remaining missing values were imputed using MICE
* **Outlier Treatment:** Winsorization method applied to handle extreme values
* **Multicollinearity Reduction:** VIF analysis used to remove highly correlated predictors
* **Normalization:** Z-score normalization applied to all features
* **Class Balancing:** SMOTE oversampling technique applied to address class imbalance

### Modeling Approaches

* **Classification Models:** Logistic Regression, Naive Bayes, SVM, and Artificial Neural Network
* **Clustering:** K-means with elbow and silhouette methods for optimal cluster identification
* **Dimensionality Reduction:** Principal Component Analysis (PCA) for feature reduction

## Results

### Key Findings

* **Top Predictors:** QALY, respiratory rate, and body height were most significant in predicting survival
* **Best Performing Model:** Naive Bayes achieved the highest AUC score (0.485) on precision-recall curves
* **Clustering:** Optimal clustering identified at k=2 using silhouette analysis
* **Dimensionality Reduction:** PCA reduced features by 75% (12→9) with only 3% sensitivity decrease

### Performance Metrics

| Model               | Sensitivity | Accuracy | F1-Score |
| ------------------- | ----------- | -------- | -------- |
| Naive Bayes         | 23%         | 87%      | 36%      |
| ANN                 | 33%         | 93%      | 47%      |
| SVM                 | 17%         | 82%      | 28%      |
| Logistic Regression | 16%         | 83%      | 28%      |

## Installation & Usage

### Prerequisites

* Python 3.7+
* Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, statsmodels, imbalanced-learn

### Installation

```bash
git clone [repository-url]
cd covid-risk-prediction
pip install -r requirements.txt
```

### Running the Analysis

The project contains several Jupyter notebooks for different analysis stages:

* **Preprocessing:** preprocessing.ipynb
* **Logistic Regression:** logistic\_regression\_model.ipynb
* **Naive Bayes:** naive\_bayes\_model.ipynb
* **SVM:** svm\_model.ipynb
* **ANN:** ann\_model.ipynb
* **ROC Analysis:** roc\_curve\_plot.ipynb
* **Clustering:** kmeans.ipynb
* **PCA:** pca.ipynb

Run notebooks in order or execute specific analysis components as needed.

## Project Structure

```
covid-risk-prediction/
├── data/
│   ├── conditions.csv
│   ├── patients.csv
│   ├── observations.csv
│   ├── careplans.csv
│   └── encounters.csv
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── logistic_regression_model.ipynb
│   ├── naive_bayes_model.ipynb
│   ├── svm_model.ipynb
│   ├── ann_model.ipynb
│   ├── roc_curve_plot.ipynb
│   ├── kmeans.ipynb
│   └── pca.ipynb
├── results/
│   ├── figures/
│   └── processed_data/
├── requirements.txt
└── README.md
```

## References

* Walonoski, J., et al. (2020). Synthea™ Novel Coronavirus (COVID-19) model and Synthetic Data Set
* Prieto, L., & Sacristán, J. A. (2003). Problems and solutions in estimating quality-adjusted life years
* Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
* Wold, S., et al. (1987). Principal component analysis

## Author

Thuppahiralalage Eranga De Saa
University of British Columbia

## License

This project is for academic research purposes. Please cite appropriately if using any part of this work.
