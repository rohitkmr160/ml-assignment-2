# Machine Learning Assignment 2 - Classification Models & Streamlit Deployment

## 1. Problem Statement
Build and compare five machine-learning classifiers that predict whether a breast tumour is **malignant or benign** from measurements of cell nuclei taken from digitised fine-needle-aspirate images. The models are evaluated with Accuracy, AUC, Precision, Recall, F1 and MCC, and are deployed in an interactive Streamlit web app where a user can upload test data, pick a model and see its evaluation results.

## 2. Dataset Description
- **Name:** Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository (also bundled with scikit-learn as `load_breast_cancer`).
- **Type:** Binary classification (0 = malignant, 1 = benign).
- **Instances:** 569 (minimum required: 500).
- **Features:** 30 numeric features (minimum required: 12) - the mean, standard error and "worst" value of 10 nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Class balance:** 212 malignant, 357 benign. No missing values.
- **Split:** 80/20 stratified train/test split (`random_state=42`) -> 455 training and 114 test rows. The test rows are saved in `test_data.csv`.
- **Preprocessing:** `StandardScaler` for Logistic Regression, KNN and SVM (built into each saved pipeline); Decision Tree and Naive Bayes use raw features.

## 3. GitHub Repository Link
https://github.com/rohitkmr160/ml-assignment-2

Live Streamlit app: https://rohitkumar-2026aj05332-ml-assignment2.streamlit.app

## 4. Models Used
Logistic Regression, Decision Tree (max_depth=5), K-Nearest Neighbors (k=5), Gaussian Naive Bayes, Support Vector Machine (RBF kernel). All were trained on the same training split and evaluated on the same test split.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall model: highest AUC (0.9954) and tied-best accuracy/F1/MCC. The classes are close to linearly separable once features are standardised, so a simple linear model generalises very well and is also easy to interpret. |
| Decision Tree | Weakest model (accuracy 0.9211, AUC 0.9163, MCC 0.8341). A single depth-limited tree has high variance and splits the feature space coarsely, so it misclassifies more borderline tumours and gives less well-ranked probabilities. |
| KNN | Good, balanced performance (accuracy 0.9561, recall 0.9722) after scaling. It is a little behind LR/SVM because distance in 30 dimensions is noisier and results depend on k and the training sample. |
| Naive Bayes | Accuracy of 0.9386 is modest, but AUC is high (0.9878), so it ranks cases well. The independence assumption is violated because many features are strongly correlated (e.g. radius, perimeter, area), which hurts the calibrated decision threshold. |
| Support Vector Machine | Ties Logistic Regression on accuracy, precision, recall, F1 and MCC with a near-identical AUC (0.9950). The RBF kernel on scaled features separates the classes very well; it is slightly more costly to train and less interpretable. |
| **Overall, Winner for your dataset?** | **Logistic Regression** - best AUC and joint-best on every other metric (SVM ties on the threshold metrics), and it is the simplest, fastest and most interpretable. Note: the test set has only 114 rows, so differences of one or two predictions between the top models are not statistically meaningful. |

## 5. Execution Environment
All models were trained and evaluated on **BITS Virtual Lab** by running `python train_models.py` (screenshot included in the submitted PDF). The saved models in `model/` were produced by that run.

## 6. Repository Structure
```
ml-assignment-2/
|-- app.py                # Streamlit app
|-- train_models.py       # trains, evaluates and saves the 5 models
|-- requirements.txt
|-- README.md
|-- test_data.csv         # held-out test set (30 features + target)
|-- model/                # saved .joblib models + metrics_summary.csv
```

## 7. Run Locally / App Usage
```
pip install -r requirements.txt
python train_models.py        # optional: re-create models and test_data.csv
streamlit run app.py
```
Then upload `test_data.csv` in the sidebar, select a model, and view metrics, confusion matrix and classification report. Models are loaded from `model/`; nothing is retrained in the app.

### Streamlit app features
- CSV upload (use `test_data.csv`; must contain the 30 feature columns and `target`)
- Model selection dropdown (5 models)
- Display of Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix and classification report
- Side-by-side comparison of all 5 models on the uploaded data
