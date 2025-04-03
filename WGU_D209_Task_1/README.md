# 🧠 Summary of WGU D209 Task 1: Classification Analysis using KNN

This report focuses on building a predictive model to identify patients at risk of **hospital readmission** using a classification method. The research question centers on whether patient readmission can be predicted based on medical and demographic attributes. The **k-nearest neighbors (KNN)** algorithm was selected due to its interpretability and suitability for binary classification tasks. The study leverages patient data to build and evaluate two KNN models and provides recommendations for hospitals to act on the results.

## 🔬 Methodology and Data Preparation

The data preprocessing phase involved cleaning the dataset by removing personally identifiable and irrelevant columns such as geographic identifiers, job, and customer ID. Missing values were handled, and categorical data was converted using binary encoding or one-hot encoding, depending on the type of variable. Columns representing patient survey responses (Item1–Item8) were renamed for clarity (e.g., `Timely_admis`, `Reliability`). The `SelectKBest` function was used to extract statistically significant predictors based on a p-value threshold of 0.05.

Data was standardized using `StandardScaler` before being split into **training (75%) and testing (25%)** datasets. This ensured that all features contributed equally to distance calculations in KNN. Variables with the highest significance included `TotalCharge`, `Initial_days`, various `Services` (e.g., MRI, CTScan), marital status, and type of initial admission.

## 🤖 Model Development and Evaluation

Two KNN models were trained: one using the default `k=5` and another using `k=86` (based on the square root rule for the number of training instances). The model with `k=5` achieved **97% accuracy**, **97% sensitivity**, and **97% specificity**, outperforming the second model, which yielded 90% accuracy. ROC-AUC scores further confirmed the superior performance of the first model (AUC = 0.99 vs. 0.97), indicating near-perfect classification capability.

## 📈 Implications and Recommendations

The high accuracy and AUC of the initial KNN model suggest it is effective for predicting readmissions. Hospitals can use these results to proactively identify high-risk patients and tailor interventions to reduce readmission rates, improve care, and cut costs. However, the report notes that **hyperparameter tuning** (e.g., varying `k`, exploring different distance metrics) could further optimize performance. The recommendation is to explore these enhancements and perform additional validation to ensure generalizability.

This project exemplifies how machine learning techniques like KNN, when paired with sound feature selection and preprocessing, can offer **actionable insights in healthcare** and enhance patient outcome forecasting.
