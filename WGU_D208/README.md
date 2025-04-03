# 📘 Summary of WGU D208 Predictive Modeling Report

This research project addresses the relationship between various patient medical and demographic variables and the **initial length of hospital stay**, with the ultimate goal of identifying factors that can help **reduce readmission rates**. The study is framed around the research question: *Is there a correlation between patient observations and the initial length of their hospital stay?* The report thoroughly documents the steps from exploratory data analysis to multiple linear regression modeling, providing hospitals with actionable insights for improving patient outcomes and operational efficiency.

## 📊 Methodology and Data Preparation

The methodology relies on **multiple linear regression** to identify predictors of initial hospital stay length. Prior to modeling, the dataset undergoes extensive cleaning: irrelevant features like geographic identifiers (e.g., ZIP code, city) and personally identifying information (e.g., UID, Job, Gender) are removed. Categorical variables are encoded using one-hot encoding, and survey-related columns (Item1–Item8) are renamed for interpretability. This transformation ensures compatibility with linear regression assumptions and improves model clarity.

The dataset is verified for completeness, showing **no missing or duplicated records**. Statistical summaries and visual inspections (histograms, boxplots, and scatterplots) are used to assess distribution and identify potential outliers in variables such as income, meals eaten, and vitamin D supplements. Variables that are not normally distributed are noted, though linear regression is still employed due to its interpretability and usefulness in generating actionable results for healthcare settings.

## 📉 Model Building and Evaluation

Two linear regression models are built: a **full model** including all variables, and a **reduced model** created by removing predictors with low significance or high multicollinearity. The report includes detailed comparisons of these models using evaluation metrics such as R² and adjusted R². Visual analysis, including scatterplots between predictors and the `Initial_days` variable, is used to assess linearity and inform model refinement.

## 📌 Key Findings and Implications

The analysis identifies several variables with strong correlations to hospital stay length, such as patient age, vitamin D levels, frequency of doctor visits, and responses from patient experience surveys (e.g., Timely treatment, Courteous staff). By focusing on these variables, hospitals can tailor interventions to manage resources more efficiently and reduce both the cost and risk associated with extended stays or readmissions. The study concludes with a recommendation for using the reduced model for real-world applications due to its simplicity and similar predictive power compared to the full model.

---

This project exemplifies how predictive modeling can directly support healthcare decision-making. By operationalizing patient and administrative data into actionable predictors, the study provides a framework for continuous hospital process improvement.
