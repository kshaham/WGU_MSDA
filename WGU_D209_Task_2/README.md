# 📊 Summary of WGU D209 Task 2: Predictive Modeling with Linear Regression

This Jupyter notebook explores the use of **multiple linear regression** to predict the total charge for a patient based on a variety of medical and demographic features. The research question focuses on whether patient characteristics—such as age, days of initial stay, and medical services used—can reliably predict the total hospital charge. The goal is to develop a model that provides insight into what drives hospital costs and to assist healthcare institutions in budgeting and operational planning.

## 🧹 Data Cleaning and Preparation

The dataset was first cleaned by removing personally identifiable information and irrelevant geographic variables. Survey response columns (Item1–Item8) were renamed to more descriptive names, such as `Timely_treat` and `Active_listening`. Binary variables were encoded numerically, and categorical variables were transformed using one-hot encoding. The `SelectKBest` function was employed to identify the most statistically significant predictors (based on p-values < 0.05) for inclusion in the linear regression model.

The features identified as significant included: `Initial_days`, `ReAdmis`, various `Marital` statuses, `Initial_admin` types, and `Services` received (e.g., MRI, CTScan). The data was then standardized using `StandardScaler` to meet the assumptions of linear regression and to ensure all features contributed equally to the model. A refined dataset was prepared with only the significant features and was saved as a CSV for future reference.

## 🔍 Model Building and Evaluation

A multiple linear regression model was built using the standardized features. The model achieved an **R² value of 0.83**, indicating that 83% of the variance in total hospital charges could be explained by the input features. Visual inspection of residual plots confirmed that the assumptions of linearity, homoscedasticity, and normality of residuals were reasonably met. The coefficients of the model revealed key drivers of hospital costs, such as longer stays and certain types of admission (e.g., emergency or elective).

## 🏁 Key Findings and Recommendations

The analysis concluded that features like `Initial_days`, readmission status (`ReAdmis`), and services such as `MRI` and `CTScan` were among the most influential in predicting hospital charges. These insights could help hospital administrators develop cost-reduction strategies by focusing on modifiable factors such as improving discharge processes or optimizing patient intake. Future improvements could involve regularization techniques (e.g., Ridge or Lasso regression) to handle potential multicollinearity and increase model robustness.

This notebook showcases a clear, reproducible framework for predictive analysis using real-world healthcare data. The structured use of feature selection, data standardization, and model validation makes it an excellent template for similar applications in healthcare cost modeling.
