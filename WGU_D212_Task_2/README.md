# WGU D212 Task 2: Dimension Reduction Methods

## Overview

This project explores the application of Principal Component Analysis (PCA) to reduce the dimensionality of a medical dataset while retaining meaningful patient information. The primary research question is whether PCA can effectively reduce the dataset's dimensions without losing critical insights into patient characteristics, which are vital for hospitals aiming to minimize readmission rates.

## Analysis Goal

The goal of this analysis is to simplify the medical dataset by reducing its dimensions, thereby making it easier to explore and visualize. This simplification is achieved by identifying and retaining the variables that explain the most variability. By doing so, the dataset becomes more manageable, allowing for a more straightforward analysis while preserving the most significant information. This is particularly important for hospitals that need to understand patient data to reduce readmission rates.

## Methodology

Principal Component Analysis (PCA) is employed to analyze the dataset. The process involves several key steps:

- **Standardization**: Continuous variables are standardized to ensure they contribute equally to the analysis, as PCA is sensitive to variance.
- **Covariance Matrix**: This matrix is computed to understand variable correlations, helping to identify and remove redundant information.
- **Principal Components**: Initial components are computed based on the covariance matrix, ordered by explained variance.
- **Variance Threshold**: Components explaining at least 80% of the variance are selected, reducing the dataset's dimensionality by focusing on the most important components.

## Results

The analysis results in a set of principal components that capture the majority of the dataset's variance. These components are visualized to provide insights into the data's structure, helping to identify key variables that influence patient readmissions.