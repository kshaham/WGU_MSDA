# 🏘️ Summary of `gnn_real_estate.py`: Predicting CT Home Prices Using a Graph Neural Network

This Python script builds a comprehensive pipeline for predicting single-family residential property prices in Connecticut using a **Graph Neural Network (GNN)**—specifically a **GraphSAGE** architecture. The approach combines geospatial relationships, engineered features, and real estate market data to model housing prices more accurately by leveraging the structure of a property graph.

## 🧹 Data Preparation and Feature Engineering

The dataset, filtered from `CT_Real_Estate_Listings_Updated.csv`, includes Connecticut single-family homes priced between $100,000 and $10,000,000. The script performs extensive feature engineering, including:
- **Price per square foot**
- **Property age**
- **Bathroom-to-bedroom ratios**
- **Lot-to-living area ratios**

Missing values and outliers are handled using median imputation and scaling, and categorical features (`isNew`, `isHot`, `is_virtual_tour`) are one-hot encoded. The target variable (`price`) is log-transformed and standardized for modeling.

## 🌐 Graph Construction

To represent spatial relationships between properties, the script constructs a **k-nearest neighbors (KNN) graph** using latitude and longitude. Edge weights are calculated using an exponential decay function based on haversine distances. Invalid or out-of-bounds edges are filtered to maintain graph integrity.

## 🧠 Model Architecture and Training

A custom **GraphSAGE neural network** is defined with residual connections and dropout layers to prevent overfitting. The model takes node features and the spatial graph as input and outputs price predictions. Training is done using:
- **AdamW optimizer**
- **MSE loss function**
- **Learning rate scheduling**
- **Early stopping based on validation loss**

The model is trained on a GPU if available, with support for fallback to CPU.

## 📊 Evaluation and Results

Post-training, the model's predictions are evaluated using:
- **MAE, RMSE, R², and Adjusted R²**
- **MAPE and Median Absolute Percentage Error**
- **Feature importance** via perturbation
- **Visualization** of actual vs. predicted prices on a map and through loss/error plots

The model achieves solid predictive performance, identifying `livingArea`, `property_age`, and `distance_to_coast` as some of the most influential features.

---

This script showcases an end-to-end application of Graph Neural Networks for real estate price modeling, demonstrating how spatial relationships and engineered features can be combined to enhance predictive accuracy in geospatial property datasets.
