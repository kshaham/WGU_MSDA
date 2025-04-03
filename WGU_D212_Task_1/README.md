# 🔍 Summary of WGU D212 Task 1: Clustering Techniques for Patient Segmentation

This Jupyter notebook focuses on applying **unsupervised machine learning**, specifically clustering techniques, to group patients based on shared characteristics. The research question seeks to explore: *Can patients be grouped using clustering methods based on their observed medical data?* The goal is to help healthcare providers better understand patient subgroups, enabling more personalized care strategies and resource allocation.

## 🧪 Technique and Data Preparation

The notebook justifies the use of **K-Means Clustering**, a technique that groups data points into `k` clusters based on proximity in feature space. The key assumption is that patients within the same cluster share similar characteristics. Python packages used include `Pandas`, `Scikit-learn`, `Matplotlib`, and `Seaborn` for data processing, clustering, and visualization.

Data preparation steps involve cleaning the dataset by removing unnecessary columns (e.g., IDs and geographic data), encoding categorical variables, and normalizing features to standardize scale. Significant features for clustering are selected using domain knowledge and statistical techniques like `SelectKBest`. Columns such as `Initial_days`, `TotalCharge`, `ReAdmis`, and selected patient experience survey responses (e.g., `Timely_treat`, `Active_listening`) are retained for clustering.

## 📊 Cluster Analysis and Evaluation

The elbow method is used to determine the optimal number of clusters, visualized using the **inertia score** curve. K-Means clustering is then performed using the identified optimal `k`, and cluster assignments are appended to the dataset. Visualizations such as **pair plots**, **cluster centroids**, and **distribution plots** are used to interpret the characteristics of each cluster.

The results show meaningful patient groupings, with distinct clusters varying in terms of hospital stay length, service usage (e.g., MRI or CTScan), and cost. This segmentation reveals patterns such as high-cost/high-visit groups versus low-visit/low-cost groups, providing actionable insights for healthcare optimization.

## 🧠 Insights and Recommendations

By grouping patients into clusters, healthcare organizations can tailor interventions—such as offering additional care coordination to high-risk groups or preventive care to low-engagement groups. The analysis demonstrates the power of clustering as a tool for **patient stratification** and **healthcare planning**. Future work could include hierarchical clustering for comparison or integrating additional time-series patient data for more dynamic modeling.

Overall, the notebook effectively applies K-Means clustering to real-world healthcare data, offering a scalable and insightful approach to uncover hidden structures within patient populations.
