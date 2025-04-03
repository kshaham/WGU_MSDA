# 🛒 Summary of WGU D212 Task 3: Pattern Prediction with Market Basket Analysis

This notebook applies **Market Basket Analysis (MBA)** to healthcare service data to discover patterns in how patient services are grouped. The research question focuses on whether patterns of services co-occurring can be identified to better understand patient needs and optimize service delivery. The goal is to use **association rule mining** to uncover rules that can guide resource allocation, streamline operations, and improve patient care outcomes.

## 🧾 Transactions and Technique Justification

Market Basket Analysis is based on the idea that **co-occurring services or behaviors** can reveal meaningful associations. For example, patients receiving an MRI may often also receive a CT scan. The notebook uses a binary-encoded dataset where each row represents a patient and each column indicates whether a specific service (e.g., `MRI`, `CTScan`, `BloodWork`) was used. The assumption is that analyzing such transaction-like records can yield association rules similar to those found in retail analytics.

The `mlxtend` library is used to generate frequent itemsets with the **Apriori algorithm**. Rules are then derived from these itemsets based on support, confidence, and lift. These metrics quantify how common a rule is, how often it is correct, and how much stronger it is compared to random chance.

## 📊 Association Rule Discovery and Results

The Apriori algorithm identifies frequent combinations of services with a minimum support threshold (e.g., 0.01). The `association_rules` function is used to filter the results based on meaningful thresholds for **confidence** and **lift**. Rules such as `{BloodWork} → {CTScan}` with high lift values indicate strong associations.

Visualizations include bar plots of the top frequent itemsets and rules, helping to easily interpret the strength and relevance of identified patterns. These insights highlight combinations of services that could be bundled, scheduled together, or prioritized for certain types of patients.

## 💡 Summary and Business Implications

This pattern prediction task provides a **data-driven approach to service bundling** in the healthcare domain. It uncovers frequently co-occurring medical services that can inform decision-making in areas like staff scheduling, equipment placement, and patient workflow optimization. For instance, identifying that patients receiving blood work are also likely to receive CT scans can lead to streamlined protocols that reduce wait times and improve service quality.

The project showcases the power of market basket analysis beyond retail, providing a robust toolkit for pattern discovery in healthcare settings. Future work could involve incorporating temporal or demographic data to further refine the service pairing logic.
