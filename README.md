# 🚀 ML Explorer

A Streamlit app to explore classification and anomaly-detection pipelines on synthetic datasets! 🧪


<img width="1863" alt="image" src="https://github.com/user-attachments/assets/c5f1048a-0551-4fb6-b60c-d7965a5a990d" />


## 🔍 Features

* **Synthetic Data Generators**:

  * `make_classification`, `make_moons`, `make_circles`, `make_blobs`, `make_gaussian_quantiles`
* **Feature Selection**: VarianceThreshold, SelectKBest (ANOVA F-test, Mutual Information), tree-based importance
* **Dimensionality Reduction**: PCA, Kernel PCA, UMAP
* **Scaling**: StandardScaler, MinMaxScaler, RobustScaler
* **Classification & Anomaly Detection**:

  * Logistic Regression, SVM, k-NN, Decision Trees, Random/Extra Forests, AdaBoost, GradientBoosting, Bagging, GaussianNB, QDA, MLP, SGD, Passive-Aggressive
  * IsolationForest, One-Class SVM, Local Outlier Factor
* **Interactive Metrics**: Confusion matrix counts (TP, TN, FP, FN), Accuracy, Precision, Recall, F1-score, G-Mean, TPR, TNR, FPR, FNR
* **Decision Boundary Visualizations**: 2D plots with fullscreen toggle 📈

## ⚙️ Installation

1. Clone this repo:

   ```bash
   git clone https://github.com/yourusername/ml-explorer.git
   cd ml-explorer
   ```
2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

* Use the sidebar to select dataset, preprocessing steps, and models
* View performance metrics in an interactive table (sortable)
* Expand each model's section to see its decision boundary plot
* Toggle **Fullscreen plots** to enlarge charts inside the main view

## 🤝 Contributing

Feel free to open issues or PRs! ⭐

## 📜 License

MIT License © 2025 eldraco
