import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import math

# --- Sidebar configuration ---
st.sidebar.title("Configuration")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Select synthetic dataset:",
    ("make_classification", "make_moons", "make_circles", "make_blobs", "make_gaussian_quantiles")
)
# Samples and features
n_samples = st.sidebar.slider("Number of samples:", 100, 2000, 500, step=100)
n_features = st.sidebar.slider("Number of features:", 2, 20, 10)

# Feature selection
fs_method = st.sidebar.selectbox(
    "Feature selection method:",
    ("None", "VarianceThreshold", "SelectKBest - ANOVA F-test", "SelectKBest - Mutual Information", "Tree-based importance")
)
fs_k = None
if fs_method.startswith("SelectKBest") or fs_method == "Tree-based importance":
    fs_k = st.sidebar.slider("Number of features to select (k):", 1, n_features, min(2, n_features))

# Feature reduction
fr_method = st.sidebar.selectbox(
    "Feature reduction method:",
    ("None", "PCA", "KernelPCA (RBF)")
)
fr_components = None
if fr_method in ("PCA", "KernelPCA (RBF)"):
    fr_components = 2  # reduce to 2D for plotting

# Scaling
scaler_name = st.sidebar.selectbox(
    "Scaling method:",
    ("None", "StandardScaler", "MinMaxScaler", "RobustScaler")
)

# --- Generate synthetic data ---
def get_data(name):
    if name == "make_classification":
        return datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features / 2),
            n_redundant=int(n_features / 4),
            n_clusters_per_class=1,
            random_state=42
        )
    elif name == "make_moons":
        return datasets.make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif name == "make_circles":
        return datasets.make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=42)
    elif name == "make_blobs":
        return datasets.make_blobs(n_samples=n_samples, centers=2, n_features=n_features, random_state=42)
    elif name == "make_gaussian_quantiles":
        return datasets.make_gaussian_quantiles(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)
    else:
        raise ValueError("Unknown dataset")

X, y = get_data(dataset_name)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Feature Selection ---
if fs_method == "VarianceThreshold":
    sel = VarianceThreshold(threshold=0.1)
    X_train_sel = sel.fit_transform(X_train)
    X_test_sel = sel.transform(X_test)
elif fs_method == "SelectKBest - ANOVA F-test":
    sel = SelectKBest(score_func=f_classif, k=fs_k)
    X_train_sel = sel.fit_transform(X_train, y_train)
    X_test_sel = sel.transform(X_test)
elif fs_method == "SelectKBest - Mutual Information":
    sel = SelectKBest(score_func=mutual_info_classif, k=fs_k)
    X_train_sel = sel.fit_transform(X_train, y_train)
    X_test_sel = sel.transform(X_test)
elif fs_method == "Tree-based importance":
    from sklearn.ensemble import RandomForestClassifier as RFC
    sel_model = RFC()
    sel_model.fit(X_train, y_train)
    importances = sel_model.feature_importances_
    idxs = np.argsort(importances)[-fs_k:]
    X_train_sel = X_train[:, idxs]
    X_test_sel = X_test[:, idxs]
else:
    X_train_sel, X_test_sel = X_train, X_test

# --- Feature Reduction ---
if fr_method == "PCA":
    reducer = PCA(n_components=fr_components)
    X_train_red = reducer.fit_transform(X_train_sel)
    X_test_red = reducer.transform(X_test_sel)
elif fr_method == "KernelPCA (RBF)":
    reducer = KernelPCA(n_components=fr_components, kernel="rbf", gamma=0.1)
    X_train_red = reducer.fit_transform(X_train_sel)
    X_test_red = reducer.transform(X_test_sel)
else:
    X_train_red, X_test_red = X_train_sel, X_test_sel

# --- Scaling ---
if scaler_name == "StandardScaler":
    scaler = StandardScaler()
    X_train_pre = scaler.fit_transform(X_train_red)
    X_test_pre = scaler.transform(X_test_red)
elif scaler_name == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_train_pre = scaler.fit_transform(X_train_red)
    X_test_pre = scaler.transform(X_test_red)
elif scaler_name == "RobustScaler":
    scaler = RobustScaler()
    X_train_pre = scaler.fit_transform(X_train_red)
    X_test_pre = scaler.transform(X_test_red)
else:
    X_train_pre, X_test_pre = X_train_red, X_test_red

# Ensure 2D for plotting
if X_train_pre.shape[1] < 2:
    st.error("Need at least 2 dimensions after preprocessing for plotting.")
    st.stop()

# --- Setup classifiers ---
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Linear SVM": LinearSVC(max_iter=5000),
    "Kernel SVM (RBF)": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Bagging": BaggingClassifier(),
    "Gaussian NB": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "MLP": MLPClassifier(max_iter=1000),
    "SGD": SGDClassifier(max_iter=1000),
    "Passive Aggressive": SGDClassifier(max_iter=1000, loss="hinge")
}

# Iterate through classifiers and compute metrics
results = []
for name, clf in classifiers.items():
    model = clone(clf)
    model.fit(X_train_pre, y_train)
    y_pred = model.predict(X_test_pre)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # Rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity / Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = precision_score(y_test, y_pred)
    recall = tpr
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    gmean = math.sqrt(tpr * tnr)
    results.append({
        "Classifier": name,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "TPR": tpr, "TNR": tnr, "FPR": fpr, "FNR": fnr,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "G-Mean": gmean
    })

# Display metrics table (sortable)
metrics_df = pd.DataFrame(results)
st.subheader("Performance Metrics on Test Set")
st.dataframe(metrics_df, use_container_width=True)

# Plot decision boundaries
x_vis_train = X_train_pre[:, :2]
for _, row in metrics_df.iterrows():
    name = row["Classifier"]
    exp = st.expander(f"Decision Boundary: {name}")
    with exp:
        model_vis = clone(classifiers[name])
        model_vis.fit(x_vis_train, y_train)
        x_min, x_max = x_vis_train[:, 0].min() - 1, x_vis_train[:, 0].max() + 1
        y_min, y_max = x_vis_train[:, 1].min() - 1, x_vis_train[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.figure(figsize=(6, 4))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(x_vis_train[:, 0], x_vis_train[:, 1], c=y_train, edgecolor='k', s=20)
        plt.title(name)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        st.pyplot(plt)

