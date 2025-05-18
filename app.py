import streamlit as st
st.set_page_config(page_title="ML Explorer", layout="wide")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, KernelPCA
import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import math

# --- Sidebar configuration with hover tooltips ---
st.sidebar.title("Configuration")
# Fullscreen option for plots
fullscreen = st.sidebar.checkbox(
    "Fullscreen plots",
    False,
    help="Toggle to display plots in full container width/height for detail."
)

st.sidebar.title("Configuration")
# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Select synthetic dataset:",
    ("make_classification", "make_moons", "make_circles", "make_blobs", "make_gaussian_quantiles"),
    help="Pick the type of synthetic data generation function from scikit-learn."
)
# Dataset descriptions
with st.sidebar.expander("Dataset descriptions", expanded=False):
    st.markdown(
        """
- **make_classification**: a random n-class classification problem with informative/redundant features.
- **make_moons**: two interleaving half circles suitable for non-linear classifiers.
- **make_circles**: large circle containing a smaller circle, good for kernel methods.
- **make_blobs**: isotropic Gaussian blobs for clustering and simple separability.
- **make_gaussian_quantiles**: Gaussian samples divided into discrete quantiles (classes).
        """
    )
# Samples and features
n_samples = st.sidebar.slider(
    "Number of samples:", 100, 2000, 500, step=100,
    help="Choose how many data points (rows) to generate in the dataset."
)
n_features = st.sidebar.slider(
    "Number of features:", 2, 20, 10,
    help="Select the dimensionality (number of features) for the generated data."
)
# Feature selection
fs_method = st.sidebar.selectbox(
    "Feature selection method:",
    ("None", "VarianceThreshold", "SelectKBest - ANOVA F-test", "SelectKBest - Mutual Information", "Tree-based importance"),
    help="Choose a technique to remove or select the most relevant features before training."
)
fs_k = None
if fs_method.startswith("SelectKBest") or fs_method == "Tree-based importance":
    fs_k = st.sidebar.slider(
        "Number of features to select (k):", 1, n_features, min(2, n_features),
        help="When selecting features, choose the exact number of top features to keep."
    )
# Feature reduction
fr_method = st.sidebar.selectbox(
    "Feature reduction method:",
    ("None", "PCA", "KernelPCA (RBF)", "UMAP"),
    help="Choose a dimensionality reduction method to project features into 2D space for visualization."
)
fr_components = None
if fr_method in ("PCA", "KernelPCA (RBF)", "UMAP"):
    fr_components = 2  # always reduce to 2D for plotting
# Scaling
scaler_name = st.sidebar.selectbox(
    "Scaling method:",
    ("None", "StandardScaler", "MinMaxScaler", "RobustScaler"),
    help="Apply normalization or scaling to features to improve model performance."
)

# --- Generate synthetic data ---
def get_data(name):
    if name == "make_classification":
        return datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features/2),
            n_redundant=int(n_features/4),
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
    X_train_sel, X_test_sel = sel.fit_transform(X_train), sel.transform(X_test)
elif fs_method == "SelectKBest - ANOVA F-test":
    sel = SelectKBest(score_func=f_classif, k=fs_k)
    X_train_sel, X_test_sel = sel.fit_transform(X_train, y_train), sel.transform(X_test)
elif fs_method == "SelectKBest - Mutual Information":
    sel = SelectKBest(score_func=mutual_info_classif, k=fs_k)
    X_train_sel, X_test_sel = sel.fit_transform(X_train, y_train), sel.transform(X_test)
elif fs_method == "Tree-based importance":
    model_fs = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    idxs = np.argsort(model_fs.feature_importances_)[-fs_k:]
    X_train_sel, X_test_sel = X_train[:, idxs], X_test[:, idxs]
else:
    X_train_sel, X_test_sel = X_train, X_test
# --- Feature Reduction ---
if fr_method == "PCA":
    reducer = PCA(n_components=fr_components)
    X_train_red, X_test_red = reducer.fit_transform(X_train_sel), reducer.transform(X_test_sel)
elif fr_method == "KernelPCA (RBF)":
    reducer = KernelPCA(n_components=fr_components, kernel="rbf", gamma=0.1)
    X_train_red, X_test_red = reducer.fit_transform(X_train_sel), reducer.transform(X_test_sel)
elif fr_method == "UMAP":
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_train_red, X_test_red = reducer.fit_transform(X_train_sel), reducer.transform(X_test_sel)
else:
    X_train_red, X_test_red = X_train_sel, X_test_sel
# --- Scaling ---
if scaler_name == "StandardScaler":
    scaler = StandardScaler()
    X_train_pre, X_test_pre = scaler.fit_transform(X_train_red), scaler.transform(X_test_red)
elif scaler_name == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_train_pre, X_test_pre = scaler.fit_transform(X_train_red), scaler.transform(X_test_red)
elif scaler_name == "RobustScaler":
    scaler = RobustScaler()
    X_train_pre, X_test_pre = scaler.fit_transform(X_train_red), scaler.transform(X_test_red)
else:
    X_train_pre, X_test_pre = X_train_red, X_test_red
# Ensure 2D for plotting
if X_train_pre.shape[1] < 2:
    st.error("Need at least 2 dimensions after preprocessing for plotting.")
    st.stop()
# --- Setup models ---
models = {
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
    "Passive Aggressive": SGDClassifier(max_iter=1000, loss="hinge"),
    # Anomaly detection
    "Isolation Forest": IsolationForest(random_state=42),
    "One-Class SVM": OneClassSVM(gamma='auto'),
    "Local Outlier Factor": LocalOutlierFactor(novelty=True)
}
# --- Evaluate models ---
results = []
for name, model in models.items():
    est = clone(model)
    # Fit
    if name == "Local Outlier Factor":
        est.fit(X_train_pre)
        y_pred_raw = est.predict(X_test_pre)
    else:
        est.fit(X_train_pre, y_train if name not in ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"] else None)
        y_pred_raw = est.predict(X_test_pre)
    # Map anomaly outputs to 0/1
    if name in ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]:
        y_pred = (y_pred_raw > 0).astype(int)
    else:
        y_pred = y_pred_raw
    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0
    tnr = tn/(tn+fp) if (tn+fp)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    fnr = fn/(fn+tp) if (fn+tp)>0 else 0
    precision = precision_score(y_test, y_pred)
    recall = tpr
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    gmean = math.sqrt(tpr*tnr)
    results.append({
        "Model": name,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "TPR": tpr, "TNR": tnr, "FPR": fpr, "FNR": fnr,
        "Accuracy": accuracy, "Precision": precision,
        "Recall": recall, "F1-Score": f1, "G-Mean": gmean
    })
# Display metrics
metrics_df = pd.DataFrame(results)
st.subheader("Performance Metrics on Test Set")
st.dataframe(metrics_df, use_container_width=True)
# Plot decision boundaries
x_vis = X_train_pre[:, :2]
for _, row in metrics_df.iterrows():
    name = row["Model"]
    exp = st.expander(f"Decision Boundary: {name}")
    with exp:
        est_vis = clone(models[name])
        if name in ["Local Outlier Factor"]:
            est_vis.fit(x_vis)
        else:
            est_vis.fit(x_vis, y_train if name not in ["Isolation Forest", "One-Class SVM"] else None)
        xx, yy = np.meshgrid(
            np.linspace(x_vis[:,0].min()-1, x_vis[:,0].max()+1, 200),
            np.linspace(x_vis[:,1].min()-1, x_vis[:,1].max()+1, 200)
        )
        Z_raw = est_vis.predict(np.c_[xx.ravel(), yy.ravel()])
        if name in ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]:
            Z = (Z_raw>0).astype(int).reshape(xx.shape)
        else:
            Z = Z_raw.reshape(xx.shape)
        fig_w, fig_h = (12, 8) if fullscreen else (4, 3)
        plt.figure(figsize=(fig_w, fig_h))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(x_vis[:,0], x_vis[:,1], c=y_train, edgecolor='k', s=20)
        plt.title(name)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        st.pyplot(plt)

