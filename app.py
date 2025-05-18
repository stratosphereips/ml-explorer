import streamlit as st
import numpy as np
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
    fr_components = 2  # For 2D plotting

# Scaling
scaler_name = st.sidebar.selectbox(
    "Scaling method:",
    ("None", "StandardScaler", "MinMaxScaler", "RobustScaler")
)

# Classifier selection
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
classifier_name = st.sidebar.selectbox("Classifier:", list(classifiers.keys()))

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

# --- Split and preprocess ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Selection
if fs_method == "VarianceThreshold":
    sel = VarianceThreshold(threshold=0.1)
    X_train = sel.fit_transform(X_train)
    X_test = sel.transform(X_test)
elif fs_method == "SelectKBest - ANOVA F-test":
    sel = SelectKBest(score_func=f_classif, k=fs_k)
    X_train = sel.fit_transform(X_train, y_train)
    X_test = sel.transform(X_test)
elif fs_method == "SelectKBest - Mutual Information":
    sel = SelectKBest(score_func=mutual_info_classif, k=fs_k)
    X_train = sel.fit_transform(X_train, y_train)
    X_test = sel.transform(X_test)
elif fs_method == "Tree-based importance":
    from sklearn.ensemble import RandomForestClassifier as RFC
    sel_model = RFC()
    sel_model.fit(X_train, y_train)
    importances = sel_model.feature_importances_
    idxs = np.argsort(importances)[-fs_k:]
    X_train = X_train[:, idxs]
    X_test = X_test[:, idxs]

# Feature Reduction
if fr_method == "PCA":
    reducer = PCA(n_components=fr_components)
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)
elif fr_method == "KernelPCA (RBF)":
    reducer = KernelPCA(n_components=fr_components, kernel="rbf", gamma=0.1)
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

# Scaling
if scaler_name == "StandardScaler":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
elif scaler_name == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
elif scaler_name == "RobustScaler":
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# --- Train classifier ---
model = classifiers[classifier_name]
model.fit(X_train, y_train)

# --- Plot decision boundary ---
# We limit to 2D for visualization
if X_train.shape[1] != 2:
    st.warning("Data is not 2D; plotting uses first two features/components.")
# create mesh grid
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
ny_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(ny_min, y_max, 200)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
# plot training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', s=20)
plt.title(f"Decision Boundary: {classifier_name}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
st.pyplot(plt)
