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

# --- Sidebar configuration ---
st.sidebar.title("Configuration")
# Fullscreen toggle for plots
fullscreen = st.sidebar.checkbox(
    "Fullscreen plots",
    False,
    help="When enabled, plots will expand to fill the container width."
)
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
    "Number of samples:", 100, 2000, 500, step=100
)
n_features = st.sidebar.slider(
    "Number of features:", 2, 20, 10
)
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
    ("None", "PCA", "KernelPCA (RBF)", "UMAP")
)
fr_components = 2 if fr_method in ("PCA", "KernelPCA (RBF)", "UMAP") else None
# Scaling
scaler_name = st.sidebar.selectbox(
    "Scaling method:",
    ("None", "StandardScaler", "MinMaxScaler", "RobustScaler")
)

# --- Generate & preprocess data ---
def get_data(name):
    if name == "make_classification":
        return datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                           n_informative=n_features//2, n_redundant=n_features//4,
                                           n_clusters_per_class=1, random_state=42)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Feature selection
if fs_method == "VarianceThreshold":
    sel = VarianceThreshold(0.1); X_train, X_test = sel.fit_transform(X_train), sel.transform(X_test)
elif fs_method == "SelectKBest - ANOVA F-test":
    sel = SelectKBest(f_classif, k=fs_k); X_train, X_test = sel.fit_transform(X_train,y_train), sel.transform(X_test)
elif fs_method == "SelectKBest - Mutual Information":
    sel = SelectKBest(mutual_info_classif, k=fs_k); X_train, X_test = sel.fit_transform(X_train,y_train), sel.transform(X_test)
elif fs_method == "Tree-based importance":
    model_fs = RandomForestClassifier(random_state=42).fit(X_train,y_train)
    idxs = np.argsort(model_fs.feature_importances_)[-fs_k:]
    X_train, X_test = X_train[:,idxs], X_test[:,idxs]
# Reduction
if fr_method == "PCA":
    reducer = PCA(n_components=2); X_train, X_test = reducer.fit_transform(X_train), reducer.transform(X_test)
elif fr_method == "KernelPCA (RBF)":
    reducer = KernelPCA(n_components=2, kernel='rbf', gamma=0.1); X_train, X_test = reducer.fit_transform(X_train), reducer.transform(X_test)
elif fr_method == "UMAP":
    reducer = umap.UMAP(n_components=2, random_state=42); X_train, X_test = reducer.fit_transform(X_train), reducer.transform(X_test)
# Scaling
if scaler_name == "StandardScaler":
    scaler = StandardScaler(); X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
elif scaler_name == "MinMaxScaler":
    scaler = MinMaxScaler(); X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
elif scaler_name == "RobustScaler":
    scaler = RobustScaler(); X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Ensure 2D
if X_train.shape[1] != 2:
    st.error("2D data required for boundary plots."); st.stop()

# --- Models & evaluation ---
models = {
    "Logistic Regression": LogisticRegression(),
    "Linear SVM": LinearSVC(max_iter=5000),
    "Kernel SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradBoost": GradientBoostingClassifier(),
    "Bagging": BaggingClassifier(),
    "GaussNB": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "MLP": MLPClassifier(max_iter=1000),
    "SGD": SGDClassifier(max_iter=1000),
    "Passive Aggressive": SGDClassifier(max_iter=1000, loss='hinge'),
    # Anomaly
    "IsoForest": IsolationForest(random_state=42),
    "OneClassSVM": OneClassSVM(gamma='auto'),
    "LOF": LocalOutlierFactor(novelty=True)
}
results = []
for name, clf in models.items():
    est = clone(clf)
    if name == "LOF":
        est.fit(X_train); y_pred_raw = est.predict(X_test)
    else:
        fit_args = (X_train,y_train) if name not in ["IsoForest","OneClassSVM","LOF"] else (X_train,)
        est.fit(*fit_args); y_pred_raw = est.predict(X_test)
    # map anomalies
    y_pred = (y_pred_raw>0).astype(int) if name in ["IsoForest","OneClassSVM","LOF"] else y_pred_raw
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr, tnr = tp/(tp+fn), tn/(tn+fp)
    fpr, fnr = fp/(fp+tn), fn/(fn+tp)
    precision = precision_score(y_test,y_pred)
    recall, f1 = tpr, f1_score(y_test,y_pred)
    acc = accuracy_score(y_test,y_pred)
    gmean = math.sqrt(tpr*tnr)
    results.append({"Model":name,"TP":tp,"TN":tn,"FP":fp,"FN":fn,
                    "TPR":tpr,"TNR":tnr,"FPR":fpr,"FNR":fnr,
                    "Accuracy":acc,"Precision":precision,
                    "Recall":recall,"F1":f1,"G-Mean":gmean})
# Show table
st.subheader("Performance Metrics")
st.dataframe(pd.DataFrame(results), use_container_width=True)

# Plot decision boundaries
x_min,x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
y_min,y_max = X_train[:,1].min()-1, X_train[:,1].max()+1
xx,yy = np.meshgrid(np.linspace(x_min,x_max,200),np.linspace(y_min,y_max,200))
for name in models:
    exp = st.expander(f"Decision Boundary: {name}")
    with exp:
        est = clone(models[name])
        if name == "LOF": est.fit(X_train)
        else:
            fit_args = (X_train,y_train) if name not in ["IsoForest","OneClassSVM","LOF"] else (X_train,)
            est.fit(*fit_args)
        Z = est.predict(np.c_[xx.ravel(),yy.ravel()])
        if name in ["IsoForest","OneClassSVM","LOF"]: Z = (Z>0).astype(int)
        Z = Z.reshape(xx.shape)
        fig_w,fig_h = (12,8) if fullscreen else (6,4)
        plt.figure(figsize=(fig_w,fig_h))
        plt.contourf(xx,yy,Z,alpha=0.3)
        plt.scatter(X_train[:,0],X_train[:,1],c=y_train,edgecolor='k',s=20)
        plt.title(name)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        st.pyplot(plt, use_container_width=fullscreen)

