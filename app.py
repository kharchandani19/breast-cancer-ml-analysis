import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, roc_curve, auc

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@st.cache_data
def get_data():
    data = load_breast_cancer()
    means = data.data.mean(axis=0)
    return means, data

mean_vals, bc_data = get_data()
feature_names = list(bc_data.feature_names)

st.sidebar.markdown("---")
with st.sidebar.expander("About this app"):
    st.markdown("""
    **Breast Cancer Predictor**  
    - Dataset: UCI Wisconsin Breast Cancer (569 samples, 30 features)  
    - Model: Logistic Regression  
    - Accuracy: 97.4%  
    - Author: Kanishka Harchandani
    """)

st.sidebar.header("Input Features")
input_data = []
for i, feature in enumerate(feature_names):
    val = st.sidebar.number_input(feature, value=float(np.round(mean_vals[i], 3)), format="%.3f", key=f"input_{i}")
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)

tab1, tab2 = st.tabs(["Breast Cancer Prediction", "Model Performance"])

with tab1:
    st.title("Breast Cancer Prediction App")
    st.write("Enter feature values in the sidebar and click Predict.")
    if st.button("Predict"):
        prediction = model.predict(scaled_input)[0]
        proba_malignant = model.predict_proba(scaled_input)[0][0]
        if prediction == 0:
            st.error("Result: Malignant tumor detected")
        else:
            st.success("Result: Benign tumor detected")
        st.write(f"**Malignancy Probability: {proba_malignant:.1%}**")
        st.progress(float(proba_malignant))
        st.subheader("Top 10 Most Influential Features")
        coef = np.abs(model.coef_).flatten()
        top_indices = np.argsort(coef)[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_values = coef[top_indices]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_values[::-1], y=top_features[::-1], orient="h", ax=ax, color="steelblue")
        ax.set_xlabel("Coefficient Magnitude")
        ax.set_title("Top 10 Feature Importances")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab2:
    st.title("Model Performance")
    X_all = scaler.transform(bc_data.data)
    y_true = bc_data.target
    y_pred = model.predict(X_all)
    y_score = model.predict_proba(X_all)[:, 0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "97.4%")
    col2.metric("Dataset", "569 samples")
    col3.metric("Features", "30")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, xticklabels=["Malignant","Benign"], yticklabels=["Malignant","Benign"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    plt.close()
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=0)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
    plt.close()
