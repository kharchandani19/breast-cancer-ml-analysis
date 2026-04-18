import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load breast cancer dataset for feature names and means
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Default values (just put 0.0 or some constant)
mean_vals = [0.0] * len(feature_names)

st.title("Breast Cancer Prediction App")
st.write(
    "Enter the values for each feature in the sidebar and click Predict to see the likelihood of malignancy. (0 = malignant, 1 = benign)"
)

# Sidebar inputs
st.sidebar.header("Input Features")

input_data = []
for i, feature in enumerate(feature_names):
    val = st.sidebar.number_input(
        feature, 
        value=float(np.round(mean_vals[i], 3)),
        format="%.3f"
    )
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)

if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][0]  # Probability for class 0 (malignant)
    if prediction == 0:
        st.error(
            f"Prediction: Malignant\nProbability (malignant): {proba:.3f}"
        )
    else:
        st.success(
            f"Prediction: Benign\nProbability (malignant): {proba:.3f}"
        )
        # 1. Use real dataset means as default input values
        from sklearn.datasets import load_breast_cancer
        import matplotlib.pyplot as plt
        import streamlit as st
        import numpy as np
        import seaborn as sns

        # Only do this once (prevent re-loading);
        @st.cache_data
        def get_means_and_labels():
            data = load_breast_cancer()
            means = data.data.mean(axis=0)
            labels = data.target
            features = data.feature_names
            return means, labels, data

        mean_vals, y, bc_data = get_means_and_labels()

        # Add About section in the sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("About")
        st.sidebar.info(
            """
            **Breast Cancer Predictor**
            - Predicts malignancy/benignancy based on input features.
            - Built using a machine learning model trained on the [UCI Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset).
            - Author: Kanishka Harchandani
            """
        )

        # Tabs for main content
        tabs = st.tabs(["Prediction", "Model Performance"])

        with tabs[0]:
            # Sidebar inputs for features with real means
            st.subheader("Input Features")
            input_data = []
            for i, feature in enumerate(feature_names):
                val = st.sidebar.number_input(
                    feature, 
                    value=float(np.round(mean_vals[i], 3)),
                    format="%.3f",
                    key=f"input_{i}"
                )
                input_data.append(val)

            input_array = np.array(input_data).reshape(1, -1)
            scaled_input = scaler.transform(input_array)

            if st.button("Predict"):
                prediction = model.predict(scaled_input)[0]
                proba = model.predict_proba(scaled_input)[0][0]  # Probability for class 0 (malignant)
                if prediction == 0:
                    st.error(
                        f"Prediction: Malignant\nProbability (malignant): {proba:.3f}"
                    )
                else:
                    st.success(
                        f"Prediction: Benign\nProbability (malignant): {proba:.3f}"
                    )
                # Show progress bar for probability (malignant)
                st.write("Malignancy Probability:")
                st.progress(proba)

                # Feature importance chart
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    # Indices of top 10 features
                    top_indices = np.argsort(importances)[::-1][:10]
                    plt.figure(figsize=(7,4))
                    sns.barplot(
                        x=importances[top_indices],
                        y=[feature_names[i] for i in top_indices],
                        orient="h"
                    )
                    plt.title("Top 10 Feature Importances")
                    plt.xlabel("Importance")
                    plt.ylabel("Feature")
                    st.pyplot(plt)
                else:
                    st.info("Model does not provide feature importances.")

        with tabs[1]:
            st.subheader("Model Performance")

            # Calculate and display confusion matrix and ROC curve
            from sklearn.metrics import confusion_matrix, roc_curve, auc

            y_true = bc_data.target
            X = scaler.transform(bc_data.data)
            y_pred = model.predict(X)
            y_score = model.predict_proba(X)[:, 0]  # prob of class 0 (malignant)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', ax=ax_cm)
            ax_cm.set_xlabel("Predicted Labels")
            ax_cm.set_ylabel("True Labels")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=0)  # 0 = malignant
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
            ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)