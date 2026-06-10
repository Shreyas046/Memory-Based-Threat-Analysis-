import streamlit as st
import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from mlp_model import MLP

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Malware Detection Dashboard", layout="wide")

st.title("🛡️ Malware Detection Dashboard")
st.markdown("### Memory-Based Malware Classification using ML & DL")

# ---------------- LOAD FILES ----------------
try:
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
except:
    st.error("❌ Run main.py first")
    st.stop()

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("data/malmem.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ---------------- PREPARE FEATURES ----------------
feature_cols = [c for c in df.columns if c not in ['Class','Category','Filename']]
X = df[feature_cols]
X = X.select_dtypes(include=['number'])
X_scaled = scaler.transform(X)

# ---------------- LABELS ----------------
y_true = le.transform(df['Class'].astype(str).str.strip())

# ---------------- LOAD MLP ----------------
input_size = X_scaled.shape[1]
num_classes = len(le.classes_)

mlp_model = MLP(input_size, num_classes)
mlp_model.load_state_dict(torch.load("mlp_model.pth"))
mlp_model.eval()

# ---------------- TRAIN RF (ONCE) ----------------
@st.cache_resource
def load_rf():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_scaled, y_true)
    return model

rf_model = load_rf()

# ---------------- MLP PREDICTION ----------------
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

with torch.no_grad():
    outputs = mlp_model(X_tensor)
    probs = torch.softmax(outputs, dim=1)
    mlp_conf, mlp_preds = torch.max(probs, 1)

mlp_labels = le.inverse_transform(mlp_preds.numpy())

# ---------------- RF PREDICTION ----------------
rf_preds = rf_model.predict(X_scaled)
rf_labels = le.inverse_transform(rf_preds)
rf_conf = rf_model.predict_proba(X_scaled).max(axis=1)

# ---------------- ACCURACY ----------------
mlp_acc = accuracy_score(y_true, mlp_preds.numpy())
rf_acc = accuracy_score(y_true, rf_preds)

# ---------------- SUMMARY ----------------
st.subheader("📌 Summary")

total = len(df)
benign = (mlp_labels == 'Benign').sum()
malware = total - benign

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", total)
col2.metric("Benign", benign)
col3.metric("Malware", malware)

# ---------------- ACCURACY DISPLAY ----------------
st.subheader("🎯 Model Comparison")

col1, col2 = st.columns(2)
col1.metric("MLP Accuracy", f"{mlp_acc * 100:.2f}%")
col2.metric("Random Forest Accuracy", f"{rf_acc * 100:.2f}%")

# ---------------- IMPROVED ROC CURVE ----------------
st.subheader("📊 ROC Curve Comparison")

try:
    import matplotlib.pyplot as plt

    # Probabilities
    mlp_probs = torch.softmax(outputs, dim=1).numpy()
    rf_probs = rf_model.predict_proba(X_scaled)

    mlp_scores = mlp_probs[:, 1]
    rf_scores = rf_probs[:, 1]

    # ROC
    fpr_mlp, tpr_mlp, _ = roc_curve(y_true, mlp_scores)
    fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_scores)

    auc_mlp = auc(fpr_mlp, tpr_mlp)
    auc_rf = auc(fpr_rf, tpr_rf)

    # -------- STYLE --------
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(5, 3))  # smaller size

    ax.plot(fpr_mlp, tpr_mlp, label=f"MLP (AUC = {auc_mlp:.3f})", linewidth=2)

    ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {auc_rf:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    ax.set_xlabel("FPR")

    ax.set_ylabel("TPR")

    ax.set_title("ROC Curve")

    ax.grid(alpha=0.3)

    ax.legend(loc="lower right")

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    st.pyplot(fig, use_container_width=False)

    
except Exception as e:
    st.warning(f"ROC error: {e}")

# ---------------- CHART ----------------
st.subheader("📈 Prediction Distribution (MLP)")
st.bar_chart(pd.Series(mlp_labels).value_counts())

# ---------------- RESULTS ----------------
st.subheader("🔍 Sample Predictions")

df_result = df.copy()
df_result['MLP_Prediction'] = mlp_labels
df_result['MLP_Confidence'] = mlp_conf.numpy()

df_result['RF_Prediction'] = rf_labels
df_result['RF_Confidence'] = rf_conf

st.dataframe(df_result[['MLP_Prediction','MLP_Confidence','RF_Prediction','RF_Confidence']].head(20))

# ---------------- DOWNLOAD ----------------
csv = df_result.to_csv(index=False).encode('utf-8')

st.download_button(
    "⬇️ Download Full Results",
    csv,
    "malware_results.csv",
    "text/csv"
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed using Memory-Based Malware Detection (CIC-MalMem-2022)")