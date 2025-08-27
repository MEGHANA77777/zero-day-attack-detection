import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import shap
import matplotlib.pyplot as plt

# ---------- Custom CSS for enhanced UI ----------
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main {
        background-color: #f5f7fa;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #f5f7fa;
    }
    h1, h2, h3, h4 {
        color: #0072ff;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .stDataFrame {
        background-color: #eaf6fb;
        border-radius: 8px;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.markdown(
    "<h2 style='color:#0072ff;'>🛡️ Cyber Zero-Day Detector</h2>", unsafe_allow_html=True
)
st.sidebar.write("## Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
n_samples = st.sidebar.slider("Number of samples", 1000, 10000, 5000, step=500)
show_xai = st.sidebar.checkbox("Show XAI (SHAP) explanations", value=True)

# ---------- Main Header ----------
st.markdown(
    "<h1 style='text-align:center; color:#0072ff;'>🚨 Zero-Day Attack Detection using Deep Learning</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#2e3b4e; font-size:1.2em;'>"
    "Train and explain a deep learning model for zero-day attack detection.<br>"
    "Upload your own network traffic data or use synthetic samples."
    "</p>", unsafe_allow_html=True
)

# ---------- Data Generation ----------
def generate_data(n_samples):
    data = {
        'packet_size': np.random.randint(64, 1500, n_samples),
        'duration': np.random.rand(n_samples) * 10,
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.6, 0.3, 0.1]),
        'src_port': np.random.randint(1, 65535, n_samples),
        'dst_port': np.random.randint(1, 65535, n_samples),
        'num_packets': np.random.randint(1, 100, n_samples),
        'num_bytes': np.random.randint(100, 150000, n_samples),
        'label': np.random.choice(['normal', 'attack', 'zero_day'], n_samples, p=[0.7, 0.2, 0.1])
    }
    df = pd.DataFrame(data)
    # Add attack/zero-day characteristics
    df.loc[df['label'] == 'attack', 'packet_size'] = np.random.randint(1000, 1500, df[df['label'] == 'attack'].shape[0])
    df.loc[df['label'] == 'attack', 'duration'] = np.random.rand(df[df['label'] == 'attack'].shape[0]) * 5 + 10
    df.loc[df['label'] == 'zero_day', 'packet_size'] = np.random.randint(1200, 2000, df[df['label'] == 'zero_day'].shape[0])
    df.loc[df['label'] == 'zero_day', 'duration'] = np.random.rand(df[df['label'] == 'zero_day'].shape[0]) * 7 + 15
    return df

def load_user_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Expect columns: packet_size, duration, protocol, src_port, dst_port, num_packets, num_bytes, label
    # If protocol is not categorical, convert if needed
    if 'protocol' in df.columns and not pd.api.types.is_object_dtype(df['protocol']):
        df['protocol'] = df['protocol'].astype(str)
    return df

# ---------- Data Selection ----------
if uploaded_file is not None:
    df = load_user_data(uploaded_file)
    st.success("✅ Custom dataset loaded!")
else:
    df = generate_data(n_samples)
    st.info("ℹ️ Using synthetic data. Upload a CSV to use your own dataset.")

# ---------- Show Sample Data ----------
st.markdown("### 📊 Sample Data")
st.dataframe(df.head(), use_container_width=True)

# ---------- Preprocessing ----------
df = pd.get_dummies(df, columns=['protocol'], drop_first=True)
X = df.drop('label', axis=1)
y = df['label']
label_mapping = {'normal': 0, 'attack': 1, 'zero_day': 2}
y_encoded = y.map(label_mapping)
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

numerical_features = ['packet_size', 'duration', 'src_port', 'dst_port', 'num_packets', 'num_bytes']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ---------- Model ----------
input_shape = (X_train.shape[1],)
model = Sequential([
    Input(shape=input_shape),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(label_mapping), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ---------- Training ----------
with st.spinner("🔄 Training model..."):
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)
st.success("✅ Model trained.")

# ---------- Results Section ----------
st.markdown("---")
st.markdown("## 🧪 Model Results")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📝 Classification Report")
    st.text(classification_report(y_test, y_pred := np.argmax(model.predict(X_test), axis=1), target_names=[inverse_label_mapping[i] for i in sorted(inverse_label_mapping.keys())]))
with col2:
    st.markdown("#### 🔢 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

# ---------- SHAP Explanations ----------
if show_xai:
    st.markdown("---")
    st.markdown("<h2 style='color:#00c6ff;'>🧠 SHAP Explanations</h2>", unsafe_allow_html=True)
    X_train_numeric = X_train.astype(np.float32)
    X_test_numeric = X_test.astype(np.float32)
    background_data = X_train_numeric.iloc[np.random.choice(X_train_numeric.shape[0], 100, replace=False)]
    explainer = shap.Explainer(model, background_data)
    num_instances_to_explain = min(len(X_test_numeric), 10)
    shap_values = explainer(X_test_numeric.iloc[:num_instances_to_explain])

    st.markdown("#### 💧 SHAP Waterfall Plot (first 3 test instances)")
    for i in range(min(3, num_instances_to_explain)):
        predicted_class_index = np.argmax(model.predict(X_test_numeric.iloc[i:i+1], verbose=0), axis=1)[0]
        predicted_label = inverse_label_mapping.get(predicted_class_index, 'Unknown')
        explanation = shap.Explanation(
            values=shap_values[i, :, predicted_class_index],
            base_values=shap_values.base_values[i][predicted_class_index] if hasattr(shap_values.base_values, '__getitem__') and len(shap_values.base_values.shape) > 1 else shap_values.base_values[i],
            data=X_test_numeric.iloc[i].values,
            feature_names=X_test_numeric.columns.tolist()
        )
        st.write(f"**Instance {i} (Predicted: {predicted_label})**")
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)
        st.info(
            "Interpretation: The SHAP waterfall plot above shows how each feature contributed to the model's prediction for this instance. "
            "Features pushing the prediction towards attack or zero-day are shown in red, while those pushing towards normal are in blue. "
            "The base value is the average model output, and the sum of feature effects leads to the final prediction probability."
        )

    st.markdown("#### 🌈 SHAP Summary Plot (Attack class)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[:, :, 1], X_test_numeric.iloc[:num_instances_to_explain], show=False)
    st.pyplot(fig)
    st.info(
        "Interpretation: The SHAP summary plot above ranks features by their overall impact on the model's predictions for the 'attack' class. "
        "Features at the top are most influential. The color shows feature value (red = high, blue = low) and their effect on predicting attacks. "
        "This helps identify which network characteristics are most important for detecting attacks."
    )

st.sidebar.caption("© 2025 <span style='color:#0072ff;'>Zero-Day Attack Detection Demo</span> | Deep Learning + SHAP", unsafe_allow_html=True)