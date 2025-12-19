import numpy as np
import streamlit as st
import pickle

# -------------------- Page config --------------------
st.set_page_config(
    page_title="Iris SVM Classifier",
    page_icon="🌸",
    layout="centered"
)

# -------------------- Custom CSS --------------------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        }
        .title {
            font-size: 40px;
            font-weight: 800;
            color: #3f3d56;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 16px;
            color: #6c6c80;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff8a00, #e52e71);
            color: white;
            border-radius: 999px;
            padding: 0.6rem 2.5rem;
            border: 0px;
            font-weight: 600;
        }
        .stButton>button:hover {
            opacity: 0.9;
        }
        .result-box {
            padding: 1rem 1.5rem;
            border-radius: 0.8rem;
            background-color: #ffffffdd;
            border: 1px solid #e0e0f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- Title --------------------
st.markdown('<div class="title">Iris Flower Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Support Vector Machine (SVM) model trained on the classic Iris dataset</div>',
    unsafe_allow_html=True
)

# -------------------- Load trained model --------------------
@st.cache_resource
def load_model():
    with open("svm_iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------- Sidebar inputs --------------------
st.sidebar.title("Input Features")
st.sidebar.write("Adjust the sliders to set flower measurements (in cm).")

sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 5.8, 0.1)
sepal_width  = st.sidebar.slider("Sepal width",  2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 4.3, 0.1)
petal_width  = st.sidebar.slider("Petal width",  0.1, 2.5, 1.3, 0.1)

# Show current values in main area
st.markdown("### Your Input")
col1, col2 = st.columns(2)
with col1:
    st.metric("Sepal length (cm)", f"{sepal_length:.1f}")
    st.metric("Petal length (cm)", f"{petal_length:.1f}")
with col2:
    st.metric("Sepal width (cm)", f"{sepal_width:.1f}")
    st.metric("Petal width (cm)", f"{petal_width:.1f}")

# -------------------- Prediction --------------------
user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("🔍 Predict Species"):
    prediction = model.predict(user_data)[0]
    
    species_pretty = {
        "setosa": "Iris Setosa 🌿",
        "versicolor": "Iris Versicolor 🌺",
        "virginica": "Iris Virginica 🌼"
    }.get(prediction, prediction)

    st.markdown("### Prediction")
    st.markdown(
        f'<div class="result-box"><b>Predicted Species:</b> {species_pretty}</div>',
        unsafe_allow_html=True
    )
else:
    st.info("Click **Predict Species** to see the model’s output.")
