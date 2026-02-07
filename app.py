import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌿 Black Gram Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #E6F2E6;
    color: #1B3B17;
}
.stButton>button {
    background-color: #2E7D32;
    color: white;
    font-weight: bold;
}
.stFileUploader>div>input {
    border: 2px solid #2E7D32;
    border-radius: 5px;
}
.conf-bar {
    height: 20px;
    border-radius: 10px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DISEASE CLASSES ----------------
CLASS_NAMES = ['Anthracnose', 'Healthy', 'Leaf Crinkle', 'Powdery Mildew', 'Yellow Mosaic']

# ---------------- REMEDIES ----------------
REMEDIES = {
    "Anthracnose": "Remove affected leaves, apply fungicide, and maintain proper spacing.",
    "Healthy": "The plant looks healthy. Continue proper watering and nutrient management.",
    "Leaf Crinkle": "Prune damaged leaves and monitor for insect infestation.",
    "Powdery Mildew": "Apply neem oil or fungicide, improve airflow around plants.",
    "Yellow Mosaic": "Remove infected plants and control whitefly population."
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights="DEFAULT")
    for p in model.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load("blackgram_multidisease.pth")["model_state"], strict=False)
    model.eval()
    model.to(device)
    return model, device

model, device = load_model()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 Black Gram Disease Detector")
st.sidebar.write("Upload one or multiple leaf images to predict diseases.")
mode = st.sidebar.radio("Upload Mode:", ["Single Image", "Multiple Images"])

uploaded_files = None
if mode == "Single Image":
    uploaded_files = [st.file_uploader("Choose a leaf image", type=["jpg","jpeg","png"])]
else:
    uploaded_files = st.file_uploader("Choose leaf images", type=["jpg","jpeg","png"], accept_multiple_files=True)

results = []

# ---------------- PROCESS AND DISPLAY ----------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", width=700)  # set width in pixels


        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probabilities)
        disease = CLASS_NAMES[top_idx]
        confidence = probabilities[top_idx]*100

        # Store result for CSV
        results.append({
            "Image": uploaded_file.name,
            "Predicted Disease": disease,
            "Confidence (%)": f"{confidence:.2f}"
        })

        # ---------------- RESULT CARD ----------------
        st.markdown("---")
        st.markdown(f"""
        <div style="padding:15px; border-radius:12px; background: linear-gradient(to right, #C8E6C9, #A5D6A7);
                    border:2px solid #2E7D32; box-shadow: 3px 3px 8px rgba(0,0,0,0.1)">
            <h3 style="color:#1B5E20">🦠 Prediction: {disease}</h3>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- CONFIDENCE BARS ----------------
        st.write("Confidence Breakdown:")
        for i, cls in enumerate(CLASS_NAMES):
            bar_color = "#2E7D32" if i==top_idx else "#81C784"
            st.markdown(f"""
            <div style="background-color:#E0E0E0; border-radius:10px; width:100%; margin-bottom:5px;">
                <div class='conf-bar' style='width:{probabilities[i]*100:.1f}%; background-color:{bar_color}'></div>
            </div>
            <p>{cls}: {probabilities[i]*100:.2f}%</p>
            """, unsafe_allow_html=True)

        # ---------------- REMEDY CARD ----------------
        st.subheader("🌱 Suggested Remedy")
        st.markdown(f"""
<div style='padding:15px; border-radius:12px; 
            background-color:#D0F0C0;  /* slightly darker green */
            border:2px solid #2E7D32; 
            color:#1B3B17; /* dark green text */
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1)'>
    <p>{REMEDIES[disease]}</p>
</div>
""", unsafe_allow_html=True)


# ---------------- DOWNLOAD CSV ----------------
if results:
    df = pd.DataFrame(results)
    st.markdown("---")
    st.subheader("📄 Download Report")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='leaf_disease_report.csv',
        mime='text/csv'
    )

st.markdown("---")
st.caption("Note: Prototype AI system. Accuracy improves with more images and environmental data.")
