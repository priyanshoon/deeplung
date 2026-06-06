import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import get_model

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection AI", 
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🫁 Pneumonia Detection AI")
st.markdown("### Advanced Chest X-Ray Analysis System")
st.markdown("---")

# Determine model path automatically
default_path = "checkpoints/best_model.pth"
if not os.path.exists(default_path) and os.path.exists("checkpoints_dryrun/best_model.pth"):
    default_path = "checkpoints_dryrun/best_model.pth"

model_path = default_path

# Class names
CLASS_NAMES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    if not os.path.exists(path):
        return None, f"Model file not found at {path}"
    
    try:
        # Fix for SSL certificate error on Mac (just in case model needs to download weights again)
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        model = get_model(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Load model silently
model, error = load_model(model_path)

if error:
    st.error(f"⚠️ System Error: {error}")
    st.warning("Please ensure the model is trained and checkpoints are available.")
else:
    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload X-Ray")
        uploaded_file = st.file_uploader("Drop your image here", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Input Image', use_container_width=True)

    with col2:
        st.subheader("Analysis Results")
        if uploaded_file is not None:
            if st.button("Run Diagnostics"):
                with st.spinner("Processing image..."):
                    # Preprocess
                    device = next(model.parameters()).device
                    img_tensor = process_image(image).to(device)
                    
                    # Inference
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        top_prob, top_class = torch.max(probabilities, 1)
                        
                    # Result
                    predicted_class = CLASS_NAMES[top_class.item()]
                    confidence = top_prob.item()
                    
                    # Metrics display
                    st.metric(label="Diagnosis", value=predicted_class)
                    st.progress(confidence)
                    st.caption(f"Confidence Score: {confidence*100:.2f}%")
                    
                    st.markdown("---")
                    st.markdown("#### Detailed Probability Distribution")
                    
                    # Create a cleaner chart data structure
                    probs_dict = {name: prob.item() for name, prob in zip(CLASS_NAMES, probabilities[0])}
                    st.bar_chart(probs_dict)
        else:
            # Add vertical spacer to align with the file uploader button
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.info("Please upload an X-ray image to start the analysis.")

# Footer
st.markdown("---")
st.markdown("*Note: This tool is for educational purposes only and should not be used for medical diagnosis.*")
