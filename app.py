import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import tempfile
import timm
from recommendation import cnv, dme, drusen, normal

# Page config
st.set_page_config(
    page_title="OCT Retinal Analysis Platform",
    page_icon="👁️",
    layout="wide"
)

# Global variables for model and transforms
model = None
transform = None
device = None
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

@st.cache_resource
def load_model_and_transforms():
    """Load PyTorch model and preprocessing transforms"""
    global model, transform, device
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load("eye_disease_model_pytorch.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Image preprocessing (ImageNet normalization for MobileNetV3)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    st.success(f"✅ Model loaded on {device}")
    return model, transform, device

# PyTorch Model Prediction
def model_prediction(test_image_path):
    """PyTorch model prediction function"""
    global model, transform, device
    
    # Load and preprocess image
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    return predicted.item(), confidence.item(), probabilities.cpu().numpy()

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

# Load model on first run
if model is None:
    load_model_and_transforms()

#Main Page
if app_mode == "Home":
    st.markdown("""
    ## **OCT Retinal Analysis Platform**

    #### **Welcome to the Retinal OCT Analysis Platform**

    **Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

    ##### **Why OCT Matters**
    OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

    ---

    #### **Key Features of the Platform**

    - **Automated Image Analysis**: Our platform uses state-of-the-art **PyTorch-based** machine learning models (MobileNetV3 Large) to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
    - **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
    - **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.
    - **Production Ready**: Built with PyTorch for better compatibility and deployment stability.

    ---

    #### **Understanding Retinal Diseases through OCT**

    1. **Choroidal Neovascularization (CNV)**
       - Neovascular membrane with subretinal fluid
       
    2. **Diabetic Macular Edema (DME)**
       - Retinal thickening with intraretinal fluid
       
    3. **Drusen (Early AMD)**
       - Presence of multiple drusen deposits

    4. **Normal Retina**
       - Preserved foveal contour, absence of fluid or edema

    ---

    #### **About the Dataset**

    Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
    - **Normal**
    - **CNV**
    - **DME**
    - **Drusen**

    Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification.

    ---

    #### **Get Started**

    - **Upload OCT Images**: Begin by uploading your OCT scans for analysis.
    - **Explore Results**: View categorized scans and detailed diagnostic insights.
    - **Learn More**: Dive deeper into the different retinal diseases and how OCT helps diagnose them.

    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. 
                Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.
                (A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). 
                (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). 
                (Middle right) Multiple drusen (arrowheads) present in early AMD. 
                (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

                ---

                #### Content
                The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). 
                There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

                **Model Architecture**: MobileNetV3 Large (PyTorch implementation via TIMM library)
                **Pretrained**: ImageNet weights fine-tuned on OCT dataset
                **Input Size**: 224x224 RGB images
                **Output**: 4-class classification (CNV, DME, DRUSEN, NORMAL)

                ---

                #### Technical Stack
                - **Framework**: PyTorch 2.0+
                - **Model Library**: TIMM (Torch Image Models)
                - **Deployment**: Streamlit
                - **Image Processing**: torchvision transforms
                """)

# Prediction Page - FIXED VERSION
elif app_mode == "Disease Identification":
    st.header("🔬 Disease Identification")
    st.markdown("Upload your OCT retinal scan for automated analysis")
    
    # Check if model is loaded
    if model is None:
        st.error("❌ Model not loaded. Please refresh the page or check model file.")
        st.info("📁 Required file: `eye_disease_model_pytorch.pth`")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            test_image = st.file_uploader("Choose OCT image...", 
                                        type=['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'],
                                        help="Upload a retinal OCT scan")
            
            if test_image is not None:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(test_image.getvalue())
                    temp_file_path = tmp_file.name
                
                # Display uploaded image
                image = Image.open(temp_file_path)
                st.image(image, caption="Uploaded OCT Scan", use_column_width=True)
                st.session_state.temp_file_path = temp_file_path  # Store path
        
        with col2:
            if 'temp_file_path' in st.session_state and st.button("🚀 Predict Disease", type="primary"):
                temp_file_path = st.session_state.temp_file_path
                
                with st.spinner("🔬 Analyzing retinal scan..."):
                    result_index, confidence, probabilities = model_prediction(temp_file_path)
                    
                    if result_index is not None:
                        # Display results
                        st.success(f"**Predicted Disease:** {class_names[result_index].upper()}")
                        st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Progress bars for all classes
                        st.subheader("📊 Prediction Probabilities")
                        for i, class_name in enumerate(class_names):
                            col_prob1, col_prob2 = st.columns([3, 1])
                            with col_prob1:
                                progress_bar = st.progress(float(probabilities[i]))  # Convert to Python float
                            with col_prob2:
                                st.metric(class_name, f"{probabilities[i]:.1%}")
                        
                        # Recommendations
                        with st.expander("📖 Learn More About This Condition", expanded=True):
                            st.image(test_image, caption="Your OCT Scan", use_column_width=True)
                            
                            if result_index == 0:
                                st.markdown("### Choroidal Neovascularization (CNV)")
                                st.markdown(cnv)
                            elif result_index == 1:
                                st.markdown("### Diabetic Macular Edema (DME)")
                                st.markdown(dme)
                            elif result_index == 2:
                                st.markdown("### Drusen (Early AMD)")
                                st.markdown(drusen)
                            elif result_index == 3:
                                st.markdown("### Normal Retina")
                                st.markdown(normal)
                    else:
                        st.error("Prediction failed. Please try again.")
            else:
                st.info("👈 Please upload an OCT image on the left to get started!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #808080; padding: 2rem 0;'>
        <h3 style='color: #ff4757;'>Built with ❤️ using PyTorch <br/> Lovey dovey yooooo</h3>
        <p>OCT Retinal Disease Classification</p>
        <small>PyTorch • TIMM • Streamlit • MobileNetV3 Large</small>
    </div>
    """, 
    unsafe_allow_html=True
)

