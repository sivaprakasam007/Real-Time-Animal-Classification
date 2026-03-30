import streamlit as st
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import timm
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import os
import glob
import cv2
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Configuration ---
MODEL_NAME = 'convnextv2_tiny'
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'raw-img'

# Italian to English Mapping (for display)
LABEL_MAPPING = {
    'cane': 'Dog', 
    'cavallo': 'Horse', 
    'elefante': 'Elephant', 
    'farfalla': 'Butterfly', 
    'gallina': 'Chicken', 
    'gatto': 'Cat', 
    'mucca': 'Cow', 
    'pecora': 'Sheep', 
    'ragno': 'Spider', 
    'scoiattolo': 'Squirrel'
}

# --- Page Setup ---
st.set_page_config(
    page_title="Animal classification Command Center",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode & Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.title("🦁 Animal classification Command Center")
st.markdown("### Powered by ConvNeXt V2 | Advanced Species Identification System")

# --- Classes & Helpers ---

class TransformedSubset(Dataset):
    """
    Wrapper to apply specific transforms to a Subset of a Dataset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# --- Model Loading ---
@st.cache_resource
def load_zoo_model():
    """
    Loads the trained ConvNeXt V2 model and class names.
    """
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    
    try:
        checkpoint = torch.load('zoo_bundle.pth', map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        class_names = checkpoint['class_names']
    except FileNotFoundError:
        st.error("⚠️ Model file 'zoo_bundle.pth' not found. Please run 'train_zoo.py' first.")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None

    model.to(DEVICE)
    model.eval()
    return model, class_names

@st.cache_resource
def load_yolo_model():
    """
    Loads YOLO model for animal counting.
    """
    return YOLO("yolov8n.pt")

model, class_names = load_zoo_model()
yolo_model = load_yolo_model()

# --- Data Loading for Diagnostics ---
@st.cache_resource
def load_test_data():
    """
    Re-creates the Test Split exactly as done in training.
    """
    if not os.path.exists(DATA_DIR):
        return None, None

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    base_dataset = datasets.ImageFolder(root=DATA_DIR)
    
    total_len = len(base_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    _, _, test_subset = random_split(
        base_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = TransformedSubset(test_subset, eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    return test_loader, base_dataset.classes

# --- Animal Counting Function ---
def count_animals_in_image(image_pil):
    """
    Counts detectable animals in the uploaded image using YOLO.
    """
    img_np = np.array(image_pil)

    results = yolo_model(img_np)

    valid_animals = [
        'dog', 'cat', 'horse', 'cow', 'sheep', 'elephant', 'bird'
    ]

    total_count = 0
    detected_animals = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = yolo_model.names[cls_id]

            if class_name.lower() in valid_animals:
                total_count += 1
                detected_animals.append(class_name.capitalize())

    return total_count, detected_animals

# --- Application Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🦁 Identification", 
    "📊 Intelligence", 
    "📂 The Archive", 
    "🧠 Neural Architecture"
])

# ==========================================
# TAB 1: Identification (The Scanner)
# ==========================================
with tab1:
    st.header("Scanning Interface")
    
    col_upload, col_result = st.columns([1, 2])
    
    with col_upload:
        uploaded_file = st.file_uploader("Upload Image to Scan", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert('RGB')
            st.image(image_pil, caption="Source Image", width="stretch")
            
    with col_result:
        if uploaded_file is not None and model is not None:
            # Preprocess
            infer_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = infer_transform(image_pil).unsqueeze(0).to(DEVICE)
            img_np = np.array(image_pil.resize((224, 224), resample=Image.BILINEAR)) / 255.0
            
            # Predict with ConvNeXt
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)

            # Count animals with YOLO
            animal_count, detected_animals = count_animals_in_image(image_pil)
                
            pred_class = class_names[pred_idx.item()]
            confidence = conf.item() * 100
            
            # Display Result
            st.subheader(f"Identification: **{pred_class}**")
            
            # Confidence Bar
            st.progress(int(confidence))
            st.caption(f"Confidence Level: {confidence:.2f}%")
            
            # Hybrid Warning
            if confidence < 50:
                st.warning("⚠️ **Low Confidence Detected**: The system is unsure. This might be a hybrid species or an unknown entity.")
            else:
                st.success("✅ **Positive Identification**")

            # Animal Count Display
            st.subheader("Animal Count")

            if animal_count > 0:
                st.info(f"🐾 Number of animals detected: **{animal_count}**")
                st.write("Detected species by YOLO:", ", ".join(detected_animals))
            else:
                st.warning("No detectable animals found in the image.")
            
            # Grad-CAM X-Ray
            with st.expander("Activate X-Ray Vision (Grad-CAM)"):
                try:
                    target_layers = [model.stages[-1]]
                    cam = GradCAM(model=model, target_layers=target_layers)
                    
                    targets = [ClassifierOutputTarget(pred_idx.item())]
                    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    
                    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                    st.image(visualization, caption=f"Attention Map for {pred_class}", width="stretch")
                    st.info("The heatmap shows where the Neural Network is 'looking' to make its decision.")
                except Exception as e:
                    st.error(f"X-Ray module unavailable: {e}")

# ==========================================
# TAB 2: Intelligence (Performance)
# ==========================================
with tab2:
    st.header("System Diagnostics & Performance Metrics")
    
    if st.button("🚀 Run Full System Diagnostics"):
        test_loader, _ = load_test_data()
        
        if test_loader is None:
            st.error(f"Dataset not found at {DATA_DIR}. Cannot run diagnostics.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_preds = []
            all_labels = []
            
            total_batches = len(test_loader)
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    images = images.to(DEVICE)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    
                    progress = (i + 1) / total_batches
                    progress_bar.progress(progress)
                    status_text.text(f"Processing Batch {i+1}/{total_batches}...")
            
            status_text.text("Diagnostics Complete.")
            
            y_true = [class_names[i] for i in all_labels]
            y_pred = [class_names[i] for i in all_preds]
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=class_names)
            
            fig = px.imshow(
                cm, 
                x=class_names, 
                y=class_names,
                text_auto=True,
                color_continuous_scale='Viridis',
                labels=dict(x="Predicted", y="True", color="Count"),
                title="Confusion Matrix Heatmap"
            )
            fig.update_layout(width=700, height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification Report
            st.subheader("Detailed Classification Report")
            report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose()
            
            st.dataframe(
                report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                               .format("{:.2%}", subset=['precision', 'recall', 'f1-score'])
            )

# ==========================================
# TAB 3: The Archive (EDA)
# ==========================================
with tab3:
    st.header("Data Archive Analysis")
    
    if os.path.exists(DATA_DIR):
        class_counts = {}
        example_images = {}
        
        folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        
        for folder in folders:
            eng_name = LABEL_MAPPING.get(folder, folder)
            files = glob.glob(os.path.join(DATA_DIR, folder, '*.*'))
            class_counts[eng_name] = len(files)
            
            if files:
                example_images[eng_name] = np.random.choice(files)
        
        # Distribution Chart
        st.subheader("Dataset Balance")
        df_counts = pd.DataFrame(list(class_counts.items()), columns=['Species', 'Count'])
        
        fig_bar = px.bar(
            df_counts, 
            x='Species', 
            y='Count', 
            color='Species',
            title="Distribution of Specimens in Archive"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Gallery
        st.subheader("Specimen Gallery")
        cols = st.columns(5)
        
        for idx, (species, img_path) in enumerate(example_images.items()):
            col = cols[idx % 5]
            with col:
                try:
                    img = Image.open(img_path)
                    st.image(img, caption=species, width="stretch")
                except Exception:
                    st.error(f"Error loading {species}")
                    
    else:
        st.error(f"Archive directory '{DATA_DIR}' not found.")

# ==========================================
# TAB 4: Neural Architecture
# ==========================================
with tab4:
    st.header("Neural Architecture: ConvNeXt V2")
    
    st.markdown("""
    ### Modernizing the ConvNet
    **ConvNeXt V2** represents the evolution of Convolutional Neural Networks (ConvNets) in the era of Transformers. While Vision Transformers (ViT) gained popularity for their global attention mechanisms, they are often computationally heavy and harder to train on smaller datasets. ConvNeXt brings the "pure ConvNet" back to state-of-the-art performance by adopting design choices from Transformers.
    
    #### Key Innovations:
    1.  **Depthwise Convolutions**:
        * Similar to the self-attention mechanism in Transformers, depthwise convolutions operate on each channel separately.
        * We use a kernel size of `7x7`, allowing the network to "see" a wider context.
    
    2.  **Layer Normalization**:
        * Replaces the traditional Batch Normalization for more stable training.
    
    3.  **Inverted Bottlenecks**:
        * Expands channel dimension in hidden layers and compresses it back for better feature learning.
    
    4.  **Global Response Normalization (GRN)**:
        * Unique to V2, GRN prevents feature collapse and helps the network learn more diverse features.
        
    #### Why ConvNeXt V2 vs. ViT?
    * **Efficiency**: Faster inference than ViT on standard hardware.
    * **Inductive Bias**: ConvNets naturally understand locality and translation invariance, making them better for medium-sized datasets like Animal-10.
    """)
    
    st.info(f"Current Model: **{MODEL_NAME}** running on **{DEVICE}**")
