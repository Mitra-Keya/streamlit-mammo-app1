# ========================== #

# ----- Imports -----
import sys
import os
import types
import torch
import base64
import pickle
import tempfile
from datetime import datetime
from io import BytesIO

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
from PIL import Image
from fpdf import FPDF

import streamlit as st
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights



# ----- Device Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----- Resource Loaders -----
@st.cache_resource
def load_resources():
    with open("word_to_idx.pkl", "rb") as f:
        word_to_idx = pickle.load(f)
    with open("idx_to_word.pkl", "rb") as f:
        idx_to_word = pickle.load(f)
    embedding_matrix = torch.load("embedding_matrix.pt")
    return word_to_idx, idx_to_word, embedding_matrix

word_to_idx, idx_to_word, embedding_matrix = load_resources()


# ----- Decoder Definition -----
class MammoReportDecoder(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, dropout_rate=0.3):
        super(MammoReportDecoder, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.image_fc = nn.Linear(2048, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, captions):
        # Not used in inference
        pass


@st.cache_resource
def load_decoder():
    model = MammoReportDecoder(embedding_matrix)
    model.load_state_dict(torch.load("best_mammo_decoder.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

decoder = load_decoder()


# ----- Image Transformation -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ----- ResNet50 Feature Extractor -----
@st.cache_resource
def load_resnet():
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    return model

resnet = load_resnet()


# ----- Helper Functions -----
def tokens_to_text(token_ids, idx_to_word):
    words = []
    for idx in token_ids:
        word = idx_to_word.get(idx, "<UNK>")
        if word == "<EOS>":
            break
        if word not in ("<PAD>", "<SOS>"):
            words.append(word)
    return " ".join(words)


def generate_report_with_confidence(image_tensor, decoder, max_len=50):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        features = resnet(image_tensor).squeeze(-1).squeeze(-1)

        input_seq = torch.tensor([[word_to_idx["<SOS>"]]], device=device)
        decoded_indices = []
        confidences = []

        h = torch.tanh(decoder.image_fc(features)).unsqueeze(0)
        c = torch.zeros_like(h)
        hidden = (h, c)

        for _ in range(max_len):
            embed = decoder.embedding(input_seq)
            lstm_out, hidden = decoder.lstm(embed, hidden)
            output = decoder.fc_out(decoder.dropout(lstm_out))
            output = output[:, -1, :]

            probs = torch.softmax(output, dim=-1)
            confidence, predicted_idx = probs.max(dim=-1)
            predicted_id = predicted_idx.item()

            if predicted_id == word_to_idx["<EOS>"]:
                break

            decoded_indices.append(predicted_id)
            confidences.append(confidence.item())
            input_seq = predicted_idx.unsqueeze(0)

        report_text = tokens_to_text(decoded_indices, idx_to_word)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        return report_text, avg_confidence, confidences


def save_report_as_pdf(report_text, image_pil, dicom_metadata=None, image_name="Uploaded Image", gradcam_img=None):
    pdf = FPDF()
    pdf.add_page()

    # Draw border (A4 page is 210x297 mm; margins are 10 mm by default)
    pdf.set_draw_color(0, 0, 0)  # black
    pdf.set_line_width(0.5)
    pdf.rect(x=5, y=5, w=200, h=287)  # Draw a border with a small margin


    # Title/Header
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, "Breast Imaging Mammography Report", ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, "Radiology Department - ABC Medical Center", ln=True, align="C")
    pdf.ln(5)

    # Timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, f"Report Generated: {now}", ln=True, align="R")
    pdf.ln(3)

    # Patient Info Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Use a table for patient metadata
    pdf.set_font("Arial", '', 12)
    patient_name = dicom_metadata.get("PatientName", "N/A") if dicom_metadata else "N/A"
    patient_id = dicom_metadata.get("PatientID", "AUTO12345") if dicom_metadata else "AUTO12345"
    age = dicom_metadata.get("PatientAge", "N/A") if dicom_metadata else "N/A"
    gender = dicom_metadata.get("PatientSex", "N/A") if dicom_metadata else "N/A"
    study_date = dicom_metadata.get("StudyDate", "N/A") if dicom_metadata else "N/A"
    modality = dicom_metadata.get("Modality", "N/A") if dicom_metadata else "N/A"

    info_data = [
        ("Patient Name:", patient_name),
        ("Patient ID:", patient_id),
        ("Age:", age),
        ("Gender:", gender),
        ("Study Date:", study_date),
        ("Modality:", modality),
        ("Image File:", image_name)
    ]

    col1_width = 40
    col2_width = 150
    line_height = 8
    # Arrange patient info in two columns (side by side)
    pdf.set_font("Arial", '', 12)
    num_items = len(info_data)
    half = (num_items + 1) // 2  # Split into two halves

    col1 = info_data[:half]
    col2 = info_data[half:]

    for i in range(half):
        # First column
        label1, value1 = col1[i]
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(35, line_height, label1, border=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(55, line_height, str(value1), border=0)

        # Second column (if exists)
        if i < len(col2):
            label2, value2 = col2[i]
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(35, line_height, label2, border=0)
            pdf.set_font("Arial", '', 12)
            pdf.cell(55, line_height, str(value2), border=0)

        pdf.ln(line_height)

    pdf.ln(8)

    # Insert Original + Grad-CAM side by side if gradcam_img is available
    if gradcam_img is not None:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Original and Grad-CAM Visualization", ln=True)

        # Add horizontal line under heading
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())

        pdf.ln(4)

        # Resize both images to 224x224
        gradcam_pil = Image.fromarray(gradcam_img).resize((224, 224))
        resized_image_pil = image_pil.resize((224, 224))



        # Save both images temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig_file, \
            tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_gradcam_file:

            resized_image_pil.save(tmp_orig_file.name)
            gradcam_pil.save(tmp_gradcam_file.name)

            x_orig = pdf.get_x()
            y_orig = pdf.get_y()

            # Show original image
            pdf.image(tmp_orig_file.name, x=x_orig, y=y_orig, w=80)

            # Show Grad-CAM image right next to it
            pdf.image(tmp_gradcam_file.name, x=x_orig + 85, y=y_orig, w=80)

        pdf.ln(85)  # Move down to avoid overlap
    else: 
        # Fallback: only show original image if Grad-CAM is not available
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Original Mammography Image", ln=True)
        # Add horizontal line under heading
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())

        pdf.ln(4)

        resized_img = image_pil.copy()
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.ANTIALIAS  # For Pillow < 10

        # Resize the image to 224x224
        resized_img = resized_img.resize((224, 224), resample)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img_file:
            resized_img.save(tmp_img_file.name)
            pdf.image(tmp_img_file.name, x=pdf.get_x(), w=80, h=80)  # Optional: match width & height in PDF
        pdf.ln(8)  # Adjust spacing accordingly

    # Findings Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Findings", ln=True)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Keywords to highlight
    keywords = {"mass", "lesion", "calcification", "nodule", "density", "asymmetry", "distortion",
    "architectural distortion", "spiculated", "lobulated", "cyst", "fibroadenoma",
    "microcalcification", "macrocalcification", "clustered", "cluster", "focal",
    "scar", "skin thickening", "skin retraction", "axillary lymph nodes", "malignant",
    "benign", "suspicious", "BI-RADS", "enhancement", "lesion size", "margins",
    "hypoechoic", "hyperechoic", "complex cyst", "infiltrating", "ductal", "lobular",
    "calcified ducts", "edema", "mass effect", "skin", "subcutaneous tissue",
    "Cooper’s ligaments", "fatty tissue", "glandular tissue","benign-looking","malignant-looking","BIRADS"}

    pdf.set_font("Arial", '', 12)
    sentences = report_text.split('. ')
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Safe bullet character
        pdf.cell(5, 8, '-', ln=0)  # or use chr(149)

        words = sentence.split(' ')
        for word in words:
            check_word = word.strip('.,').lower()
            if check_word in keywords:
                pdf.set_text_color(255, 0, 0)
                pdf.set_font("Arial", 'B', 12)
            else:
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Arial", '', 12)
            pdf.cell(pdf.get_string_width(word + ' '), 8, word + ' ', ln=0)

        pdf.ln(8)

    # pdf.ln(10)

    # Move cursor to fixed position near bottom of the page
    page_height = 297  # A4 page height in mm
    footer_y = 264     # Y-position for footer (adjust if needed)

    if pdf.get_y() < footer_y:
        pdf.set_y(footer_y)

    # Signature and footer
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 6, "Radiologist Signature: ______________________", ln=True)
    pdf.cell(0, 6, f"Date: {now.split()[0]}", ln=True)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name



# Add Grad-CAM class and helper
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.append(forward_handle)
        self.hook_handles.append(backward_handle)

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()

        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam = torch.relu(grad_cam)

        grad_cam = torch.nn.functional.interpolate(
            grad_cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )
        grad_cam = grad_cam.squeeze().cpu().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        return grad_cam

# Grad-CAM visualization function
def overlay_gradcam_on_image(image_pil, cam):
    image_np = np.array(image_pil.resize((224, 224))).astype(np.uint8)
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np]*3, axis=-1)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlayed = heatmap * 0.4 + image_np * 0.6
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    return overlayed



# ========================== #
# Streamlit User Interface
# ========================== #

st.title("🩺 Mammography Report Generator")

uploaded_file = st.file_uploader(
    "Upload a mammography image (DICOM or standard image formats)",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "dcm"]
)

# Detect file change to reset session state
if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None

if uploaded_file is not None and uploaded_file != st.session_state["last_uploaded_file"]:
    for key in ["PatientName", "PatientID", "PatientAge", "PatientSex", "StudyDate", "Modality"]:
        st.session_state.pop(key, None)
    st.session_state["last_uploaded_file"] = uploaded_file
    # Store an upload count or version for keys
    if "upload_version" not in st.session_state:
        st.session_state["upload_version"] = 0
    st.session_state["upload_version"] += 1


if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    dicom_metadata = {}

    def show_or_na(value):
        """Show 'N/A' in input box if value is empty or only whitespace."""
        return value.strip() if value.strip() else "N/A"

    if file_type in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        # Non-DICOM image
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("### 🧍 Enter Patient Information")
        version = st.session_state.get("upload_version", 0)

        dicom_metadata["PatientName"] = st.text_input("Patient Name", value="N/A", key=f"PatientName_{version}")
        dicom_metadata["PatientID"] = st.text_input("Patient ID", value="N/A", key=f"PatientID_{version}")
        dicom_metadata["PatientAge"] = st.text_input("Age", value="N/A", key=f"PatientAge_{version}")
        dicom_metadata["PatientSex"] = st.selectbox("Gender", options=["N/A", "Male", "Female", "Other"], index=0, key=f"PatientSex_{version}")
        dicom_metadata["StudyDate"] = st.text_input("Study Date (YYYYMMDD)", value="N/A", key=f"StudyDate_{version}")
        dicom_metadata["Modality"] = st.text_input("Modality (e.g., US, MR)", value="N/A", key=f"Modality_{version}")


    elif file_type == 'dcm':
        try:
            dicom_data = pydicom.dcmread(uploaded_file)
            image_array = dicom_data.pixel_array

            image_array = image_array.astype(np.float32)
            image_array -= np.min(image_array)
            image_array /= np.max(image_array)
            image_array *= 255.0
            image_array = image_array.astype(np.uint8)

            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)

            image = Image.fromarray(image_array)

            # Read existing metadata or set as empty string
            dicom_metadata = {
                "PatientName": str(dicom_data.get("PatientName", "")).strip(),
                "PatientID": str(dicom_data.get("PatientID", "")).strip(),
                "PatientAge": str(dicom_data.get("PatientAge", "")).strip(),
                "PatientSex": str(dicom_data.get("PatientSex", "")).strip(),
                "StudyDate": str(dicom_data.get("StudyDate", "")).strip(),
                "Modality": str(dicom_data.get("Modality", "")).strip()
            }

            st.markdown("### 🧍 Verify or Fill Missing Patient Information")
            for key, label in {
                "PatientName": "Patient Name",
                "PatientID": "Patient ID",
                "PatientAge": "Age",
                "PatientSex": "Gender",
                "StudyDate": "Study Date (YYYYMMDD)",
                "Modality": "Modality (e.g., US, MR)"
            }.items():
                current_value = show_or_na(dicom_metadata.get(key, ""))

                if key == "PatientSex":
                    gender_options = ["N/A", "Male", "Female", "Other"]
                    default_index = gender_options.index(current_value) if current_value in gender_options else 0
                    dicom_metadata[key] = st.selectbox(label, options=gender_options, index=default_index, key=key)
                else:
                    dicom_metadata[key] = st.text_input(label, value=current_value, key=key)

        except Exception as e:
            st.error(f"❌ Failed to read DICOM image: {e}")
            st.stop()

    else:
        st.error("❌ Unsupported file type!")
        st.stop()

    # Resize image for display (optional, for consistency)
    resized_image = image.resize((250, 250))  # You can adjust the size as needed

    # Convert PIL to bytes for HTML display
    
    buffer = BytesIO()
    resized_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Display image in center with caption
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,{img_str}" style="border-radius: 10px; border: 2px solid #ccc; width: 250px;" alt="Uploaded Image" />
            <p style="font-size: 14px; color: gray;">Uploaded Image</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Convert image to tensor
    image_tensor = transform(image)
    show_gradcam = st.toggle("🔍 Why this report?")


    # --- Generate Report Button Centered ---
    generate_clicked = st.button("📝 Generate Report", use_container_width=True)

    if generate_clicked:
        with st.spinner("Generating report..."):
           
            report_text, avg_confidence, word_confidences = generate_report_with_confidence(image_tensor, decoder)

        # Show report text below
        st.markdown("### 🧾 Generated Report")
        for line in report_text.split('. '):
            line = line.strip()
            if line:
                st.markdown(f"- {line.strip('.')}.")

        st.markdown(f"### 🧠 Model Confidence Score: ")
        st.write(f"{avg_confidence:.2f} (average softmax max values)")

        if show_gradcam:
            # Grad-CAM setup
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Now shape becomes [1, C, H, W]

            resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            target_layer = resnet_model.layer4[-1]
            gradcam = GradCAM(resnet_model.to(device), target_layer)

            # Generate Grad-CAM
            cam_output = gradcam(image_tensor)
            gradcam_overlay = overlay_gradcam_on_image(image, cam_output)

            pdf_path = save_report_as_pdf(report_text, image, dicom_metadata, image_name=uploaded_file.name, gradcam_img=gradcam_overlay)

            # Display
            st.markdown("### 🔥 Grad-CAM Visualization")
            st.image(gradcam_overlay, caption="Regions influencing the report",use_container_width=False, width=300)

            # Grad-CAM image
            cam_img = Image.fromarray(gradcam_overlay)
            buf = BytesIO()
            cam_img.save(buf, format="PNG")
            st.download_button("📥 Download Grad-CAM", buf.getvalue(), file_name="gradcam.png", mime="image/png")
        else:
            pdf_path = save_report_as_pdf(report_text, image, dicom_metadata, image_name=uploaded_file.name)
        
        # Allow user to download the PDF
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name="mammography_report.pdf",
                mime="application/pdf"
            )
