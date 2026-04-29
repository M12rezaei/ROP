ROP-Screening-System: Multimodal Deep Learning for ROP Classification


 **Project Overview**
**Retinopathy of Prematurity (ROP)**is a leading cause of preventable childhood blindness. This project presents a lightweight, explainable multimodal AI system for automated ROP stage classification.

The system integrates:

Retinal fundus images
Clinical data (Gestational Age & Birth Weight)

to improve diagnostic reliability, particularly in resource-limited neonatal environments.

**Key Features**
Multimodal Learning: Combines image + clinical data for improved performance
Dual Backbone Model: ConvNeXt-Tiny + EfficientNet-B3 ensemble
Vessel-Aware Input: U-Net generated vessel mask as 4th channel
Explainable AI: Grad-CAM heatmaps for clinical interpretability
High-Sensitivity Design: Minimises false negatives (clinically critical)
Web Deployment: Streamlit-based real-time interface
**System Architecture**

**The pipeline consists of:**

Input Stage
Fundus image (RGB)
Clinical inputs (GA, BW)
Preprocessing
Ben Graham enhancement
Normalisation
Vessel segmentation (U-Net)
Feature Extraction
Dual backbone (ConvNeXt + EfficientNet)
Multimodal Fusion
Concatenation of image features + clinical features
Prediction
Ordinal classification (Normal / Mild / Severe)
Confidence calibration (temperature scaling)
Explainability
Grad-CAM heatmap generation
Output
Prediction + probabilities
Clinical recommendation
Export (PNG / PDF)

**Getting Started**
Prerequisites
Python 3.9+
PyTorch
Streamlit
(Optional) CUDA-enabled GPU for training
**Installation**
git clone https://github.com/your-username/ROP-Screening-System.git
cd ROP-Screening-System
pip install -r requirements.txt
Run the Application
streamlit run UI/app.py
** Usage**
Upload a retinal fundus image
Enter:
Gestational Age (22–40 weeks)
Birth Weight (400–2500 g)
Run analysis

**Outputs include:**

Predicted ROP stage
Confidence score
Grad-CAM heatmap
Clinical recommendation

**Ethical Compliance & Safety**
Data Privacy: Follows GDPR principles (anonymised datasets)
Clinical Safety: High-recall tuning to reduce false negatives
Human-in-the-Loop: Final decision remains with clinician
Transparency: Model Card documents bias, limitations, and performance
**Project Structure**
ROP-Screening-System/
│── scripts/              # Models and training code
│── UI/                   # Streamlit interface
│── results/              # Saved models
│── data/                 # Dataset references
│── docs/                 # Model card and documentation
│── README.md
 License

This project is licensed under the MIT License.

**Contact**

Lead Developer: Mahbouba Rezaei
Project: ROP Stage Classification (BEng Final Year Project)
Academic Year: 2025–2026
