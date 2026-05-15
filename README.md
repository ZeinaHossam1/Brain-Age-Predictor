# Brain Age Predictor using MRI Scans and Deep Learning

AI-powered medical imaging platform for predicting biological brain age from MRI scans using deep learning, ensemble CNN models, and FastAPI deployment.

---

# ⚠️ Model Files

Due to GitHub and submission size limitations, the trained deep learning model files are hosted externally.

## Download Models

Download the trained model files from:

(https://drive.google.com/drive/folders/14037gzkNL3VXJHRG8qsSW2XYmYO1JsxS?usp=sharing)

After downloading, place the model files inside:

```text id="4f4ib1"
backend/
```

Required model files:

```text id="zl8l3j"
model_EfficientNetB0_best.keras
model_ResNet50_best.keras
model_DenseNet121_best.keras
```

Without these files, the backend inference API will not run correctly.

# 🧠 Project Overview

This project predicts a patient's **brain age** using structural MRI brain scans in **NIfTI (.nii / .nii.gz)** format.

The system combines:

* Medical image preprocessing
* Deep learning models
* Ensemble learning
* FastAPI backend
* Interactive web interface

The project explores multiple approaches including:

* 2D CNN models
* 3D CNN volumetric models
* Transfer learning
* Ensemble learning

The final deployed version uses a **2D Ensemble CNN model** because it achieved the best balance between:

* Accuracy
* Stability
* Fast inference
* Real-time deployment compatibility

---

# 🚀 Final Results

| Metric                     | Result         |
| -------------------------- | -------------- |
| Accuracy (within ±5 years) | **66.8%**      |
| MAE                        | **4.66 years** |
| RMSE                       | **6.61**       |
| R² Score                   | **0.9251**     |

---

# 🏗️ Project Architecture

```text
MRI Scan (.nii/.nii.gz)
        ↓
Preprocessing Pipeline
        ↓
2D Slice Extraction + Normalization
        ↓
Ensemble CNN Models
(EfficientNetB0 + ResNet50 + DenseNet121)
        ↓
Age Prediction
        ↓
FastAPI Backend
        ↓
Interactive Web Interface
```

---

# 🧪 Experiments Conducted

## 1️⃣ 2D Ensemble CNN (Final Production Model)

Models used:

* EfficientNetB0
* ResNet50
* DenseNet121

Technique:

* Transfer Learning
* Ensemble Averaging

Final Performance:

* Accuracy: 66.8%
* MAE: 4.66 years

Why selected?

* Stable predictions
* Fast inference
* Easy deployment
* Best compatibility with web app

---

## 2️⃣ 3D CNN Model

A full volumetric MRI approach using:

* Conv3D
* BatchNorm
* MaxPooling3D

Advantages:

* Uses complete brain volume
* Better spatial feature learning

Challenges:

* Heavy GPU usage
* Large memory requirements
* Slower inference

Performance:

* ~38% accuracy using limited samples

---

## 3️⃣ MONAI + 3D ResNet50

Advanced medical imaging pipeline using:

* MONAI preprocessing
* Soft classification over age bins
* 3D ResNet50 backbone

Best experimental result:

* MAE ≈ 3.7 years

However, deployment complexity and inference instability made the 2D ensemble model more practical for production.

---

# 📂 Dataset

Dataset Used:

* `radiata-ai/brain-structure`

Contains:

* T1-weighted MRI scans
* Brain volumes in NIfTI format
* Age labels
* Multiple public MRI datasets

Included subsets:

* IXI
* OASIS-1
* OASIS-2
* NKI-RS
* DLBS

---

# ⚙️ Preprocessing Pipeline

MRI preprocessing included:

## ✅ MRI Loading

Using:

```python
nibabel
```

## ✅ Slice Extraction

Five representative brain slices were extracted:

```python
z // 5
2 * z // 5
z // 2
3 * z // 5
4 * z // 5
```

## ✅ Averaging

Slices were averaged to preserve more anatomical information.

## ✅ Normalization

Min-max normalization:

```python
(x - min) / (max - min)
```

## ✅ Resizing

MRI slices resized to:

```text
128 × 128
```

## ✅ RGB Conversion

Converted grayscale MRI slices into 3-channel input for pretrained CNNs.

---

# 🖥️ Web Application

## Backend

Built using:

* FastAPI
* TensorFlow / Keras
* NumPy
* OpenCV
* Nibabel

Features:

* MRI upload
* Real-time prediction
* AI inference API
* MRI preprocessing

---

## Frontend

Features:

* Modern medical UI
* MRI upload support
* Prediction visualization
* Processing information display
* Responsive design

Supported formats:

* `.nii`
* `.nii.gz`

---

# 🧠 Models

The project used multiple trained models:

* EfficientNetB0
* ResNet50
* DenseNet121
* 3D CNN
* SFCN 3D
* MONAI 3D ResNet50

---

# 📦 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/ZeinaHossam1/Brain-Age-Predictor.git
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Backend

Inside `backend/`:

```bash
uvicorn main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

# 🌐 Running the Frontend

Open:

```text
brain_age_frontend.html
```

in your browser.

---

# 🧪 API Endpoints

## Health Check

```http
GET /health
```

---

## Brain Age Prediction

```http
POST /predict
```

Upload:

* `.nii`
* `.nii.gz`

Returns:

```json
{
  "predicted_age": 39.8,
  "filename": "scan.nii.gz",
  "processing_time_s": 0.33
}
```

---

# 📸 Example Prediction

| MRI Scan   | Predicted Age |
| ---------- | ------------- |
| IXI Sample | 39.8 years    |

---

# ⚠️ Challenges Faced

During development several challenges were solved:

* TensorFlow/Keras compatibility issues
* MRI preprocessing inconsistencies
* 3D memory limitations
* Model weight loading mismatches
* FastAPI deployment issues
* Google Drive dataset limitations

---

# 🔮 Future Improvements

Potential future work:

* Vision Transformers (ViT)
* Explainable AI heatmaps
* Larger MRI datasets
* Cloud deployment
* Multi-modal neurological biomarkers
* Improved 3D architectures

---

# 🛠️ Technologies Used

| Category        | Technologies          |
| --------------- | --------------------- |
| Programming     | Python                |
| Deep Learning   | TensorFlow, Keras     |
| Medical Imaging | Nibabel               |
| Backend         | FastAPI               |
| Frontend        | HTML, CSS, JavaScript |
| Data Processing | NumPy, OpenCV, Pandas |
| Deployment      | Uvicorn               |
| MRI Format      | NIfTI (.nii/.nii.gz)  |

---

# 📚 References

1. TensorFlow Documentation
2. Keras Documentation
3. FastAPI Documentation
4. MONAI Documentation
5. Nibabel Documentation
6. Radiata-AI Brain Structure Dataset

---

# ⭐ Final Note

This project demonstrates the integration of:

* Medical imaging
* Deep learning
* AI deployment
* Full-stack AI engineering

to build a real-world healthcare AI application for brain age prediction from MRI scans.
