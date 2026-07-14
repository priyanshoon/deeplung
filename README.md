# DeepLung: Pneumonia Detection AI 🫁

Welcome to **DeepLung**! This project uses Artificial Intelligence (AI) to analyze chest X-ray images and detect signs of pneumonia and other lung conditions. Think of it as a smart assistant for doctors.

## 🌟 What is this?
This is a Deep Learning project that uses a **Convolutional Neural Network (CNN)**. It looks at an X-ray image and tries to classify it into one of these categories:
- **Normal** (Healthy lung)
- **Bacterial Pneumonia**
- **Viral Pneumonia**
- **Corona Virus Disease**
- **Tuberculosis**

## 📂 Project Structure (Explained for Beginners)
Here is what each file in this project does:

- **`src/server.py`**: 🔌 **The API Server**. Flask backend that loads the model and serves predictions to the React frontend.
- **`frontend/`**: 🖥️ **The Web App**. React + Vite UI for uploading X-rays and viewing results.
- **`src/app.py`**: *(legacy)* Old Streamlit interface, kept for reference.
- **`src/train.py`**: 🎓 **The Teacher**. This script "trains" the AI. It shows the AI thousands of X-ray images so it can learn what pneumonia looks like.
- **`src/evaluate.py`**: 📝 **The Examiner**. After training, this script tests the AI to see how smart it is (calculates accuracy).
- **`src/model.py`**: 🧠 **The Brain**. This file defines the structure of the AI model (the neural network).
- **`src/data_setup.py`**: 📦 **The Unpacker**. If you have a zip file of images, this script unzips it and organizes the folders for you.
- **`src/data_loader.py`**: 🚚 **The Feeder**. This script loads the images and feeds them to the AI during training or testing.
- **`requirements.txt`**: 📋 **The Shopping List**. Lists Python libraries (PyTorch, Flask, etc.) needed for the backend.

## 🚀 How to Run It

### 1. Install Dependencies
First, you need to install the necessary tools. Open your terminal and run:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data
Make sure you have your dataset ready. If it's in a zip file (named `archive.zip`), run:
```bash
python src/data_setup.py
```
This will extract the images into a `data/` folder.

### 3. Train the Model (Optional)
If you want to train the AI yourself (this can take a long time!), run:
```bash
python src/train.py
```
This will save a file named `best_model.pth` in the `checkpoints/` folder. This file contains the "knowledge" of the AI.

### 4. Get a Trained Model
You need `checkpoints/best_model.pth` (from training) or a placeholder for testing:
```bash
python save_dummy.py
```
This creates `checkpoints_dryrun/best_model.pth` so the app can run without full training.

### 5. Run the App (React + Flask)
You need **two terminals**:

**Terminal 1 — start the Python API:**
```bash
python src/server.py
```

**Terminal 2 — start the React frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite (usually `http://localhost:5173`). Upload a chest X-ray to get a prediction.

> **Legacy:** The old Streamlit UI is still available via `streamlit run src/app.py`, but the React app is the supported interface.

## ⚠️ Note
This project is for **educational purposes only**. It is not a certified medical device and should not be used for actual medical diagnosis.

## 🛡️ Security & Validation
The API includes a robust validation step to ensure uploaded images are actual chest X-rays. It checks for aspect ratio, color variance, contrast, and brightness distribution to reject regular photos, screenshots, or non-medical images before processing.
