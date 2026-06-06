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

- **`src/app.py`**: 🖥️ **The Web App**. This is the file you run to see the graphical interface. You can upload an image here, and the AI will tell you the result.
- **`src/train.py`**: 🎓 **The Teacher**. This script "trains" the AI. It shows the AI thousands of X-ray images so it can learn what pneumonia looks like.
- **`src/evaluate.py`**: 📝 **The Examiner**. After training, this script tests the AI to see how smart it is (calculates accuracy).
- **`src/model.py`**: 🧠 **The Brain**. This file defines the structure of the AI model (the neural network).
- **`src/data_setup.py`**: 📦 **The Unpacker**. If you have a zip file of images, this script unzips it and organizes the folders for you.
- **`src/data_loader.py`**: 🚚 **The Feeder**. This script loads the images and feeds them to the AI during training or testing.
- **`requirements.txt`**: 📋 **The Shopping List**. Opens a list of all the software libraries (like PyTorch, Streamlit) that this project needs to run.

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

### 4. Run the App
To use the AI with a nice interface, run:
```bash
streamlit run src/app.py
```
A web page will open in your browser where you can upload X-ray images and get results!

## ⚠️ Note
This project is for **educational purposes only**. It is not a certified medical device and should not be used for actual medical diagnosis.
