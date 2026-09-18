# Sign Language Translator

This project uses MediaPipe and Machine Learning to detect sign language gestures and translate them into multiple languages (English, Hindi, Tamil, etc.).

## Prerequisites

- Python 3.9 (recommended)
- Webcam

## Setup

A virtual environment has been created for you using Python 3.9 to ensure compatibility with MediaPipe.

1.  **Activate the virtual environment:**
    ```powershell
    .\venv\Scripts\activate
    ```
    (Or just use `.\venv\Scripts\python` for commands)

2.  **Install dependencies (if not already done):**
    ```powershell
    pip install -r requirements.txt
    ```

## How to Run

### 1. Collect Data
First, you need to collect images for the signs you want to recognize.
Run the data collection script:
```powershell
.\venv\Scripts\python collect_data.py
```
- It will ask for the name of the class (e.g., "Hello").
- Press 'Q' when you are ready to start capturing.
- It will capture 100 images per class.
- Repeat for all classes (default is 3 classes).

### 2. Create Dataset
Process the collected images to extract hand landmarks:
```powershell
.\venv\Scripts\python create_dataset.py
```
This will create a `data.pickle` file.

### 3. Train Model
Train the Random Forest classifier:
```powershell
.\venv\Scripts\python train_model.py
```
This will create a `model.p` file.

### 4. Run the App
Start the Streamlit application:
```powershell
.\venv\Scripts\streamlit run app.py
```
- The app will open in your browser.
- Select the target language from the sidebar.
- Check "Run Camera" to start translation.

## Project Structure

- `collect_data.py`: Captures images from webcam.
- `create_dataset.py`: Extracts MediaPipe landmarks from images.
- `train_model.py`: Trains a classifier on the landmarks.
- `app.py`: Real-time inference and translation web app.
- `data/`: Stores raw images.
- `models/`: Stores trained models.
