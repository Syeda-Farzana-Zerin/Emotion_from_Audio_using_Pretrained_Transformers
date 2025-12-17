# 🎙️ Speech Emotion Recognition using Pretrained wav2vec2

A simple and robust **Streamlit application** for **speech emotion recognition** using a **popular pretrained wav2vec2 model** from Hugging Face.  
The app allows users to select an audio file from emotion-labeled folders, run inference, and view the predicted emotion with confidence scores — **no training required**.

---

## 🚀 Features

- 🎧 Audio playback inside the browser
- 🧠 Pretrained wav2vec2 emotion recognition model
- 📂 Ground-truth labels inferred from folder names
- 📊 Confidence scores for all emotion classes
- ⚡ Lightweight and stable Streamlit UI
- 🐍 Python 3.8 compatible

---

## 📁 Dataset Structure

The dataset must follow this structure:

dataset/
├── Angry/
├── Happy/
├── Sad/
└── Neutral/

Each folder should contain `.wav` audio files corresponding to that emotion.

---

## 🤖 Pretrained Model

This project uses the following **widely used pretrained model**:

- **Model name:** `superb/wav2vec2-base-superb-er`
- **Framework:** Hugging Face Transformers
- **Benchmark:** SUPERB (Speech processing Universal PERformance Benchmark)
- **Input:** Raw audio waveform
- **Output labels:** `angry`, `happy`, `sad`, `neutral`

No feature extraction (MFCCs or spectrograms) is required.

---

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies
pip install streamlit transformers torchaudio torch soundfile


✔ Tested with Python 3.8

▶️ Run the Application
streamlit run app.py


The application will automatically open in your web browser.

🧪 How the App Works

Select an emotion folder (used as ground truth)

Select an audio file from that folder

The audio waveform is passed directly to the pretrained wav2vec2 model

The model predicts emotion probabilities

The app displays:

Ground-truth emotion

Predicted emotion

Confidence score

Scores for all emotion classes

📊 Example Output
Ground Truth: Happy
Predicted Emotion: happy
Confidence: 0.91

All Emotion Scores:
angry:   0.02
happy:   0.91
sad:     0.04
neutral: 0.03

✅ Why This Repository

✔ Uses a research-grade pretrained model

✔ No training required

✔ No handcrafted audio features

✔ Stable and beginner-friendly

✔ Ideal for demos, teaching, and baseline experiments
