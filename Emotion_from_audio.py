import os
import streamlit as st
import torch
import torchaudio
from transformers import pipeline

# ---------------------------
# CONFIG
# ---------------------------
DATASET_DIR = "dataset"
EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

# ---------------------------
# LOAD PRETRAINED MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return pipeline(
        task="audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=0 if torch.cuda.is_available() else -1
    )

classifier = load_model()

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🎙️ Speech Emotion Recognition (Pretrained)")
st.write("Model: wav2vec2 (SUPERB Emotion Recognition)")

gt_emotion = st.selectbox("Ground Truth Emotion (Folder)", EMOTIONS)

folder_path = os.path.join(DATASET_DIR, gt_emotion)
audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

audio_file = st.selectbox("Select Audio File", audio_files)
file_path = os.path.join(folder_path, audio_file)

st.audio(file_path)

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict Emotion"):
    waveform, sr = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    waveform = waveform.squeeze().numpy()

    results = classifier(waveform, sampling_rate=sr)

    st.subheader("Results")
    st.write(f"**Ground Truth:** {gt_emotion}")
    st.write(f"**Predicted Emotion:** {results[0]['label']}")
    st.write(f"**Confidence:** {results[0]['score']:.3f}")

    st.subheader("All Emotion Scores")
    for r in results:
        st.write(f"{r['label']}: {r['score']:.3f}")

