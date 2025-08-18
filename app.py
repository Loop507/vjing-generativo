import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import moviepy.editor as mpy
import tempfile
import os

# Titolo dell'app
st.title("VJing Generativo")
st.caption("by Loop507")

# Upload audio
uploaded_file = st.file_uploader("Carica un file audio (.mp3 o .wav)", type=["mp3", "wav"])

# Input titolo video
video_title = st.text_input("Titolo del video", "Titolo Demo")

# Posizione titolo
position = st.selectbox("Posizione del titolo", ["Sopra", "Sotto", "Destra", "Sinistra"])

# Scelta formato
aspect_ratio = st.selectbox("Formato video", ["16:9", "1:1", "9:16"])

# Pulsante genera
if uploaded_file and st.button("Genera Video"):
    # Salvataggio temporaneo audio
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_audio.write(uploaded_file.read())
    tmp_audio.close()

    # Caricamento audio con librosa
    y, sr = librosa.load(tmp_audio.name, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Parametri video
    if aspect_ratio == "16:9":
        size = (1280, 720)
    elif aspect_ratio == "1:1":
        size = (720, 720)
    else:  # 9:16
        size = (720, 1280)

    fps = 30
    n_frames = int(duration * fps)

    # Funzione animazione (placeholder: cerchi pulsanti)
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    circle, = ax.plot([], [], 'k', lw=2)

    def init():
        circle.set_data([], [])
        return circle,

    def animate(i):
        theta = np.linspace(0, 2*np.pi, 100)
        r = 0.5 + 0.1*np.sin(2*np.pi * i / fps)  # pulsazione
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        circle.set_data(x, y)
        return circle,

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, blit=True)

    # Salva animazione come mp4 (senza audio)
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    anim.save(tmp_video.name, fps=fps, extra_args=['-vcodec', 'libx264'])

    # Aggiungi audio + titolo con MoviePy
    video_clip = mpy.VideoFileClip(tmp_video.name).set_audio(mpy.AudioFileClip(tmp_audio.name))

    # Overlay testo titolo
    txt_clip = mpy.TextClip(
        video_title, fontsize=60, color='white'
    ).set_duration(video_clip.duration)

    if position == "Sopra":
        txt_clip = txt_clip.set_position(("center", "top"))
    elif position == "Sotto":
        txt_clip = txt_clip.set_position(("center", "bottom"))
    elif position == "Destra":
        txt_clip = txt_clip.set_position(("right", "center"))
    else:  # Sinistra
        txt_clip = txt_clip.set_position(("left", "center"))

    final = mpy.CompositeVideoClip([video_clip, txt_clip])

    # Salvataggio finale
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    final.write_videofile(output_file.name, codec="libx264", audio_codec="aac")

    # Download
    with open(output_file.name, "rb") as f:
        st.download_button("📥 Scarica Video", f, file_name="vjing_output.mp4", mime="video/mp4")

    # Cleanup
    os.remove(tmp_audio.name)
    os.remove(tmp_video.name)
    os.remove(output_file.name)
