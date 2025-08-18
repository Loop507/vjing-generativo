import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os
import random
import ffmpeg

# Configurazione pagina
st.set_page_config(page_title="VJing Generativo", layout="wide")

st.title("🎵 VJing Generativo - Illusioni Ottiche")
st.caption("by Loop507 | Arte cinetica sincronizzata al suono")

# Sidebar
st.sidebar.header("⚙️ Controlli")

uploaded_file = st.file_uploader("🎵 Carica un file audio (.mp3 o .wav)", type=["mp3", "wav"])

st.sidebar.subheader("🎨 Personalizzazione Colori")
line_color = st.sidebar.color_picker("Colore linee/forme", "#FFFFFF")
bg_color = st.sidebar.color_picker("Colore sfondo", "#000000")

illusion_type = st.sidebar.selectbox(
    "🌀 Tipo di Illusione", 
    ["Illusory Tilt", "Illusory Motion", "Y-Junctions", "Drifting Spines", "Spiral Illusion"]
)

video_title = st.text_input("📝 Titolo del video", "Visual Synthesis")
position = st.selectbox("📍 Posizione del titolo", ["Sopra", "Sotto", "Destra", "Sinistra", "Centro"])
aspect_ratio = st.selectbox("📺 Formato video", ["16:9", "1:1", "9:16"])
intensity = st.sidebar.slider("🔥 Intensità effetti", 0.1, 2.0, 1.0, 0.1)

# -------------------------------
# ANALISI AUDIO
# -------------------------------
def analyze_audio(audio_path, duration, fps):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    frame_length = int(sr / fps)
    n_frames = int(duration * fps)
    bass_values, mid_values, high_values = [], [], []

    for i in range(n_frames):
        start = i * frame_length
        end = min(start + frame_length, len(y))
        frame_audio = y[start:end] if start < len(y) else np.zeros(frame_length)
        fft = np.abs(np.fft.fft(frame_audio))
        freqs = np.fft.fftfreq(len(fft), 1/sr)

        bass_values.append(np.mean(fft[(freqs>=20)&(freqs<=250)]) if np.any((freqs>=20)&(freqs<=250)) else 0)
        mid_values.append(np.mean(fft[(freqs>=250)&(freqs<=4000)]) if np.any((freqs>=250)&(freqs<=4000)) else 0)
        high_values.append(np.mean(fft[(freqs>=4000)&(freqs<=20000)]) if np.any((freqs>=4000)&(freqs<=20000)) else 0)

    bass_values = np.array(bass_values); mid_values = np.array(mid_values); high_values = np.array(high_values)
    if bass_values.max()>0: bass_values /= bass_values.max()
    if mid_values.max()>0: mid_values /= mid_values.max()
    if high_values.max()>0: high_values /= high_values.max()

    return {"tempo": tempo, "bass": bass_values, "mid": mid_values, "high": high_values}

# -------------------------------
# ILLUSIONE SEMPLICE
# -------------------------------
def generate_illusion_frame(width, height, frame, audio_features, intensity, seed):
    np.random.seed(seed+frame)
    img = np.zeros((height, width, 3))
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    step = int(40 + bass_val*30)
    for y in range(0, height, step):
        for x in range(0, width, step):
            if (x//step + y//step + frame//10) % 2 == 0:
                size = int(15 + bass_val*25)
                img[y:y+size, x:x+size] = [1,1,1]
    return img

def apply_colors(img, line_color, bg_color):
    line_rgb = np.array([int(line_color[1:3],16)/255, int(line_color[3:5],16)/255, int(line_color[5:7],16)/255])
    bg_rgb   = np.array([int(bg_color[1:3],16)/255, int(bg_color[3:5],16)/255, int(bg_color[5:7],16)/255])
    colored = np.zeros_like(img)
    mask = img[:,:,0] > 0.5
    colored[mask] = line_rgb
    colored[~mask] = bg_rgb
    return colored

# -------------------------------
# MAIN
# -------------------------------
if uploaded_file and st.button("🚀 Genera Video Illusorio", type="primary"):
    with st.spinner("🎨 Creazione video in corso..."):
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_audio.write(uploaded_file.read())
        tmp_audio.close()

        y, sr = librosa.load(tmp_audio.name, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        st.info(f"🎵 Durata audio: {duration:.2f} sec")

        if aspect_ratio == "16:9": size=(1280,720)
        elif aspect_ratio == "1:1": size=(720,720)
        else: size=(720,1280)

        fps=30; n_frames=int(duration*fps)
        audio_features=analyze_audio(tmp_audio.name, duration, fps)
        tempo_display = audio_features["tempo"] if isinstance(audio_features["tempo"],(int,float)) else 120.0
        st.info(f"🎯 BPM: {tempo_display:.1f}")

        seed=random.randint(1,10000)
        fig, ax = plt.subplots(figsize=(size[0]/100,size[1]/100), dpi=100)
        ax.axis("off"); im=ax.imshow(np.zeros((size[1],size[0],3)),aspect="equal")

        def animate(frame):
            illusion=generate_illusion_frame(size[0],size[1],frame,audio_features,intensity,seed)
            colored=apply_colors(illusion,line_color,bg_color)
            im.set_array(colored)
            return [im]

        anim=FuncAnimation(fig,animate,frames=n_frames,blit=True,interval=1000/fps)
        tmp_video=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
        anim.save(tmp_video.name,fps=fps,extra_args=['-vcodec','libx264'])
        plt.close(fig)

        # ffmpeg merge audio + video
        output_file=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
        (
            ffmpeg
            .input(tmp_video.name)
            .output(tmp_audio.name)
        )
        ffmpeg.input(tmp_video.name).output(tmp_audio.name).run()

        # finale con titolo
        video = ffmpeg.input(tmp_video.name)
        audio = ffmpeg.input(tmp_audio.name)
        final = ffmpeg.output(video, audio, output_file.name, vcodec="libx264", acodec="aac", strict="experimental")

        # Overlay titolo se serve
        if video_title.strip():
            pos_map={
                "Sopra":"(w-text_w)/2:20",
                "Sotto":"(w-text_w)/2:h-text_h-20",
                "Destra":"w-text_w-20:(h-text_h)/2",
                "Sinistra":"20:(h-text_h)/2",
                "Centro":"(w-text_w)/2:(h-text_h)/2"
            }
            final = ffmpeg.filter([video], 'drawtext',
                text=video_title,
                x=pos_map[position].split(":")[0],
                y=pos_map[position].split(":")[1],
                fontsize=48, fontcolor="white"
            )
            final = ffmpeg.output(final, audio, output_file.name, vcodec="libx264", acodec="aac", strict="experimental")

        ffmpeg.run(final, overwrite_output=True)

        with open(output_file.name,"rb") as f:
            st.download_button("📥 Scarica Video Illusorio",f,file_name="vjing_output.mp4",mime="video/mp4")

        os.remove(tmp_audio.name); os.remove(tmp_video.name); os.remove(output_file.name)
        st.success("✨ Video generato con successo!")
