import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon
import tempfile
import os
import random
import ffmpeg
from scipy import ndimage
from skimage.draw import line, circle, polygon

# Configurazione pagina
st.set_page_config(page_title="VJing Generativo", layout="wide")

st.title("🎵 VJing Generativo - Illusioni Ottiche Scientifiche")
st.caption("by Loop507 | Arte cinetica sincronizzata al suono con implementazioni neuropsicologiche accurate")

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

# (--- tutte le funzioni illusioni rimangono invariate, come nel tuo file ---)
# ...

# -------------------------------
# MAIN
# -------------------------------
if uploaded_file and st.button("🚀 Genera Video Illusorio Scientifico", type="primary"):
    with st.spinner("🎨 Creazione video con illusioni scientificamente accurate..."):
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
        st.info(f"🎯 BPM rilevato: {tempo_display:.1f}")
        st.info(f"🧬 Illusione selezionata: {illusion_type} (implementazione scientifica)")

        seed=random.randint(1,10000)
        fig, ax = plt.subplots(figsize=(size[0]/100,size[1]/100), dpi=100)
        ax.axis("off"); 
        ax.set_facecolor('black')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        
        im=ax.imshow(np.zeros((size[1],size[0],3)),aspect="equal")

        def animate(frame):
            illusion = generate_illusion_frame(size[0], size[1], frame, audio_features, 
                                             intensity, illusion_type, seed)
            colored = apply_colors(illusion, line_color, bg_color)
            im.set_array(colored)
            return [im]

        anim=FuncAnimation(fig,animate,frames=n_frames,blit=True,interval=1000/fps)
        tmp_video=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
        
        # ✅ Correzione qui: uso FFMpegWriter invece di plt.writers
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Loop507'), bitrate=1800)
        anim.save(tmp_video.name, writer=writer)
        plt.close(fig)

        # 🔥 Merge audio + video con ffmpeg
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video = ffmpeg.input(tmp_video.name)
        audio = ffmpeg.input(tmp_audio.name)
        final = ffmpeg.output(video, audio, output_file.name, 
                             vcodec="libx264", acodec="aac", 
                             shortest=None, strict="experimental")
        ffmpeg.run(final, overwrite_output=True, quiet=True)

        # Overlay titolo (se inserito)
        if video_title.strip():
            pos_map={
                "Sopra":"(w-text_w)/2:20",
                "Sotto":"(w-text_w)/2:h-text_h-20",
                "Destra":"w-text_w-20:(h-text_h)/2",
                "Sinistra":"20:(h-text_h)/2",
                "Centro":"(w-text_w)/2:(h-text_h)/2"
            }
            titled_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            (
                ffmpeg
                .input(output_file.name)
                .output(
                    titled_file.name,
                    vf=f"drawtext=text='{video_title}':fontcolor=white:fontsize=48:fontfile=/System/Library/Fonts/Arial.ttf:x={pos_map[position].split(':')[0]}:y={pos_map[position].split(':')[1]}",
                    vcodec="libx264", acodec="aac", strict="experimental"
                )
                .run(overwrite_output=True, quiet=True)
            )
            os.replace(titled_file.name, output_file.name)

        with open(output_file.name,"rb") as f:
            st.download_button("📥 Scarica Video Illusorio Scientifico",f,
                             file_name=f"vjing_{illusion_type.lower().replace(' ', '_')}_output.mp4",
                             mime="video/mp4")

        # Cleanup
        os.remove(tmp_audio.name); os.remove(tmp_video.name); os.remove(output_file.name)
        st.success("✨ Video generato con successo! Implementazioni neuropsicologiche accurate.")
        
        # Info scientifica
        st.info(f"""
        🧬 **Implementazione Scientifica Utilizzata:**
        - **{illusion_type}**: Basato su ricerca neuropsicologica
        - **Sincronizzazione Audio**: BPM→velocità transizioni, Bassi→movimenti globali, Medi→deformazioni, Alti→micro-dettagli
        - **Algoritmi**: Mather & Takeuchi (Motion), Retinal Slip (Y-Junctions), Phi Motion Effects
        """)
