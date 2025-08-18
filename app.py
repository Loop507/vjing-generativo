import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import moviepy.editor as mpy
import tempfile
import os
import random

# Configurazione pagina
st.set_page_config(page_title="VJing Generativo", layout="wide")

# Titolo dell'app
st.title("🎵 VJing Generativo - Illusioni Ottiche")
st.caption("by Loop507 | Arte cinetica sincronizzata al suono")

# Sidebar per controlli
st.sidebar.header("⚙️ Controlli")

# Upload audio
uploaded_file = st.file_uploader("🎵 Carica un file audio (.mp3 o .wav)", type=["mp3", "wav"])

# Personalizzazione colori
st.sidebar.subheader("🎨 Personalizzazione Colori")
line_color = st.sidebar.color_picker("Colore linee/forme", "#FFFFFF")
bg_color = st.sidebar.color_picker("Colore sfondo", "#000000")

# Tipo di illusione
illusion_type = st.sidebar.selectbox(
    "🌀 Tipo di Illusione", 
    ["Illusory Tilt", "Illusory Motion", "Y-Junctions", "Drifting Spines", "Spiral Illusion"]
)

# Input titolo video
video_title = st.text_input("📝 Titolo del video", "Visual Synthesis")

# Posizione titolo
position = st.selectbox("📍 Posizione del titolo", ["Sopra", "Sotto", "Destra", "Sinistra", "Centro"])

# Scelta formato
aspect_ratio = st.selectbox("📺 Formato video", ["16:9", "1:1", "9:16"])

# Intensità effetti
intensity = st.sidebar.slider("🔥 Intensità effetti", 0.1, 2.0, 1.0, 0.1)


# -------------------------------
# ANALISI AUDIO
# -------------------------------
def analyze_audio(audio_path, duration, fps):
    """Analizza l'audio per estrarre features musicali"""
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calcola BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    # Divide in frames
    frame_length = int(sr / fps)
    n_frames = int(duration * fps)
    
    # Estrai spettro per ogni frame
    bass_values, mid_values, high_values = [], [], []
    
    for i in range(n_frames):
        start_sample = i * frame_length
        end_sample = min(start_sample + frame_length, len(y))
        
        if start_sample < len(y):
            frame_audio = y[start_sample:end_sample]
            
            # FFT per analisi frequenze
            fft = np.abs(np.fft.fft(frame_audio))
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            
            # Dividi in bande
            bass_mask = (freqs >= 20) & (freqs <= 250)
            mid_mask = (freqs >= 250) & (freqs <= 4000)
            high_mask = (freqs >= 4000) & (freqs <= 20000)
            
            bass_values.append(np.mean(fft[bass_mask]) if np.any(bass_mask) else 0)
            mid_values.append(np.mean(fft[mid_mask]) if np.any(mid_mask) else 0)
            high_values.append(np.mean(fft[high_mask]) if np.any(high_mask) else 0)
        else:
            bass_values.append(0)
            mid_values.append(0)
            high_values.append(0)
    
    # Normalizza
    bass_values = np.array(bass_values)
    mid_values = np.array(mid_values)
    high_values = np.array(high_values)
    
    if bass_values.max() > 0:
        bass_values = bass_values / bass_values.max()
    if mid_values.max() > 0:
        mid_values = mid_values / mid_values.max()
    if high_values.max() > 0:
        high_values = high_values / high_values.max()
    
    return {
        'tempo': tempo,
        'bass': bass_values,
        'mid': mid_values,
        'high': high_values
    }


# -------------------------------
# ILLUSIONI SEMPLIFICATE (demo)
# -------------------------------
def generate_illusion_frame(illusion_type, width, height, frame, audio_features, intensity, random_seed):
    """Genera frame di illusione semplice per testing"""
    img = np.zeros((height, width, 3))
    
    # valori audio
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    # Semplice griglia animata
    step = int(50 + bass_val * 30)
    for y in range(0, height, step):
        for x in range(0, width, step):
            if (x//step + y//step + frame//10) % 2 == 0:
                size = int(20 + mid_val*30)
                img[y:y+size, x:x+size] = [1,1,1]
    
    return img


def apply_colors(img, line_color, bg_color):
    """Applica colori personalizzati"""
    line_rgb = np.array([int(line_color[1:3],16)/255, int(line_color[3:5],16)/255, int(line_color[5:7],16)/255])
    bg_rgb   = np.array([int(bg_color[1:3],16)/255, int(bg_color[3:5],16)/255, int(bg_color[5:7],16)/255])
    
    colored = np.zeros_like(img)
    mask = img[:,:,0] > 0.5
    colored[mask] = line_rgb
    colored[~mask] = bg_rgb
    return colored


# -------------------------------
# MAIN APP
# -------------------------------
if uploaded_file and st.button("🚀 Genera Video Illusorio", type="primary"):
    with st.spinner("🎨 Creazione video in corso..."):
        # Salvataggio temporaneo audio
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_audio.write(uploaded_file.read())
        tmp_audio.close()

        # Caricamento audio con librosa
        y, sr = librosa.load(tmp_audio.name, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        st.info(f"🎵 Durata audio: {duration:.2f} secondi")

        # Parametri video
        if aspect_ratio == "16:9":
            size = (1280, 720)
        elif aspect_ratio == "1:1":
            size = (720, 720)
        else:  # 9:16
            size = (720, 1280)

        fps = 30
        n_frames = int(duration * fps)
        
        # Analizza audio
        st.info("🔍 Analisi audio...")
        audio_features = analyze_audio(tmp_audio.name, duration, fps)
        
        # Seed random
        random_seed = random.randint(1, 10000)
        
        # Mostra BPM
        tempo_display = float(audio_features['tempo']) if isinstance(audio_features['tempo'], (int, float, np.number)) else 120.0
        st.info(f"🎯 BPM rilevato: {tempo_display:.1f}")

        # Funzione animazione
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax.set_xlim(0, size[0])
        ax.set_ylim(0, size[1])
        ax.axis("off")
        fig.patch.set_facecolor('black')
        
        im = ax.imshow(np.zeros((size[1], size[0], 3)), aspect='equal')

        def animate(frame):
            illusion_img = generate_illusion_frame(
                illusion_type, size[0], size[1], frame,
                audio_features, intensity, random_seed
            )
            colored_img = apply_colors(illusion_img, line_color, bg_color)
            im.set_array(colored_img)
            return [im]

        anim = FuncAnimation(fig, animate, frames=n_frames, blit=True, interval=1000/fps)

        # Salva animazione (no audio)
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        anim.save(tmp_video.name, fps=fps, extra_args=['-vcodec', 'libx264'])
        plt.close(fig)

        # Aggiungi audio + titolo con MoviePy
        st.info("🎬 Composizione finale...")
        video_clip = mpy.VideoFileClip(tmp_video.name).set_audio(mpy.AudioFileClip(tmp_audio.name))

        if video_title.strip():
            txt_clip = mpy.TextClip(
                video_title, fontsize=min(60, size[0]//20), color='white', font='Arial-Bold'
            ).set_duration(video_clip.duration)

            if position == "Sopra":
                txt_clip = txt_clip.set_position(("center","top"))
            elif position == "Sotto":
                txt_clip = txt_clip.set_position(("center","bottom"))
            elif position == "Destra":
                txt_clip = txt_clip.set_position(("right","center"))
            elif position == "Sinistra":
                txt_clip = txt_clip.set_position(("left","center"))
            else:
                txt_clip = txt_clip.set_position("center")

            final = mpy.CompositeVideoClip([video_clip, txt_clip])
        else:
            final = video_clip

        # Salvataggio finale
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        final.write_videofile(output_file.name, codec="libx264", audio_codec="aac", verbose=False)

        # Download
        with open(output_file.name, "rb") as f:
            st.download_button("📥 Scarica Video Illusorio", f, file_name="vjing_output.mp4", mime="video/mp4")

        # Cleanup
        os.remove(tmp_audio.name)
        os.remove(tmp_video.name)
        os.remove(output_file.name)

        st.success("✨ Video generato con successo!")
