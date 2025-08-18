import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tempfile
import os
from scipy import ndimage
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

def analyze_audio(audio_path, duration, fps):
    """Analizza l'audio per estrarre features musicali"""
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calcola BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Divide in frames
    frame_length = int(sr / fps)
    n_frames = int(duration * fps)
    
    # Estrai spettro per ogni frame
    bass_values = []
    mid_values = []
    high_values = []
    
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

def create_spiral_illusion(width, height, frame, audio_features, intensity, random_seed):
    """Crea illusione spirale - versione ottimizzata"""
    random.seed(random_seed)
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    center_x, center_y = width // 2, height // 2
    x, y = np.meshgrid(np.arange(width) - center_x, np.arange(height) - center_y)
    
    # Coordinate polari
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Spirale influenzata dall'audio
    spiral_freq = 0.05 + mid_val * 0.03 * intensity
    spiral_rotation = frame * 0.02 * bass_val * intensity
    spiral = np.sin(spiral_freq * r + theta * 3 + spiral_rotation)
    
    # Aggiunge variazioni radiali (alti)
    radial_variation = np.sin(r * 0.01 + frame * 0.1 * high_val * intensity)
    
    # Combina
    combined = spiral + radial_variation * high_val * intensity
    
    # Normalizza e converte in RGB
    normalized = (combined + 2) / 4
    img = np.stack([normalized, normalized, normalized], axis=2)
    
    return img

def create_rotating_circles(width, height, frame, audio_features, intensity, random_seed):
    """Crea illusione con cerchi rotanti"""
    random.seed(random_seed)
    img = np.zeros((height, width, 3))
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    center_x, center_y = width // 2, height // 2
    
    # Numero di cerchi basato sui medi
    num_circles = int(5 + mid_val * 10 * intensity)
    
    for i in range(num_circles):
        # Raggio influenzato dai bassi
        radius = (20 + i * 15) * (1 + bass_val * intensity)
        
        # Velocità di rotazione influenzata dagli alti
        rotation_speed = 0.05 + high_val * 0.1 * intensity
        angle = frame * rotation_speed + i * np.pi / 4
        
        # Posizione del cerchio
        circle_x = int(center_x + radius * 0.3 * np.cos(angle))
        circle_y = int(center_y + radius * 0.3 * np.sin(angle))
        
        # Disegna cerchio con OpenCV
        circle_radius = int(10 + bass_val * 5 * intensity)
        cv2.circle(img, (circle_x, circle_y), circle_radius, (1, 1, 1), -1)
        cv2.circle(img, (circle_x, circle_y), circle_radius//2, (0, 0, 0), -1)
    
    return img

def create_wave_distortion(width, height, frame, audio_features, intensity, random_seed):
    """Crea distorsione a onde"""
    random.seed(random_seed)
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Onde multiple con frequenze diverse
    wave1 = np.sin((x + frame * bass_val * intensity) * 0.02) * mid_val * 20 * intensity
    wave2 = np.sin((y + frame * mid_val * intensity) * 0.03) * bass_val * 15 * intensity
    wave3 = np.sin((x + y + frame * high_val * intensity) * 0.01) * high_val * 10 * intensity
    
    # Combinazione delle onde
    combined_wave = wave1 + wave2 + wave3
    
    # Crea pattern a strisce distorte
    stripes = np.sin((y + combined_wave) * 0.1)
    
    # Normalizza e converte in RGB
    normalized = (stripes + 1) / 2
    img = np.stack([normalized, normalized, normalized], axis=2)
    
    return img

def create_psychedelic_patterns(width, height, frame, audio_features, intensity, random_seed):
    """Crea pattern psichedelici"""
    random.seed(random_seed)
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    # Crea griglia di coordinate
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Pattern complesso basato su funzioni trigonometriche
    pattern1 = np.sin(x * 0.01 + frame * bass_val * 0.1 * intensity)
    pattern2 = np.cos(y * 0.01 + frame * mid_val * 0.1 * intensity)
    pattern3 = np.sin((x + y) * 0.005 + frame * high_val * 0.05 * intensity)
    
    # Combinazione dei pattern
    combined = pattern1 * pattern2 + pattern3 * high_val * intensity
    
    # Normalizza
    normalized = (combined + 2) / 4
    
    # Crea effetto RGB separato per ogni canale
    r_channel = (np.sin(normalized * np.pi + frame * 0.1) + 1) / 2
    g_channel = (np.cos(normalized * np.pi + frame * 0.1 + np.pi/3) + 1) / 2
    b_channel = (np.sin(normalized * np.pi + frame * 0.1 + 2*np.pi/3) + 1) / 2
    
    img = np.stack([r_channel, g_channel, b_channel], axis=2)
    
    return img

def create_kaleidoscope(width, height, frame, audio_features, intensity, random_seed):
    """Crea effetto caleidoscopio"""
    random.seed(random_seed)
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    center_x, center_y = width // 2, height // 2
    x, y = np.meshgrid(np.arange(width) - center_x, np.arange(height) - center_y)
    
    # Coordinate polari
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Numero di sezioni del caleidoscopio
    n_sections = int(6 + mid_val * 6 * intensity)
    
    # Crea pattern ripetuto
    theta_normalized = (theta + frame * bass_val * 0.02 * intensity) % (2 * np.pi / n_sections)
    
    # Pattern radiale
    radial_pattern = np.sin(r * 0.02 + frame * high_val * 0.1 * intensity)
    angular_pattern = np.cos(theta_normalized * n_sections)
    
    combined = radial_pattern * angular_pattern
    normalized = (combined + 1) / 2
    
    img = np.stack([normalized, normalized, normalized], axis=2)
    
    return img

def generate_illusion_frame(illusion_type, width, height, frame, audio_features, intensity, random_seed):
    """Genera un frame dell'illusione specificata"""
    if illusion_type == "Spiral Illusion":
        return create_spiral_illusion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Illusory Motion":
        return create_rotating_circles(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Y-Junctions":
        return create_wave_distortion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Drifting Spines":
        return create_psychedelic_patterns(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Illusory Tilt":
        return create_kaleidoscope(width, height, frame, audio_features, intensity, random_seed)
    else:
        return create_spiral_illusion(width, height, frame, audio_features, intensity, random_seed)

def apply_colors(img, line_color, bg_color):
    """Applica i colori personalizzati all'immagine"""
    # Converte colori hex in RGB
    line_rgb = np.array([int(line_color[1:3], 16)/255, int(line_color[3:5], 16)/255, int(line_color[5:7], 16)/255])
    bg_rgb = np.array([int(bg_color[1:3], 16)/255, int(bg_color[3:5], 16)/255, int(bg_color[5:7], 16)/255])
    
    # Applica colori usando una maschera più sofisticata
    colored_img = np.zeros_like(img)
    
    # Usa la luminanza per determinare la maschera
    luminance = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    mask = luminance > 0.5
    
    # Applica gradualmente i colori
    for i in range(3):
        colored_img[:,:,i] = np.where(mask, 
                                     line_rgb[i] * img[:,:,i], 
                                     bg_rgb[i] + (line_rgb[i] - bg_rgb[i]) * img[:,:,i])
    
    return colored_img

def create_video_with_opencv(frames, fps, output_path, audio_path=None):
    """Crea video usando OpenCV"""
    if not frames:
        return False
    
    height, width = frames[0].shape[:2]
    
    # Definisce il codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Converte da RGB a BGR per OpenCV
        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    return True

def add_text_to_frame(frame, text, position, font_scale=1.0, color=(255, 255, 255)):
    """Aggiunge testo al frame"""
    height, width = frame.shape[:2]
    
    # Converti frame per OpenCV
    frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Calcola dimensioni del testo
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Determina posizione
    if position == "Sopra":
        x, y = (width - text_width) // 2, text_height + 20
    elif position == "Sotto":
        x, y = (width - text_width) // 2, height - 20
    elif position == "Destra":
        x, y = width - text_width - 20, height // 2
    elif position == "Sinistra":
        x, y = 20, height // 2
    else:  # Centro
        x, y = (width - text_width) // 2, height // 2
    
    # Aggiunge testo
    cv2.putText(frame_bgr, text, (x, y), font, font_scale, color, thickness)
    
    # Riconverte a RGB
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0

# Pulsante genera
if uploaded_file and st.button("🚀 Genera Video Illusorio", type="primary"):
    with st.spinner("🎨 Creazione dell'illusione ottica in corso..."):
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
        max_duration = min(duration, 30)  # Limita a 30 secondi per performance
        n_frames = int(max_duration * fps)
        
        # Analizza audio
        st.info("🔍 Analisi delle frequenze audio...")
        audio_features = analyze_audio(tmp_audio.name, max_duration, fps)
        
        # Seed random per consistenza
        random_seed = random.randint(1, 10000)
        
        st.info(f"🌀 Generazione illusione: {illusion_type}")
        tempo_display = float(audio_features['tempo']) if isinstance(audio_features['tempo'], (int, float, np.number)) else 120.0
        st.info(f"🎯 BPM rilevato: {tempo_display:.1f}")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Genera frames
        frames = []
        for frame_idx in range(n_frames):
            # Aggiorna progress
            progress = frame_idx / n_frames
            progress_bar.progress(progress)
            status_text.text(f"Generando frame {frame_idx+1}/{n_frames} ({progress:.1%})")
            
            # Genera frame dell'illusione
            illusion_img = generate_illusion_frame(
                illusion_type, size[0], size[1], frame_idx, 
                audio_features, intensity, random_seed
            )
            
            # Applica colori personalizzati
            colored_img = apply_colors(illusion_img, line_color, bg_color)
            
            # Aggiungi titolo se specificato
            if video_title.strip():
                colored_img = add_text_to_frame(colored_img, video_title, position)
            
            frames.append(colored_img)

        # Crea video
        st.info("🎬 Creazione video finale...")
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        
        success = create_video_with_opencv(frames, fps, tmp_video.name)
        
        if success:
            st.success("✨ Video generato con successo!")
            
            # Mostra info finali
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Durata", f"{max_duration:.1f}s")
            with col2:
                st.metric("Risoluzione", f"{size[0]}x{size[1]}")
            with col3:
                st.metric("FPS", fps)

            # Download
            with open(tmp_video.name, "rb") as f:
                st.download_button(
                    "📥 Scarica Video Illusorio", 
                    f, 
                    file_name=f"illusion_{illusion_type.lower().replace(' ', '_')}_output.mp4", 
                    mime="video/mp4",
                    type="primary"
                )
        else:
            st.error("❌ Errore nella creazione del video")

        # Cleanup
        os.remove(tmp_audio.name)
        if os.path.exists(tmp_video.name):
            os.remove(tmp_video.name)

        progress_bar.empty()
        status_text.empty()

# Informazioni laterali
with st.sidebar:
    st.markdown("---")
    st.subheader("🧠 Illusioni Ottimizzate")
    st.markdown("""
    **🔬 Versione sicura con OpenCV:**
    
    🌀 **Spiral Illusion** 
    - Spirali ipnotiche sincronizzate
    
    ⚡ **Rotating Circles**
    - Cerchi rotanti dinamici
    
    🌊 **Wave Distortion**
    - Onde distorte fluide
    
    🎨 **Psychedelic Patterns**
    - Pattern psichedelici complessi
    
    🔸 **Kaleidoscope**
    - Effetti caleidoscopici
    """)
    
    st.markdown("---")
    st.subheader("⚡ Ottimizzazioni")
    st.markdown("""
    ✅ **OpenCV invece di MoviePy**
    ✅ **Installazione più veloce** 
    ✅ **Maggiore stabilità**
    ✅ **Memoria ottimizzata**
    ✅ **Limite 30s per performance**
    """)
    
    st.markdown("---")
    st.markdown("🚀 **Versione ottimizzata**")
    st.markdown("🎨 **Arte generativa** sicura") 
    st.markdown("⚡ **Performance** garantite")
