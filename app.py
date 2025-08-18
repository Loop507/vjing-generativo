import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import moviepy.editor as mpy
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
    ["Illusory Tilt", "Moiré Patterns", "Rotating Circles", "Wave Distortion", "Spiral Illusion"]
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

def create_illusory_tilt(width, height, frame, audio_features, intensity, random_seed):
    """Crea illusione di inclinazione con griglia"""
    random.seed(random_seed)
    img = np.zeros((height, width, 3))
    
    # Parametri guidati dall'audio
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    # Griglia dinamica
    line_spacing = 15 + int(5 * bass_val * intensity)
    line_thickness = max(1, int(2 + 3 * mid_val * intensity))
    
    # Oscillazione globale (bassi)
    global_tilt = np.sin(frame * 0.1 * intensity) * bass_val * 0.3
    
    # Vibrazione (medi)
    vibration = np.sin(frame * 0.5 * intensity) * mid_val * 0.1
    
    # Distorsione locale (alti)
    noise_factor = high_val * intensity * 0.2
    
    for y in range(0, height, line_spacing):
        # Calcola inclinazione per questa linea
        local_tilt = global_tilt + vibration + random.uniform(-noise_factor, noise_factor)
        
        for x in range(width):
            # Applica distorsione
            distorted_y = int(y + x * local_tilt + np.sin(x * 0.02 + frame * 0.1) * mid_val * 5 * intensity)
            
            # Disegna linea con spessore
            for thickness in range(line_thickness):
                final_y = distorted_y + thickness
                if 0 <= final_y < height:
                    img[final_y, x] = [1, 1, 1]  # Bianco, verrà colorato dopo
    
    return img

def create_moire_patterns(width, height, frame, audio_features, intensity, random_seed):
    """Crea pattern Moiré dinamici"""
    random.seed(random_seed)
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Pattern 1 - influenzato dai bassi
    freq1 = 0.05 + bass_val * 0.03 * intensity
    pattern1 = np.sin(freq1 * (x + y + frame * bass_val * intensity))
    
    # Pattern 2 - influenzato dai medi
    freq2 = 0.07 + mid_val * 0.02 * intensity
    rotation = frame * 0.01 * mid_val * intensity
    x_rot = x * np.cos(rotation) - y * np.sin(rotation)
    y_rot = x * np.sin(rotation) + y * np.cos(rotation)
    pattern2 = np.sin(freq2 * (x_rot + y_rot))
    
    # Combinazione con interferenza (alti)
    interference = high_val * intensity
    combined = (pattern1 + pattern2) * (1 + interference)
    
    # Converte in immagine RGB
    normalized = (combined + 2) / 4  # Normalizza 0-1
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
        circle_x = center_x + radius * 0.3 * np.cos(angle)
        circle_y = center_y + radius * 0.3 * np.sin(angle)
        
        # Disegna cerchio
        y, x = np.ogrid[:height, :width]
        mask = (x - circle_x)**2 + (y - circle_y)**2 <= (10 + bass_val * 5 * intensity)**2
        img[mask] = [1, 1, 1]
        
        # Cerchio interno vuoto
        inner_mask = (x - circle_x)**2 + (y - circle_y)**2 <= (5 + bass_val * 2 * intensity)**2
        img[inner_mask] = [0, 0, 0]
    
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

def create_spiral_illusion(width, height, frame, audio_features, intensity, random_seed):
    """Crea illusione spirale"""
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

def generate_illusion_frame(illusion_type, width, height, frame, audio_features, intensity, random_seed):
    """Genera un frame dell'illusione specificata"""
    if illusion_type == "Illusory Tilt":
        return create_illusory_tilt(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Moiré Patterns":
        return create_moire_patterns(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Rotating Circles":
        return create_rotating_circles(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Wave Distortion":
        return create_wave_distortion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Spiral Illusion":
        return create_spiral_illusion(width, height, frame, audio_features, intensity, random_seed)
    else:
        return create_illusory_tilt(width, height, frame, audio_features, intensity, random_seed)

def apply_colors(img, line_color, bg_color):
    """Applica i colori personalizzati all'immagine"""
    # Converte colori hex in RGB
    line_rgb = np.array([int(line_color[1:3], 16)/255, int(line_color[3:5], 16)/255, int(line_color[5:7], 16)/255])
    bg_rgb = np.array([int(bg_color[1:3], 16)/255, int(bg_color[3:5], 16)/255, int(bg_color[5:7], 16)/255])
    
    # Applica colori
    colored_img = np.zeros_like(img)
    mask = img[:,:,0] > 0.5  # Maschera per le linee/forme
    
    colored_img[mask] = line_rgb
    colored_img[~mask] = bg_rgb
    
    return colored_img

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
        n_frames = int(duration * fps)
        
        # Analizza audio
        st.info("🔍 Analisi delle frequenze audio...")
        audio_features = analyze_audio(tmp_audio.name, duration, fps)
        
        # Seed random per consistenza con variazione
        random_seed = random.randint(1, 10000)
        
        st.info(f"🌀 Generazione illusione: {illusion_type}")
        st.info(f"🎯 BPM rilevato: {audio_features['tempo']:.1f}")

        # Funzione animazione
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax.set_xlim(0, size[0])
        ax.set_ylim(0, size[1])
        ax.axis("off")
        fig.patch.set_facecolor('black')
        
        im = ax.imshow(np.zeros((size[1], size[0], 3)), aspect='equal')

        def animate(frame):
            # Genera frame dell'illusione
            illusion_img = generate_illusion_frame(
                illusion_type, size[0], size[1], frame, 
                audio_features, intensity, random_seed
            )
            
            # Applica colori personalizzati
            colored_img = apply_colors(illusion_img, line_color, bg_color)
            
            im.set_array(colored_img)
            return [im]

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Crea animazione
        anim = FuncAnimation(fig, animate, frames=n_frames, blit=True, interval=1000/fps)

        # Salva animazione come mp4 (senza audio)
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        
        def progress_callback(current_frame, total_frames):
            progress = current_frame / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Rendering frame {current_frame}/{total_frames} ({progress:.1%})")

        anim.save(tmp_video.name, fps=fps, extra_args=['-vcodec', 'libx264'])
        plt.close(fig)

        # Aggiungi audio + titolo con MoviePy
        st.info("🎬 Composizione finale con audio e titolo...")
        video_clip = mpy.VideoFileClip(tmp_video.name).set_audio(mpy.AudioFileClip(tmp_audio.name))

        # Overlay testo titolo se specificato
        if video_title.strip():
            txt_clip = mpy.TextClip(
                video_title, 
                fontsize=min(60, size[0]//20), 
                color='white',
                font='Arial-Bold'
            ).set_duration(video_clip.duration)

            if position == "Sopra":
                txt_clip = txt_clip.set_position(("center", "top"))
            elif position == "Sotto":
                txt_clip = txt_clip.set_position(("center", "bottom"))
            elif position == "Destra":
                txt_clip = txt_clip.set_position(("right", "center"))
            elif position == "Sinistra":
                txt_clip = txt_clip.set_position(("left", "center"))
            else:  # Centro
                txt_clip = txt_clip.set_position("center")

            final = mpy.CompositeVideoClip([video_clip, txt_clip])
        else:
            final = video_clip

        # Salvataggio finale
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        final.write_videofile(output_file.name, codec="libx264", audio_codec="aac", verbose=False)

        # Successo!
        st.success("✨ Video generato con successo!")
        
        # Mostra info finali
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata", f"{duration:.1f}s")
        with col2:
            st.metric("Risoluzione", f"{size[0]}x{size[1]}")
        with col3:
            st.metric("FPS", fps)

        # Download
        with open(output_file.name, "rb") as f:
            st.download_button(
                "📥 Scarica Video Illusorio", 
                f, 
                file_name=f"illusion_{illusion_type.lower().replace(' ', '_')}_output.mp4", 
                mime="video/mp4",
                type="primary"
            )

        # Cleanup
        os.remove(tmp_audio.name)
        os.remove(tmp_video.name)
        os.remove(output_file.name)

        progress_bar.empty()
        status_text.empty()

# Informazioni laterali
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ Come funziona")
    st.markdown("""
    **Mappatura Audio → Illusione:**
    - 🔊 **Bassi** → movimento globale
    - 🎵 **Medi** → deformazioni/vibrazioni  
    - 🎶 **Alti** → dettagli rapidi/flicker
    
    **Ogni run è unico** grazie al 
    componente random che varia:
    - Posizioni iniziali
    - Intensità effetti
    - Pattern di distorsione
    """)
    
    st.markdown("---")
    st.markdown("🎨 **Arte cinetica** sincronizzata")
    st.markdown("🌀 **Illusioni ottiche** generate")
    st.markdown("🎵 **Musica come DNA** visivo")
