import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# -------------------------------
# ILLUSIONI SCIENTIFICHE ACCURATE
# -------------------------------

def illusory_tilt_line_type(width, height, frame, audio_features, intensity):
    """Implementazione Line-type: triangoli e linee che creano inclinazione percettiva"""
    img = np.zeros((height, width))
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    
    # Parametri dinamici basati sull'audio
    triangle_size = int(30 + bass_val * 40 * intensity)
    rotation_angle = frame * 0.5 + mid_val * 45
    
    # Pattern di triangoli con illusione di inclinazione
    for y in range(0, height, triangle_size * 2):
        for x in range(0, width, triangle_size * 2):
            # Triangolo principale
            center_x, center_y = x + triangle_size, y + triangle_size
            
            # Creazione triangolo con rotazione dinamica
            angle_rad = np.radians(rotation_angle + (x+y)*0.1)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Vertici del triangolo ruotati
            vertices = np.array([
                [-triangle_size//2, -triangle_size//2],
                [triangle_size//2, -triangle_size//2],
                [0, triangle_size//2]
            ])
            
            rotated_vertices = np.array([
                [v[0]*cos_a - v[1]*sin_a + center_x, v[0]*sin_a + v[1]*cos_a + center_y]
                for v in vertices
            ]).astype(int)
            
            # Disegna triangolo se dentro i bounds
            if all(0 <= v[0] < width and 0 <= v[1] < height for v in rotated_vertices):
                # Usa skimage polygon per riempire il triangolo
                rr, cc = polygon(rotated_vertices[:, 1], rotated_vertices[:, 0], (height, width))
                img[rr, cc] = 1.0
    
    return img

def illusory_tilt_mixed_type(width, height, frame, audio_features, intensity):
    """Implementazione Mixed-type: combinazione linee e bordi per massimizzare effetto"""
    img = np.zeros((height, width))
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    # Linee diagonali con bordi contrastanti
    line_spacing = int(20 + bass_val * 30)
    line_width = max(1, int(2 + high_val * 8 * intensity))
    
    for i in range(0, width + height, line_spacing):
        # Linee diagonali principali
        x1, y1 = min(i, width-1), 0
        x2, y2 = 0, min(i, height-1)
        
        if x1 >= 0 and y2 >= 0:
            # Disegna linea usando skimage
            rr, cc = line(y1, x1, y2, x2)
            valid_idx = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid_idx], cc[valid_idx]] = 1.0
            
            # Bordi di contrasto per amplificare l'illusione
            if line_width > 2:
                for offset in range(-line_width//2, line_width//2 + 1):
                    rr_offset = rr + offset
                    cc_offset = cc + offset
                    valid_offset = (rr_offset >= 0) & (rr_offset < height) & (cc_offset >= 0) & (cc_offset < width)
                    img[rr_offset[valid_offset], cc_offset[valid_offset]] = 0.7
    
    # Aggiunta pattern perpendicolari per l'effetto mixed
    perpendicular_offset = frame * 2
    for i in range(perpendicular_offset, width + height, line_spacing * 2):
        x1, y1 = 0, min(i, height-1)
        x2, y2 = min(i, width-1), 0
        
        if y1 >= 0 and x2 >= 0:
            rr, cc = line(y1, x1, y2, x2)
            valid_idx = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid_idx], cc[valid_idx]] = 0.5
    
    return img

def illusory_tilt_edge_type(width, height, frame, audio_features, intensity):
    """Implementazione Edge-type: solo contrasti di bordo per illusione pura"""
    img = np.zeros((height, width))
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    # Griglia di quadrati con bordi contrastanti
    square_size = int(40 + mid_val * 60 * intensity)
    edge_width = max(1, int(1 + high_val * 4))
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            # Alterna riempimento per creare contrasto
            fill_value = 1.0 if (x//square_size + y//square_size + frame//10) % 2 == 0 else 0.0
            
            # Riempie il quadrato
            end_x, end_y = min(x + square_size, width), min(y + square_size, height)
            img[y:end_y, x:end_x] = fill_value
            
            # Bordi di contrasto per amplificare l'illusione di inclinazione
            if end_x < width:
                img[y:end_y, end_x-edge_width:end_x] = 1.0 - fill_value
            if end_y < height:
                img[end_y-edge_width:end_y, x:end_x] = 1.0 - fill_value
    
    return img

def illusory_motion_mather_line(width, height, frame, audio_features, intensity):
    """Implementazione Mather Line-type: cerchi con pattern radiali (effetto phi)"""
    img = np.zeros((height, width))
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    tempo_factor = audio_features["tempo"] / 120.0
    
    # Centri multipli per effetto phi
    centers = [(width//4, height//4), (3*width//4, height//4), 
               (width//4, 3*height//4), (3*width//4, 3*height//4)]
    
    for center_x, center_y in centers:
        if center_x < width and center_y < height:
            # Raggio dinamico basato sull'audio
            max_radius = min(width, height) // 6
            current_radius = int(max_radius * (0.5 + 0.5 * bass_val * intensity))
            
            # Pattern radiali per effetto phi
            num_spokes = int(8 + tempo_factor * 4)
            for i in range(num_spokes):
                angle = (2 * np.pi * i / num_spokes) + (frame * 0.1 * tempo_factor)
                
                # Linee radiali che creano movimento illusorio
                end_x = int(center_x + current_radius * np.cos(angle))
                end_y = int(center_y + current_radius * np.sin(angle))
                
                if 0 <= end_x < width and 0 <= end_y < height:
                    rr, cc = line(center_y, center_x, end_y, end_x)
                    valid_idx = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid_idx], cc[valid_idx]] = 1.0
            
            # Cerchi concentrici per amplificare l'effetto
            for r in range(current_radius//4, current_radius, current_radius//4):
                if r > 0:
                    rr, cc = circle(center_y, center_x, r, (height, width))
                    img[rr, cc] = 0.7
    
    return img

def illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity):
    """Implementazione Takeuchi Mixed-type: alternanza riempimento/vuoto"""
    img = np.zeros((height, width))
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    # Pattern di elementi che alternano riempimento/vuoto
    element_size = int(25 + mid_val * 35 * intensity)
    phase = frame * 0.2
    
    for y in range(0, height, element_size):
        for x in range(0, width, element_size):
            # Calcola fase locale per movimento illusorio
            local_phase = phase + (x + y) * 0.01
            
            # Alternanza basata sulla fase e high frequencies
            should_fill = (np.sin(local_phase) + high_val) > 0.5
            
            if should_fill:
                # Elemento pieno con bordo
                cv2.rectangle(img, (x, y), 
                            (min(x + element_size, width-1), min(y + element_size, height-1)), 
                            1.0, -1)
                # Bordo di contrasto
                cv2.rectangle(img, (x, y), 
                            (min(x + element_size, width-1), min(y + element_size, height-1)), 
                            0.0, 2)
            else:
                # Solo bordo per elemento vuoto
                cv2.rectangle(img, (x, y), 
                            (min(x + element_size, width-1), min(y + element_size, height-1)), 
                            0.8, 2)
    
    return img

def y_junctions_illusion(width, height, frame, audio_features, intensity):
    """Implementazione Y-Junctions: retinal slip simulation con movimento laterale"""
    img = np.zeros((height, width))
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    
    # Griglia a scacchiera base
    square_size = int(30 + bass_val * 40 * intensity)
    
    # Movimento laterale simulato (retinal slip)
    lateral_shift = int(frame * 0.5 * mid_val * intensity) % square_size
    
    for y in range(0, height, square_size):
        for x in range(-lateral_shift, width + square_size, square_size):
            # Scacchiera base
            fill = (x//square_size + y//square_size) % 2 == 0
            
            if x >= 0 and x < width:
                end_x, end_y = min(x + square_size, width), min(y + square_size, height)
                img[y:end_y, x:end_x] = 1.0 if fill else 0.0
                
                # Y-junctions agli incroci per amplificare l'illusione
                if x > 0 and y > 0:
                    junction_x, junction_y = x, y
                    
                    # Disegna Y-junction
                    cv2.line(img, (junction_x, junction_y-5), (junction_x, junction_y+5), 0.5, 2)
                    cv2.line(img, (junction_x-5, junction_y), (junction_x+5, junction_y), 0.5, 2)
                    cv2.line(img, (junction_x-3, junction_y-3), (junction_x+3, junction_y+3), 0.5, 2)
    
    return img

def drifting_spines_illusion(width, height, frame, audio_features, intensity):
    """Implementazione Drifting Spines: elementi ripetuti che sembrano scivolare"""
    img = np.zeros((height, width))
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    tempo_factor = audio_features["tempo"] / 120.0
    
    # Direzione e velocità del drift
    drift_speed = tempo_factor * intensity
    drift_offset = (frame * drift_speed) % 100
    
    # Pattern di "spine" o frecce ripetute
    spine_spacing = int(40 + high_val * 30)
    spine_length = int(20 + high_val * 25 * intensity)
    
    for y in range(0, height, spine_spacing):
        for x in range(int(-drift_offset), width + spine_spacing, spine_spacing):
            if x >= 0 and x < width:
                # Disegna "spina" o freccia
                center_x, center_y = x, y + spine_spacing//2
                
                if center_y < height:
                    # Corpo della spina (linea verticale)
                    start_y = max(0, center_y - spine_length//2)
                    end_y = min(height-1, center_y + spine_length//2)
                    
                    if start_y < end_y:
                        rr, cc = line(start_y, center_x, end_y, center_x)
                        valid_idx = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                        img[rr[valid_idx], cc[valid_idx]] = 1.0
                    
                    # Punte della freccia per amplificare il drift
                    arrow_size = spine_length // 4
                    if arrow_size > 0:
                        # Freccia sinistra
                        arrow_x1 = max(0, center_x - arrow_size)
                        arrow_y1 = max(0, center_y - spine_length//2 + arrow_size)
                        if arrow_x1 < width and arrow_y1 < height:
                            rr, cc = line(center_y - spine_length//2, center_x, arrow_y1, arrow_x1)
                            valid_idx = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                            img[rr[valid_idx], cc[valid_idx]] = 1.0
                        
                        # Freccia destra
                        arrow_x2 = min(width-1, center_x + arrow_size)
                        if arrow_x2 >= 0 and arrow_y1 < height:
                            rr, cc = line(center_y - spine_length//2, center_x, arrow_y1, arrow_x2)
                            valid_idx = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                            img[rr[valid_idx], cc[valid_idx]] = 1.0
    
    # Pattern orizzontale per amplificare l'effetto di drift
    for x in range(0, width, spine_spacing//2):
        horizontal_y = int(height//2 + 50 * np.sin(x * 0.05 + drift_offset * 0.1))
        if 0 <= horizontal_y < height:
            radius = max(1, int(3 * high_val * intensity))
            rr, cc = circle(horizontal_y, x, radius, (height, width))
            img[rr, cc] = 0.7
    
    return img

def spiral_illusion(width, height, frame, audio_features, intensity):
    """Implementazione Spiral Illusion: spirale che sembra espandersi/contrarsi"""
    img = np.zeros((height, width))
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    
    center_x, center_y = width // 2, height // 2
    max_radius = min(width, height) // 2
    
    # Parametri della spirale dinamici
    spiral_tightness = 0.1 + bass_val * 0.2 * intensity
    rotation_speed = frame * 0.05 + mid_val * 0.1
    
    # Disegna spirale con multiple braccia
    num_arms = 3
    for arm in range(num_arms):
        arm_offset = (2 * np.pi * arm) / num_arms
        
        for r in range(5, max_radius, 3):
            angle = r * spiral_tightness + rotation_speed + arm_offset
            
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            
            if 0 <= x < width and 0 <= y < height:
                # Intensità variabile per effetto di profondità
                intensity_val = 0.8 + 0.2 * np.sin(r * 0.1 + rotation_speed)
                
                # Disegna punto della spirale usando cerchio
                radius = max(1, int(2 + bass_val * 3))
                rr, cc = circle(y, x, radius, (height, width))
                img[rr, cc] = intensity_val
    
    return img

def generate_illusion_frame(width, height, frame, audio_features, intensity, illusion_type, seed):
    """Genera frame con l'illusione specificata"""
    np.random.seed(seed + frame)
    
    # Transizione dinamica tra sottotipi basata sul BPM
    tempo_factor = audio_features["tempo"] / 120.0
    subtype_cycle = int(frame / (30 * tempo_factor)) % 3  # Cambia ogni ~1 secondo a 120 BPM
    
    if illusion_type == "Illusory Tilt":
        if subtype_cycle == 0:
            img = illusory_tilt_line_type(width, height, frame, audio_features, intensity)
        elif subtype_cycle == 1:
            img = illusory_tilt_mixed_type(width, height, frame, audio_features, intensity)
        else:
            img = illusory_tilt_edge_type(width, height, frame, audio_features, intensity)
    
    elif illusion_type == "Illusory Motion":
        if subtype_cycle == 0:
            img = illusory_motion_mather_line(width, height, frame, audio_features, intensity)
        else:
            img = illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity)
    
    elif illusion_type == "Y-Junctions":
        img = y_junctions_illusion(width, height, frame, audio_features, intensity)
    
    elif illusion_type == "Drifting Spines":
        img = drifting_spines_illusion(width, height, frame, audio_features, intensity)
    
    else:  # Spiral Illusion
        img = spiral_illusion(width, height, frame, audio_features, intensity)
    
    # Converti a 3 canali per la colorazione
    return np.stack([img, img, img], axis=2)

def apply_colors(img, line_color, bg_color):
    """Applica i colori personalizzati"""
    line_rgb = np.array([int(line_color[1:3],16)/255, int(line_color[3:5],16)/255, int(line_color[5:7],16)/255])
    bg_rgb = np.array([int(bg_color[1:3],16)/255, int(bg_color[3:5],16)/255, int(bg_color[5:7],16)/255])
    
    colored = np.zeros_like(img)
    
    # Applica colorazione basata sull'intensità
    for i in range(3):  # RGB channels
        colored[:,:,i] = img[:,:,0] * line_rgb[i] + (1 - img[:,:,0]) * bg_rgb[i]
    
    return colored

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
        
        # Salvataggio ottimizzato
        Writer = plt.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Loop507'), bitrate=1800)
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
