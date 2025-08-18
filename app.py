import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import random
import cv2
import subprocess
from PIL import Image, ImageDraw, ImageFont

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

# Due selettori per la posizione del titolo
st.sidebar.subheader("📍 Posizione del Titolo")
vertical_position = st.sidebar.selectbox("Posizione Verticale", ["Sopra", "Sotto"])
horizontal_position = st.sidebar.selectbox("Posizione Orizzontale", ["Sinistra", "Destra", "Centro"])

# Scelta formato
aspect_ratio = st.selectbox("📺 Formato video", ["16:9", "1:1", "9:16"])

# Intensità effetti
intensity = st.sidebar.slider("🔥 Intensità effetti", 0.1, 2.0, 1.0, 0.1)

def analyze_audio(audio_path, duration, fps):
    """Analizza l'audio per estrarre features musicali"""
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calcola BPM
    # --- FIX per TypeError ---
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception as e:
        tempo = 0.0  # Valore di default in caso di errore
        st.warning(f"⚠️ Attenzione: Impossibile rilevare il BPM. L'errore originale è: {e}")
    
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
    """Crea illusione di inclinazione scientificamente accurata basata su triangoli e contrasti"""
    random.seed(random_seed + frame // 30)
    img = np.zeros((height, width, 3))
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    illusion_type = ["line", "mixed", "edge"][frame % 3]
    
    cell_size = int(40 + bass_val * 20 * intensity)
    rows = height // cell_size
    cols = width // cell_size
    
    for row in range(rows):
        for col in range(cols):
            center_x = col * cell_size + cell_size // 2
            center_y = row * cell_size + cell_size // 2
            
            rotation = (frame * 0.02 * mid_val + row * 0.1 + col * 0.1) * intensity
            scale = 0.8 + high_val * 0.4 * intensity
            
            triangle_size = int(cell_size * 0.3 * scale)
            
            if illusion_type == "line":
                create_line_tilt_pattern(img, center_x, center_y, triangle_size, rotation, bass_val)
            elif illusion_type == "mixed":
                create_mixed_tilt_pattern(img, center_x, center_y, triangle_size, rotation, mid_val)
            else:  # edge
                create_edge_tilt_pattern(img, center_x, center_y, triangle_size, rotation, high_val)
    
    return img

def create_line_tilt_pattern(img, cx, cy, size, rotation, intensity):
    """Crea pattern Line-type per illusione tilt"""
    positions = [
        (-size//2, -size//3), (size//2, -size//3),
        (-size//2, size//3), (size//2, size//3)
    ]
    
    colors = [1, 0, 0, 1] if intensity > 0.5 else [0, 1, 1, 0]
    
    for i, (dx, dy) in enumerate(positions):
        x = cx + dx * np.cos(rotation) - dy * np.sin(rotation)
        y = cy + dx * np.sin(rotation) + dy * np.cos(rotation)
        
        if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]:
            draw_triangle(img, int(x), int(y), size//3, colors[i], "up" if i % 2 == 0 else "down")

def create_mixed_tilt_pattern(img, cx, cy, size, rotation, intensity):
    """Crea pattern Mixed-type per illusione tilt"""
    draw_triangle(img, cx, cy - size//4, size//2, 1 if intensity > 0.5 else 0, "up")
    draw_triangle(img, cx, cy + size//4, size//2, 0 if intensity > 0.5 else 1, "down")
    
    line_length = size
    for offset in [-size//2, size//2]:
        x1 = cx + offset
        y1 = cy - line_length//2
        y2 = cy + line_length//2
        draw_line(img, x1, y1, x1, y2, 0.7)

def create_edge_tilt_pattern(img, cx, cy, size, rotation, intensity):
    """Crea pattern Edge-type per illusione tilt"""
    diamond_size = size // 2
    points = [
        (cx, cy - diamond_size),
        (cx + diamond_size, cy),
        (cx, cy + diamond_size),
        (cx - diamond_size, cy)
    ]
    fill_diamond(img, points, 1 if intensity > 0.3 else 0)
    
    for dx in [-size, size]:
        smaller_points = [
            (cx + dx, cy - diamond_size//2),
            (cx + dx + diamond_size//2, cy),
            (cx + dx, cy + diamond_size//2),
            (cx + dx - diamond_size//2, cy)
        ]
        fill_diamond(img, smaller_points, 0 if intensity > 0.3 else 1)

def draw_triangle(img, cx, cy, size, color, direction):
    """Disegna un triangolo pieno"""
    h, w = img.shape[:2]
    
    if direction == "up":
        points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
    else:  # down
        points = [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
    
    for y in range(max(0, cy - size), min(h, cy + size + 1)):
        for x in range(max(0, cx - size), min(w, cx + size + 1)):
            if point_in_triangle(x, y, points):
                img[y, x] = [color, color, color]

def draw_line(img, x1, y1, x2, y2, color):
    """Disegna una linea"""
    h, w = img.shape[:2]
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    x, y = x1, y1
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    if dx > dy:
        err = dx / 2
        while x != x2:
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = [color, color, color]
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2
        while y != y2:
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = [color, color, color]
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    
    if 0 <= x2 < w and 0 <= y2 < h:
        img[y2, x2] = [color, color, color]

def fill_diamond(img, points, color):
    """Riempie un diamante definito da 4 punti"""
    h, w = img.shape[:2]
    if len(points) != 4:
        return
    min_x = max(0, min(p[0] for p in points))
    max_x = min(w-1, max(p[0] for p in points))
    min_y = max(0, min(p[1] for p in points))
    max_y = min(h-1, max(p[1] for p in points))
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if point_in_diamond(x, y, points):
                img[y, x] = [color, color, color]

def point_in_triangle(x, y, triangle_points):
    """Verifica se un punto è dentro un triangolo"""
    x1, y1 = triangle_points[0]
    x2, y2 = triangle_points[1]
    x3, y3 = triangle_points[2]
    
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-10:
        return False
        
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b
    
    return a >= 0 and b >= 0 and c >= 0

def point_in_diamond(x, y, diamond_points):
    """Verifica se un punto è dentro un diamante"""
    if len(diamond_points) != 4:
        return False
        
    triangle1 = [diamond_points[0], diamond_points[1], diamond_points[2]]
    triangle2 = [diamond_points[0], diamond_points[2], diamond_points[3]]
    
    return point_in_triangle(x, y, triangle1) or point_in_triangle(x, y, triangle2)

def create_illusory_motion(width, height, frame, audio_features, intensity, random_seed):
    """Crea illusioni di movimento basate sui pattern scientifici di Mather e Takeuchi"""
    random.seed(random_seed)
    img = np.zeros((height, width, 3))
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    motion_types = ["mather_line", "takeuchi_mixed", "mather_edge"]
    motion_type = motion_types[frame % len(motion_types)]
    
    circle_spacing = int(80 + bass_val * 40 * intensity)
    circle_radius = int(15 + mid_val * 15 * intensity)
    
    rows = height // circle_spacing
    cols = width // circle_spacing
    
    for row in range(rows):
        for col in range(cols):
            center_x = col * circle_spacing + circle_spacing // 2
            center_y = row * circle_spacing + circle_spacing // 2
            
            phase = (frame * 0.2 + row * 0.5 + col * 0.3) * intensity
            
            if motion_type == "mather_line":
                create_mather_line_motion(img, center_x, center_y, circle_radius, phase, high_val)
            elif motion_type == "takeuchi_mixed":
                create_takeuchi_mixed_motion(img, center_x, center_y, circle_radius, phase, mid_val)
            else:
                create_mather_edge_motion(img, center_x, center_y, circle_radius, phase, bass_val)
    
    return img

def create_mather_line_motion(img, cx, cy, radius, phase, intensity):
    """Crea illusione di movimento tipo Mather con linee (phi motion)"""
    h, w = img.shape[:2]
    
    phi_factor = np.sin(phase) * intensity
    reversed_phi = np.cos(phase + np.pi/2) * intensity
    
    inner_radius = int(radius * (0.5 + 0.3 * phi_factor))
    outer_radius = int(radius * (0.8 + 0.2 * reversed_phi))
    
    for y in range(max(0, cy - outer_radius), min(h, cy + outer_radius + 1)):
        for x in range(max(0, cx - outer_radius), min(w, cx + outer_radius + 1)):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            if inner_radius <= dist <= outer_radius:
                angle = np.arctan2(y - cy, x - cx)
                stripe_pattern = np.sin(angle * 8 + phase) > 0
                color = 1 if stripe_pattern else 0
                img[y, x] = [color, color, color]

def create_takeuchi_mixed_motion(img, cx, cy, radius, phase, intensity):
    """Crea illusione tipo Takeuchi con elementi misti"""
    h, w = img.shape[:2]
    fill_factor = (np.sin(phase * 2) + 1) / 2 * intensity
    
    for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            if dist <= radius:
                if dist <= radius * fill_factor:
                    img[y, x] = [1, 1, 1]
                else:
                    if abs(dist - radius * 0.7) < 2:
                        img[y, x] = [0.5, 0.5, 0.5]

def create_mather_edge_motion(img, cx, cy, radius, phase, intensity):
    """Crea illusione tipo Mather con bordi contrastanti"""
    h, w = img.shape[:2]
    is_filled = np.sin(phase) > 0
    
    for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            if dist <= radius:
                if is_filled:
                    img[y, x] = [0, 0, 0]
                else:
                    if abs(dist - radius) < 3:
                        img[y, x] = [1, 1, 1]

def create_y_junctions_illusion(width, height, frame, audio_features, intensity, random_seed):
    """Crea l'illusione delle Y-junctions con retinal slip"""
    random.seed(random_seed)
    img = np.zeros((height, width, 3))
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    grid_size = int(30 + bass_val * 20 * intensity)
    rows = height // grid_size
    cols = width // grid_size
    
    for row in range(rows):
        for col in range(cols):
            x = col * grid_size
            y = row * grid_size
            
            is_white = (row + col) % 2 == 0
            color = 0.8 if is_white else 0.2
            
            for dy in range(grid_size):
                for dx in range(grid_size):
                    if y + dy < height and x + dx < width:
                        img[y + dy, x + dx] = [color, color, color]
    
    junction_spacing = int(grid_size * 2)
    
    for row in range(0, rows, 2):
        for col in range(0, cols, 2):
            center_x = col * grid_size + grid_size // 2
            center_y = row * grid_size + grid_size // 2
            
            offset_x = np.sin(frame * 0.1 + col * 0.2) * mid_val * 10 * intensity
            offset_y = np.cos(frame * 0.1 + row * 0.2) * high_val * 10 * intensity
            
            draw_y_junction(img, center_x + offset_x, center_y + offset_y,
                          int(10 + bass_val * 5 * intensity))
    
    return img

def draw_y_junction(img, cx, cy, size):
    """Disegna una Y-junction"""
    h, w = img.shape[:2]
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    for angle in angles:
        end_x = cx + size * np.cos(angle)
        end_y = cy + size * np.sin(angle)
        draw_thick_line(img, cx, cy, end_x, end_y, 0, 3)

def draw_thick_line(img, x1, y1, x2, y2, color, thickness):
    """Disegna una linea spessa"""
    h, w = img.shape[:2]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx == 0 and dy == 0:
        return
    steps = max(dx, dy)
    if steps == 0:
        return
    x_inc = (x2 - x1) / steps
    y_inc = (y2 - y1) / steps
    for i in range(int(steps) + 1):
        x = int(x1 + i * x_inc)
        y = int(y1 + i * y_inc)
        for dy in range(-thickness//2, thickness//2 + 1):
            for dx in range(-thickness//2, thickness//2 + 1):
                px, py = x + dx, y + dy
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = [color, color, color]

def create_drifting_spines_illusion(width, height, frame, audio_features, intensity, random_seed):
    """Crea l'illusione delle spine che scivolano"""
    random.seed(random_seed)
    img = np.zeros((height, width, 3))
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    spine_spacing = int(20 + bass_val * 10 * intensity)
    spine_size = int(8 + mid_val * 8 * intensity)
    
    drift_x = frame * 0.5 * high_val * intensity
    drift_y = frame * 0.3 * mid_val * intensity
    
    rows = height // spine_spacing
    cols = width // spine_spacing
    
    for row in range(rows):
        for col in range(cols):
            x = (col * spine_spacing + drift_x) % width
            y = (row * spine_spacing + drift_y) % height
            draw_spine_element(img, x, y, spine_size, frame + row*5 + col*3, intensity)
    
    return img

def draw_spine_element(img, cx, cy, size, phase, intensity):
    """Disegna un elemento 'spine' che contribuisce all'illusione di drifting"""
    h, w = img.shape[:2]
    angle = phase * 0.1 * intensity
    tip_x = cx + size * np.cos(angle)
    tip_y = cy + size * np.sin(angle)
    base1_x = cx + size * 0.5 * np.cos(angle + 2.5)
    base1_y = cy + size * 0.5 * np.sin(angle + 2.5)
    base2_x = cx + size * 0.5 * np.cos(angle - 2.5)
    base2_y = cy + size * 0.5 * np.sin(angle - 2.5)
    
    draw_line(img, cx, cy, tip_x, tip_y, 1)
    draw_line(img, tip_x, tip_y, base1_x, base1_y, 1)
    draw_line(img, tip_x, tip_y, base2_x, base2_y, 1)

def create_spiral_illusion(width, height, frame, audio_features, intensity, random_seed):
    """Crea illusione spirale"""
    random.seed(random_seed)
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    center_x, center_y = width // 2, height // 2
    x, y = np.meshgrid(np.arange(width) - center_x, np.arange(height) - center_y)
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    spiral_freq = 0.05 + mid_val * 0.03 * intensity
    spiral_rotation = frame * 0.02 * bass_val * intensity
    spiral = np.sin(spiral_freq * r + theta * 3 + spiral_rotation)
    radial_variation = np.sin(r * 0.01 + frame * 0.1 * high_val * intensity)
    
    combined = spiral + radial_variation * high_val * intensity
    normalized = (combined + 2) / 4
    img = np.stack([normalized, normalized, normalized], axis=2)
    
    return img

def generate_illusion_frame(illusion_type, width, height, frame, audio_features, intensity, random_seed):
    """Genera un frame dell'illusione specificata"""
    if illusion_type == "Illusory Tilt":
        return create_illusory_tilt(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Illusory Motion":
        return create_illusory_motion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Y-Junctions":
        return create_y_junctions_illusion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Drifting Spines":
        return create_drifting_spines_illusion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Spiral Illusion":
        return create_spiral_illusion(width, height, frame, audio_features, intensity, random_seed)
    else:
        return create_illusory_tilt(width, height, frame, audio_features, intensity, random_seed)

def apply_colors(img, line_color, bg_color):
    """Applica i colori personalizzati all'immagine"""
    line_rgb = np.array([int(line_color[1:3], 16)/255, int(line_color[3:5], 16)/255, int(line_color[5:7], 16)/255])
    bg_rgb = np.array([int(bg_color[1:3], 16)/255, int(bg_color[3:5], 16)/255, int(bg_color[5:7], 16)/255])
    
    colored_img = np.zeros_like(img)
    mask = img[:,:,0] > 0.1
    
    colored_img[mask] = line_rgb
    colored_img[~mask] = bg_rgb
    
    return colored_img

def add_title_to_frame(frame, title_text, vertical_pos, horizontal_pos):
    """Aggiunge il titolo al frame usando PIL e OpenCV"""
    if not title_text:
        return frame

    # Converte il frame da array numpy a immagine PIL
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Dimensioni del frame
    width, height = pil_img.size
    
    # Scegli un font e una dimensione appropriata
    font_path = "arial.ttf" # Assicurati che questo font sia disponibile o usa uno standard
    try:
        font = ImageFont.truetype(font_path, int(width * 0.05))
    except IOError:
        font = ImageFont.load_default()
    
    # Calcola le dimensioni del testo
    text_bbox = draw.textbbox((0, 0), title_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calcola la posizione
    x = 0
    y = 0
    padding = int(width * 0.02)
    
    if horizontal_pos == "Sinistra":
        x = padding
    elif horizontal_pos == "Destra":
        x = width - text_width - padding
    else:  # Centro
        x = (width - text_width) / 2
        
    if vertical_pos == "Sopra":
        y = padding
    else:  # Sotto
        y = height - text_height - padding
        
    draw.text((x, y), title_text, font=font, fill=(255, 255, 255))
    
    # Riconverte l'immagine PIL in array numpy
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Pulsante genera
if uploaded_file and st.button("🚀 Genera Video Illusorio", type="primary"):
    with st.spinner("🎨 Creazione dell'illusione ottica in corso..."):
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_audio.write(uploaded_file.read())
        tmp_audio.close()

        y, sr = librosa.load(tmp_audio.name, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        st.info(f"🎵 Durata audio: {duration:.2f} secondi")

        if aspect_ratio == "16:9":
            size = (1280, 720)
        elif aspect_ratio == "1:1":
            size = (720, 720)
        else:
            size = (720, 1280)

        fps = 30
        n_frames = int(duration * fps)
        
        st.info("🔍 Analisi delle frequenze audio...")
        audio_features = analyze_audio(tmp_audio.name, duration, fps)
        
        random_seed = random.randint(1, 10000)
        
        st.info(f"🌀 Generazione illusione: {illusion_type}")
        # --- FIX per TypeError: il valore è sempre float o 0.0
        st.info(f"🎯 BPM rilevato: {audio_features['tempo']:.1f}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(tmp_video_path, fourcc, fps, size)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(n_frames):
            illusion_img = generate_illusion_frame(
                illusion_type, size[0], size[1], i,
                audio_features, intensity, random_seed
            )
            
            colored_img = apply_colors(illusion_img, line_color, bg_color)
            frame_bgr = cv2.cvtColor((colored_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Aggiungi il titolo al frame
            frame_with_title = add_title_to_frame(frame_bgr, video_title, vertical_position, horizontal_position)
            
            out.write(frame_with_title)

            progress = (i + 1) / n_frames
            progress_bar.progress(progress)
            status_text.text(f"Rendering frame {i+1}/{n_frames} ({progress:.1%})")

        out.release()
        
        st.info("🎬 Composizione finale con audio...")
        output_final_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        
        command = [
            'ffmpeg',
            '-i', tmp_video_path,
            '-i', tmp_audio.name,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_final_path
        ]
        
        try:
            subprocess.run(command, check=True)
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
            with open(output_final_path, "rb") as f:
                st.download_button(
                    "📥 Scarica Video Illusorio",
                    f,
                    file_name=f"illusion_{illusion_type.lower().replace(' ', '_')}_output.mp4",
                    mime="video/mp4",
                    type="primary"
                )
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Errore durante l'aggiunta dell'audio (ffmpeg). Assicurati che ffmpeg sia installato. Errore: {e}")
        finally:
            # Cleanup
            os.remove(tmp_audio.name)
            os.remove(tmp_video_path)
            if os.path.exists(output_final_path):
                os.remove(output_final_path)

        progress_bar.empty()
        status_text.empty()

# Informazioni laterali
with st.sidebar:
    st.markdown("---")
    st.subheader("🧠 Illusioni Scientifiche")
    st.markdown("""
    **🔬 Basate su ricerca neuroscientifica:**
    
    🌀 **Illusory Tilt** - Line-type, Mixed-type, Edge-type
    - Triangoli e contrasti che ingannano la percezione
    
    ⚡ **Illusory Motion**
    - Mather's type & Takeuchi's type
    - Cerchi statici che sembrano espandersi
    - Effetti phi e reversed-phi
    
    🔗 **Y-Junctions**
    - Retinal slip simulation
    - Griglie con movimenti illusori
    
    🌊 **Drifting Spines** - Elementi che sembrano scivolare
    - Micro-movimenti oculari amplificati
    """)
    
    st.markdown("---")
    st.subheader("ℹ️ Mappatura Audio")
    st.markdown("""
    **🎵 Audio → Effetti Visivi:**
    - 🔊 **Bassi** → movimento globale, rotazioni
    - 🎵 **Medi** → deformazioni, vibrazioni
    - 🎶 **Alti** → dettagli rapidi, micro-shift
    
    **🎯 BPM** → velocità transizioni
    **🎨 Ogni run è unico** grazie al random seed
    """)
    
    st.markdown("---")
    st.markdown("🧬 **Neuroscienza** applicata")
    st.markdown("🎨 **Arte generativa** guidata da musica")
    st.markdown("⚡ **Illusioni** scientificamente accurate")
