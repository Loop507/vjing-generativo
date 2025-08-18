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

def create_illusory_tilt(width, height, frame, audio_features, intensity, random_seed):
    """Crea illusione di inclinazione scientificamente accurata basata su triangoli e contrasti"""
    random.seed(random_seed + frame // 30)  # Cambia pattern ogni secondo
    img = np.zeros((height, width, 3))
    
    # Parametri guidati dall'audio
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    # Scelta dinamica del tipo di illusione (Line, Mixed, Edge)
    illusion_type = ["line", "mixed", "edge"][frame % 3]
    
    # Dimensione e spaziatura delle celle
    cell_size = int(40 + bass_val * 20 * intensity)
    rows = height // cell_size
    cols = width // cell_size
    
    for row in range(rows):
        for col in range(cols):
            center_x = col * cell_size + cell_size // 2
            center_y = row * cell_size + cell_size // 2
            
            # Variazione dinamica basata su audio
            rotation = (frame * 0.02 * mid_val + row * 0.1 + col * 0.1) * intensity
            scale = 0.8 + high_val * 0.4 * intensity
            
            # Crea pattern di triangoli per illusione tilt
            triangle_size = int(cell_size * 0.3 * scale)
            
            if illusion_type == "line":
                # Line-type: solo linee che creano illusione di tilt
                create_line_tilt_pattern(img, center_x, center_y, triangle_size, rotation, bass_val)
            elif illusion_type == "mixed":
                # Mixed-type: linee + bordi
                create_mixed_tilt_pattern(img, center_x, center_y, triangle_size, rotation, mid_val)
            else:  # edge
                # Edge-type: solo bordi contrastanti
                create_edge_tilt_pattern(img, center_x, center_y, triangle_size, rotation, high_val)
    
    return img

def create_line_tilt_pattern(img, cx, cy, size, rotation, intensity):
    """Crea pattern Line-type per illusione tilt"""
    # Triangoli bianchi e neri alternati che creano l'illusione di inclinazione
    positions = [
        (-size//2, -size//3), (size//2, -size//3),  # triangoli superiori
        (-size//2, size//3), (size//2, size//3)     # triangoli inferiori
    ]
    
    colors = [1, 0, 0, 1] if intensity > 0.5 else [0, 1, 1, 0]  # Alterna basato sull'intensità
    
    for i, (dx, dy) in enumerate(positions):
        # Applica rotazione
        x = cx + dx * np.cos(rotation) - dy * np.sin(rotation)
        y = cy + dx * np.sin(rotation) + dy * np.cos(rotation)
        
        if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]:
            draw_triangle(img, int(x), int(y), size//3, colors[i], "up" if i % 2 == 0 else "down")

def create_mixed_tilt_pattern(img, cx, cy, size, rotation, intensity):
    """Crea pattern Mixed-type per illusione tilt"""
    # Combinazione di linee e forme piene
    # Triangoli centrali
    draw_triangle(img, cx, cy - size//4, size//2, 1 if intensity > 0.5 else 0, "up")
    draw_triangle(img, cx, cy + size//4, size//2, 0 if intensity > 0.5 else 1, "down")
    
    # Linee laterali per enfatizzare l'illusione
    line_length = size
    for offset in [-size//2, size//2]:
        x1 = cx + offset
        y1 = cy - line_length//2
        y2 = cy + line_length//2
        draw_line(img, x1, y1, x1, y2, 0.7)

def create_edge_tilt_pattern(img, cx, cy, size, rotation, intensity):
    """Crea pattern Edge-type per illusione tilt"""
    # Solo bordi contrastanti per massimizzare l'effetto tilt
    # Crea diamanti alternati
    diamond_size = size // 2
    
    # Diamante centrale
    points = [
        (cx, cy - diamond_size),      # top
        (cx + diamond_size, cy),      # right  
        (cx, cy + diamond_size),      # bottom
        (cx - diamond_size, cy)       # left
    ]
    
    fill_diamond(img, points, 1 if intensity > 0.3 else 0)
    
    # Diamanti laterali più piccoli per contrasto
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
    
    # Riempie il triangolo
    for y in range(max(0, cy - size), min(h, cy + size + 1)):
        for x in range(max(0, cx - size), min(w, cx + size + 1)):
            if point_in_triangle(x, y, points):
                img[y, x] = [color, color, color]

def draw_line(img, x1, y1, x2, y2, color):
    """Disegna una linea"""
    h, w = img.shape[:2]
    
    # Bresenham line algorithm semplificato
    points = []
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
        
    # Trova bounding box
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
    
    # Calcola le coordinate baricentriche
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-10:
        return False
        
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b
    
    return a >= 0 and b >= 0 and c >= 0

def point_in_diamond(x, y, diamond_points):
    """Verifica se un punto è dentro un diamante"""
    # Un diamante può essere visto come due triangoli
    if len(diamond_points) != 4:
        return False
        
    # Primo triangolo: punti 0, 1, 2
    triangle1 = [diamond_points[0], diamond_points[1], diamond_points[2]]
    # Secondo triangolo: punti 0, 2, 3
    triangle2 = [diamond_points[0], diamond_points[2], diamond_points[3]]
    
    return point_in_triangle(x, y, triangle1) or point_in_triangle(x, y, triangle2)

def create_illusory_motion(width, height, frame, audio_features, intensity, random_seed):
    """Crea illusioni di movimento basate sui pattern scientifici di Mather e Takeuchi"""
    random.seed(random_seed)
    img = np.zeros((height, width, 3))
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    # Scelta dinamica del tipo basato sull'audio
    motion_types = ["mather_line", "takeuchi_mixed", "mather_edge"]
    motion_type = motion_types[frame % len(motion_types)]
    
    # Griglia di cerchi per l'illusione
    circle_spacing = int(80 + bass_val * 40 * intensity)
    circle_radius = int(15 + mid_val * 15 * intensity)
    
    rows = height // circle_spacing
    cols = width // circle_spacing
    
    for row in range(rows):
        for col in range(cols):
            center_x = col * circle_spacing + circle_spacing // 2
            center_y = row * circle_spacing + circle_spacing // 2
            
            # Fase per l'animazione phi
            phase = (frame * 0.2 + row * 0.5 + col * 0.3) * intensity
            
            if motion_type == "mather_line":
                create_mather_line_motion(img, center_x, center_y, circle_radius, phase, high_val)
            elif motion_type == "takeuchi_mixed":
                create_takeuchi_mixed_motion(img, center_x, center_y, circle_radius, phase, mid_val)
            else:  # mather_edge
                create_mather_edge_motion(img, center_x, center_y, circle_radius, phase, bass_val)
    
    return img

def create_mather_line_motion(img, cx, cy, radius, phase, intensity):
    """Crea illusione di movimento tipo Mather con linee (phi motion)"""
    h, w = img.shape[:2]
    
    # Pattern alternato che crea l'illusione di espansione
    phi_factor = np.sin(phase) * intensity
    reversed_phi = np.cos(phase + np.pi/2) * intensity
    
    # Cerchio interno che sembra espandersi
    inner_radius = int(radius * (0.5 + 0.3 * phi_factor))
    outer_radius = int(radius * (0.8 + 0.2 * reversed_phi))
    
    # Disegna cerchio con bordi alternati
    for y in range(max(0, cy - outer_radius), min(h, cy + outer_radius + 1)):
        for x in range(max(0, cx - outer_radius), min(w, cx + outer_radius + 1)):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            if inner_radius <= dist <= outer_radius:
                # Pattern alternato che crea l'effetto phi
                angle = np.arctan2(y - cy, x - cx)
                stripe_pattern = np.sin(angle * 8 + phase) > 0
                color = 1 if stripe_pattern else 0
                img[y, x] = [color, color, color]

def create_takeuchi_mixed_motion(img, cx, cy, radius, phase, intensity):
    """Crea illusione tipo Takeuchi con elementi misti"""
    h, w = img.shape[:2]
    
    # Cerchio che alterna riempimento (fenomeni fenomenali)
    fill_factor = (np.sin(phase * 2) + 1) / 2 * intensity
    
    # Cerchio principale
    for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            if dist <= radius:
                # Riempimento alternato basato sulla fase
                if dist <= radius * fill_factor:
                    img[y, x] = [1, 1, 1]  # Bianco
                else:
                    # Pattern di bordo per enfatizzare il movimento
                    if abs(dist - radius * 0.7) < 2:
                        img[y, x] = [0.5, 0.5, 0.5]  # Grigio

def create_mather_edge_motion(img, cx, cy, radius, phase, intensity):
    """Crea illusione tipo Mather con bordi contrastanti"""
    h, w = img.shape[:2]
    
    # Alternanza tra cerchio pieno e vuoto
    is_filled = np.sin(phase) > 0
    
    for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            if dist <= radius:
                if is_filled:
                    # Cerchio nero pieno
                    img[y, x] = [0, 0, 0]
                else:
                    # Solo il bordo
                    if abs(dist - radius) < 3:
                        img[y, x] = [1, 1, 1]
def create_y_junctions_illusion(width, height, frame, audio_features, intensity, random_seed):
    """Crea l'illusione delle Y-junctions con retinal slip"""
    random.seed(random_seed)
    img = np.zeros((height, width, 3))
    
    bass_val = audio_features['bass'][frame % len(audio_features['bass'])]
    mid_val = audio_features['mid'][frame % len(audio_features['mid'])]
    high_val = audio_features['high'][frame % len(audio_features['high'])]
    
    # Griglia di base
    grid_size = int(30 + bass_val * 20 * intensity)
    rows = height // grid_size
    cols = width // grid_size
    
    # Crea griglia a scacchiera
    for row in range(rows):
        for col in range(cols):
            x = col * grid_size
            y = row * grid_size
            
            # Pattern a scacchiera
            is_white = (row + col) % 2 == 0
            color = 0.8 if is_white else 0.2
            
            for dy in range(grid_size):
                for dx in range(grid_size):
                    if y + dy < height and x + dx < width:
                        img[y + dy, x + dx] = [color, color, color]
    
    # Aggiungi Y-junctions che creano il movimento illusorio
    junction_spacing = int(grid_size * 2)
    
    for row in range(0, rows, 2):
        for col in range(0, cols, 2):
            center_x = col * grid_size + grid_size // 2
            center_y = row * grid_size + grid_size // 2
            
            # Movimento illusorio basato sull'audio e frame
            offset_x = np.sin(frame * 0.1 + col * 0.2) * mid_val * 10 * intensity
            offset_y = np.cos(frame * 0.1 + row * 0.2) * high_val * 10 * intensity
            
            draw_y_junction(img, center_x + offset_x, center_y + offset_y, 
                          int(10 + bass_val * 5 * intensity))
    
    return img

def draw_y_junction(img, cx, cy, size):
    """Disegna una Y-junction"""
    h, w = img.shape[:2]
    
    # Tre linee che si incontrano al centro formando una Y
    angles = [0, 2*np.pi/3, 4*np.pi/3]  # 120 gradi tra le linee
    
    for angle in angles:
        end_x = cx + size * np.cos(angle)
        end_y = cy + size * np.sin(angle)
        draw_thick_line(img, cx, cy, end_x, end_y, 0, 3)

def draw_thick_line(img, x1, y1, x2, y2, color, thickness):
    """Disegna una linea spessa"""
    h, w = img.shape[:2]
    
    # Bresenham semplificato con spessore
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
        
        # Disegna pixel spessi
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
    
    # Pattern di base con elementi ripetuti
    spine_spacing = int(20 + bass_val * 10 * intensity)
    spine_size = int(8 + mid_val * 8 * intensity)
    
    # Drift speed basato sull'audio
    drift_x = frame * 0.5 * high_val * intensity
    drift_y = frame * 0.3 * mid_val * intensity
    
    rows = height // spine_spacing
    cols = width // spine_spacing
    
    for row in range(rows):
        for col in range(cols):
            # Posizione con drift
            x = (col * spine_spacing + drift_x) % width
            y = (row * spine_spacing + drift_y) % height
            
            # Crea piccoli elementi "spine" (frecce o elementi direzionali)
            draw_spine_element(img, x, y, spine_size, frame + row*5 + col*3, intensity)
    
    return img

def draw_spine_element(img, cx, cy, size, phase, intensity):
    """Disegna un elemento 'spine' che contribuisce all'illusione di drifting"""
    h, w = img.shape[:2]
    
    # Elemento a forma di freccia che cambia orientamento
    angle = phase * 0.1 * intensity
    
    # Punti della freccia
    tip_x = cx + size * np.cos(angle)
    tip_y = cy + size * np.sin(angle)
    
    base1_x = cx + size * 0.5 * np.cos(angle + 2.5)
    base1_y = cy + size * 0.5 * np.sin(angle + 2.5)
    
    base2_x = cx + size * 0.5 * np.cos(angle - 2.5)
    base2_y = cy + size * 0.5 * np.sin(angle - 2.5)
    
    # Disegna la freccia
    draw_line(img, cx, cy, tip_x, tip_y, 1)
    draw_line(img, tip_x, tip_y, base1_x, base1_y, 1)
    draw_line(img, tip_x, tip_y, base2_x, base2_y, 1)

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
    elif illusion_type == "Illusory Motion":
        return create_illusory_motion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Y-Junctions":
        return create_y_junctions_illusion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Drifting Spines":
        return create_drifting_spines_illusion(width, height, frame, audio_features, intensity, random_seed)
    elif illusion_type == "Spiral Illusion":
        return create_spiral_illusion(width, height, frame, audio_features, intensity, random_seed)
    else:
        return create_illusory_tilt(width, height, frame, audio_features, intensity, random_seed)_illusory_tilt(width, height, frame, audio_features, intensity, random_seed)

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
        # Assicura che il valore sia sempre numerico per il rendering
        if isinstance(audio_features['tempo'], (int, float, np.number)):
            tempo_display = float(audio_features['tempo'])
        else:
            tempo_display = 120.0
            
        st.info(f"🎯 BPM rilevato: {tempo_display:.1f}")

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
    st.subheader("🧠 Illusioni Scientifiche")
    st.markdown("""
    **🔬 Basate su ricerca neuroscientifica:**
    
    🌀 **Illusory Tilt** 
    - Line-type, Mixed-type, Edge-type
    - Triangoli e contrasti che ingannano la percezione
    
    ⚡ **Illusory Motion**
    - Mather's type & Takeuchi's type
    - Cerchi statici che sembrano espandersi
    - Effetti phi e reversed-phi
    
    🔗 **Y-Junctions**
    - Retinal slip simulation
    - Griglie con movimenti illusori
    
    🌊 **Drifting Spines**  
    - Elementi che sembrano scivolare
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
