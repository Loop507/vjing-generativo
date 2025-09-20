import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import tempfile
import os
import random
import ffmpeg
from skimage.draw import line, polygon, disk

# ---------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------
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
    [
        "Illusory Tilt (Line)", "Illusory Tilt (Mixed)", "Illusory Tilt (Edge)",
        "Illusory Motion (Mather)", "Illusory Motion (Takeuchi)",
        "Y-Junctions", "Drifting Spines", "Spiral Illusion", "Zollner Illusion"
    ]
)

# ---------------------------------
# SEZIONE KEYFRAME AGGIORNATA
# ---------------------------------
st.sidebar.subheader("🎥 Sequenza Keyframe (avanzato)")
use_keyframes = st.sidebar.checkbox("Usa Sequenza Keyframe", value=False)

keyframes_intensity = {}
keyframes_size = {}
keyframes_elements = {}

if use_keyframes:
    st.sidebar.caption("Definisci i keyframe (tempo_in_secondi:valore).")
    st.sidebar.info("Esempio:\n0:1.0\n10:1.5\n20:0.8")

    intensity_str = st.sidebar.text_area("Keyframes Intensità", height=100)
    size_str = st.sidebar.text_area("Keyframes Dimensione", height=100)
    elements_str = st.sidebar.text_area("Keyframes Numero Elementi", height=100)
    
    # Valori di fallback se non si usano i keyframe
    intensity = 1.0
    element_size_factor = 1.0
    num_elements_factor = 1.0

    def parse_keyframes(keyframe_string):
        keyframes_dict = {}
        for line in keyframe_string.split('\n'):
            line = line.strip()
            if line:
                try:
                    time_str, value_str = line.split(':')
                    time = float(time_str.strip())
                    value = float(value_str.strip())
                    keyframes_dict[time] = value
                except ValueError:
                    st.sidebar.warning(f"Formato keyframe non valido: '{line}'. Ignorato.")
        return keyframes_dict

    keyframes_intensity = parse_keyframes(intensity_str)
    keyframes_size = parse_keyframes(size_str)
    keyframes_elements = parse_keyframes(elements_str)
else:
    st.sidebar.subheader("🎨 Controlli Illusione")
    intensity = st.sidebar.slider("🔥 Intensità effetti", 0.1, 2.0, 1.0, 0.1)
    element_size_factor = st.sidebar.slider("📏 Densità/Dimensione", 0.5, 2.0, 1.0, 0.1)
    num_elements_factor = st.sidebar.slider("🔢 Fattore Elementi", 0.1, 2.0, 1.0, 0.1)


st.sidebar.subheader("📝 Titolo Video")
video_title = st.text_input("Testo del titolo", "")
font_size = st.sidebar.slider("Grandezza carattere", 20, 100, 48, 2)
vertical_position = st.sidebar.selectbox("Posizione verticale", ["Sopra", "Sotto", "Centro"])
horizontal_position = st.sidebar.selectbox("Posizione orizzontale", ["Sinistra", "Destra", "Centro"])

aspect_ratio = st.selectbox("📺 Formato video", ["16:9", "1:1", "9:16"])


# ---------------------------------
# ANALISI AUDIO
# ---------------------------------
def analyze_audio(audio_path, duration, fps):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    frame_length = int(sr / fps)
    n_frames = max(1, int(duration * fps))
    bass_values, mid_values, high_values = [], [], []

    for i in range(n_frames):
        start = i * frame_length
        end = min(start + frame_length, len(y))
        if start >= len(y):
            frame_audio = np.zeros(frame_length)
        else:
            frame_audio = y[start:end]
            if len(frame_audio) < frame_length:
                frame_audio = np.pad(frame_audio, (0, frame_length - len(frame_audio)))
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

# ---------------------------------
# UTILS DISEGNO / COLORI
# ---------------------------------

def apply_colors(img, line_color, bg_color):
    """Applica i colori personalizzati a un'immagine mono-canale [0..1] -> RGB."""
    line_rgb = np.array([int(line_color[1:3],16)/255, int(line_color[3:5],16)/255, int(line_color[5:7],16)/255])
    bg_rgb = np.array([int(bg_color[1:3],16)/255, int(bg_color[3:5],16)/255, int(bg_color[5:7],16)/255])

    colored = np.zeros((*img.shape, 3), dtype=float)
    for i in range(3):  # RGB channels
        colored[:,:,i] = img * line_rgb[i] + (1 - img) * bg_rgb[i]
    return colored

def clamp_rect(x1, y1, x2, y2, width, height):
    x1 = int(max(0, min(width-1, x1)))
    x2 = int(max(0, min(width-1, x2)))
    y1 = int(max(0, min(height-1, y1)))
    y2 = int(max(0, min(height-1, y2)))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return x1, y1, x2, y2

def fill_rect(img, x1, y1, x2, y2, val):
    h, w = img.shape
    x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
    img[y1:y2+1, x1:x2+1] = val

def draw_rect_border(img, x1, y1, x2, y2, val, thickness=1):
    h, w = img.shape
    x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
    for t in range(thickness):
        rr, cc = line(y1+t, x1, y1+t, x2)
        img[rr, cc] = val
        rr, cc = line(y2-t, x1, y2-t, x2)
        img[rr, cc] = val
        rr, cc = line(y1, x1+t, y2, x1+t)
        img[rr, cc] = val
        rr, cc = line(y1, x2-t, y2, x2-t)
        img[rr, cc] = val

def escape_drawtext(text: str) -> str:
    # Minima escape per drawtext ffmpeg
    return (
        text.replace("\\", "\\\\")   # \  -> \\\\
            .replace(":", "\\:")     # :  -> \:
            .replace("'", "\\'")     # '  -> \'
    )

# ---------------------------------
# ILLUSIONI SCIENTIFICHE
# ---------------------------------

def illusory_tilt_line_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    """
    EFFETTO LINEE DIAGONALI - Corretto per riempire l'intero schermo
    """
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    base_spacing = 70.0
    line_spacing = int(base_spacing / num_elements_factor * element_size_factor + bass_val * 15 * intensity)
    line_spacing = max(1, line_spacing)
    line_width = max(1, int(1 + high_val * 4))

    # Disegna linee diagonali da in basso a sinistra a in alto a destra
    for i in range(-height, width):
        x1 = max(0, i)
        y1 = max(0, -i)
        x2 = min(width - 1, i + height - 1)
        y2 = min(height - 1, -i + width - 1)
        if (i % line_spacing) == 0:
            rr, cc = line(y1, x1, y2, x2)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 1.0

    # Disegna linee diagonali perpendicolari da in alto a sinistra a in basso a destra
    perpendicular_offset = int(frame * 2)
    for i in range(-width - perpendicular_offset, height + perpendicular_offset, line_spacing):
        x1 = max(0, i)
        y1 = min(height - 1, i + width - 1)
        x2 = min(width - 1, i + width - 1)
        y2 = max(0, i)
        if (i % line_spacing) == 0:
            rr, cc = line(y1, x1, y2, x2)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 0.5
    return img

def illusory_tilt_mixed_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    """
    EFFETTO TRIANGOLI ROTANTI
    """
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    base_step_size = 50.0
    step_size = int(base_step_size / num_elements_factor * element_size_factor)
    step_size = max(1, step_size)
    rotation_angle = frame * 0.5 + mid_val * 45

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            angle_rad = np.radians(rotation_angle + (x+y)*0.05)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            triangle_size = int(25 + bass_val * 35 * intensity)
            
            vertices = np.array([
                [-triangle_size//2, -triangle_size//2],
                [ triangle_size//2, -triangle_size//2],
                [ 0,                 triangle_size//2]
            ])
            
            center_x, center_y = x + step_size // 2, y + step_size // 2
            
            rotated = np.array([
                [v[0]*cos_a - v[1]*sin_a + center_x, v[0]*sin_a + v[1]*cos_a + center_y]
                for v in vertices
            ]).astype(int)
            rr, cc = polygon(rotated[:,1], rotated[:,0], (height, width))
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 1.0
    return img

def illusory_tilt_edge_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    img = np.zeros((height, width), dtype=float)
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    base_square_size = 60.0
    square_size = int(base_square_size / num_elements_factor * element_size_factor + mid_val * 60 * intensity)
    square_size = max(1, square_size)
    edge_width = max(1, int(1 + high_val * 4))

    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            fill_value = 1.0 if (x//square_size + y//square_size + frame//10) % 2 == 0 else 0.0
            end_x, end_y = min(x + square_size, width), min(y + square_size, height)
            img[y:end_y, x:end_x] = fill_value
            if end_x < width:
                img[y:end_y, end_x-edge_width:end_x] = 1.0 - fill_value
            if end_y < height:
                img[end_y-edge_width:end_y, x:end_x] = 1.0 - fill_value
    return img

def illusory_motion_mather_line(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    tempo_factor = audio_features["tempo"] / 120.0

    centers = [(width//4, height//4), (3*width//4, height//4),
               (width//4, 3*height//4), (3*width//4, 3*height//4)]

    for cx, cy in centers:
        max_radius = min(width, height) // 6 * element_size_factor
        current_radius = int(max_radius * (0.5 + 0.5 * bass_val * intensity))
        num_spokes = int(8 * num_elements_factor + tempo_factor * 4)
        num_spokes = max(1, num_spokes)
        for i in range(num_spokes):
            angle = (2 * np.pi * i / num_spokes) + (frame * 0.1 * tempo_factor)
            ex = int(cx + current_radius * np.cos(angle))
            ey = int(cy + current_radius * np.sin(angle))
            rr, cc = line(cy, cx, ey, ex)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 1.0
        for r in range(max(1, int(current_radius//4*element_size_factor)), current_radius, max(1, int(current_radius//4*element_size_factor))):
            rr, cc = disk((cy, cx), r, shape=(height, width))
            img[rr, cc] = 0.7
    return img

def illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    img = np.zeros((height, width), dtype=float)
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    base_element_size = 40.0
    element_size = int(base_element_size / num_elements_factor * element_size_factor + mid_val * 35 * intensity)
    element_size = max(1, element_size)
    phase = frame * 0.2

    for y in range(0, height, element_size):
        for x in range(0, width, element_size):
            local_phase = phase + (x + y) * 0.01
            should_fill = (np.sin(local_phase) + high_val) > 0.5
            x2 = min(x + element_size - 1, width-1)
            y2 = min(y + element_size - 1, height-1)
            if should_fill:
                fill_rect(img, x, y, x2, y2, 1.0)
                draw_rect_border(img, x, y, x2, y2, 0.0, thickness=2)
            else:
                draw_rect_border(img, x, y, x2, y2, 0.8, thickness=2)
    return img

def y_junctions_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    
    base_square_size = 50.0
    square_size = int(base_square_size / num_elements_factor * element_size_factor + bass_val * 40 * intensity)
    square_size = max(1, square_size)
    lateral_shift = int((frame * 0.5 * mid_val * intensity) % max(1, square_size))

    start_x = -lateral_shift
    
    for y in range(0, height, square_size):
        for x in range(start_x, width + square_size, square_size):
            fill = (x//square_size + y//square_size) % 2 == 0
            
            end_x, end_y = min(x + square_size, width), min(y + square_size, height)
            if end_x > 0 and end_y > 0 and x < width and y < height:
                x1 = max(0, x)
                y1 = max(0, y)
                img[y1:end_y, x1:end_x] = 1.0 if fill else 0.0
            
            if x > 0 and y > 0 and x < width and y < height:
                jx, jy = x, y
                for d in (-1, 0, 1):
                    rr, cc = line(jy-5, jx+d, jy+5, jx+d)
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid], cc[valid]] = 0.5
                    rr, cc = line(jy+d, jx-5, jy+d, jx+5)
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid], cc[valid]] = 0.5
                    rr, cc = line(jy-3, jx-3+d, jy+3, jx+3+d)
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid], cc[valid]] = 0.5
    return img

def drifting_spines_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    img = np.zeros((height, width), dtype=float)
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    tempo_factor = audio_features["tempo"] / 120.0

    drift_speed = max(0.01, tempo_factor * intensity)
    drift_offset = (frame * drift_speed) % 100
    
    base_spine_spacing = 60.0
    spine_spacing = int(base_spine_spacing / num_elements_factor * element_size_factor + high_val * 30)
    spine_spacing = max(1, spine_spacing)
    spine_length = int(20 * element_size_factor + high_val * 25 * intensity)

    for y in range(0, height, spine_spacing):
        for x in range(int(-drift_offset), width + spine_spacing, spine_spacing):
            if 0 <= x < width:
                cx, cy = x, y + spine_spacing//2
                start_y = max(0, cy - spine_length//2)
                end_y = min(height-1, cy + spine_length//2)
                rr, cc = line(start_y, cx, end_y, cx)
                valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                img[rr[valid], cc[valid]] = 1.0
                arrow_size = spine_length // 4
                if arrow_size > 0:
                    rr, cc = line(cy - spine_length//2, cx, max(0, cy - spine_length//2 + arrow_size), max(0, cx - arrow_size))
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid], cc[valid]] = 1.0
                    rr, cc = line(cy - spine_length//2, cx, max(0, cy - spine_length//2 + arrow_size), min(width-1, cx + arrow_size))
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid], cc[valid]] = 1.0

    for x in range(0, width, max(1, spine_spacing//2)):
        hy = int(height//2 + 50 * np.sin(x * 0.05 + drift_offset * 0.1))
        if 0 <= hy < height:
            radius = max(1, int(3 * high_val * intensity))
            rr, cc = disk((hy, x), radius, shape=(height, width))
            img[rr, cc] = 0.7
    return img

def spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    cx, cy = width // 2, height // 2
    max_radius = min(width, height) // 2
    spiral_tightness = 0.1 * element_size_factor + bass_val * 0.2 * intensity
    rotation_speed = frame * 0.05 + mid_val * 0.1
    
    num_arms = max(1, int(3 * num_elements_factor))
    for arm in range(num_arms):
        arm_offset = (2 * np.pi * arm) / num_arms
        for r in range(5, max_radius, 3):
            angle = r * spiral_tightness + rotation_speed + arm_offset
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < width and 0 <= y < height:
                intensity_val = 0.8 + 0.2 * np.sin(r * 0.1 + rotation_speed)
                radius = max(1, int(2 + bass_val * 3))
                rr, cc = disk((y, x), radius, shape=(height, width))
                img[rr, cc] = intensity_val
    return img

def zollner_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    base_spacing = 70.0
    line_spacing = int(base_spacing / num_elements_factor * element_size_factor + bass_val * 20 * intensity)
    line_spacing = max(1, line_spacing)
    
    oblique_angle = np.radians(45 + mid_val * 45)
    
    horizontal_shift = int(high_val * 10)

    for x in range(0, width + line_spacing, line_spacing):
        x_shifted = x + horizontal_shift
        rr, cc = line(0, x_shifted, height - 1, x_shifted)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        img[rr[valid], cc[valid]] = 1.0

        for y in range(0, height, int(line_spacing / 2)):
            length = int(10 * element_size_factor)
            ex = int(x_shifted + length * np.cos(oblique_angle))
            ey = int(y + length * np.sin(oblique_angle))
            rr, cc = line(y, x_shifted, ey, ex)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 1.0
            
            ex = int(x_shifted - length * np.cos(oblique_angle))
            ey = int(y - length * np.sin(oblique_angle))
            rr, cc = line(y, x_shifted, ey, ex)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 1.0
            
    return img

def generate_illusion_frame(width, height, frame, audio_features, intensity, illusion_type, seed, element_size_factor, num_elements_factor):
    np.random.seed(seed + frame)

    if illusion_type == "Illusory Tilt (Line)":
        img = illusory_tilt_line_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    elif illusion_type == "Illusory Tilt (Mixed)":
        img = illusory_tilt_mixed_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    elif illusion_type == "Illusory Tilt (Edge)":
        img = illusory_tilt_edge_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    elif illusion_type == "Illusory Motion (Mather)":
        img = illusory_motion_mather_line(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    elif illusion_type == "Illusory Motion (Takeuchi)":
        img = illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    elif illusion_type == "Y-Junctions":
        img = y_junctions_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    elif illusion_type == "Drifting Spines":
        img = drifting_spines_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    elif illusion_type == "Spiral Illusion":
        img = spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    elif illusion_type == "Zollner Illusion":
        img = zollner_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    else:
        img = spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor)
    return apply_colors(img, line_color, bg_color)


# ---------------------------------
# LOGICA DI INTERPOLAZIONE
# ---------------------------------
def interpolate_value(time, keyframes):
    times = sorted(keyframes.keys())
    if not times:
        return None
    if time <= times[0]:
        return keyframes[times[0]]
    if time >= times[-1]:
        return keyframes[times[-1]]

    t1, v1 = None, None
    t2, v2 = None, None
    for i in range(len(times) - 1):
        if times[i] <= time < times[i+1]:
            t1, v1 = times[i], keyframes[times[i]]
            t2, v2 = times[i+1], keyframes[times[i+1]]
            break
    
    if t1 is not None and t2 is not None and t1 != t2:
        t = (time - t1) / (t2 - t1)
        return v1 + (v2 - v1) * t
    else:
        return v1


# ---------------------------------
# MAIN
# ---------------------------------
if uploaded_file and st.button("🚀 Genera Video Illusorio Scientifico", type="primary"):
    with st.spinner("🎨 Creazione video con illusioni scientificamente accurate..."):
        # Salva l'audio con estensione coerente
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in (".wav", ".mp3"): ext = ".wav"
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_audio.write(uploaded_file.read())
        tmp_audio.close()

        y, sr = librosa.load(tmp_audio.name, sr=None)
        duration = float(librosa.get_duration(y=y, sr=sr))
        st.info(f"🎵 Durata audio: {duration:.2f} sec")

        if aspect_ratio == "16:9": size=(1280,720)
        elif aspect_ratio == "1:1": size=(720,720)
        else: size=(720,1280)

        fps = 30
        n_frames = max(1, int(duration * fps))
        audio_features = analyze_audio(tmp_audio.name, duration, fps)
        tempo_display = audio_features["tempo"] if isinstance(audio_features["tempo"], (int, float)) else 120.0
        st.info(f"🎯 BPM rilevato: {tempo_display:.1f}")
        st.info(f"🧬 Illusione selezionata: {illusion_type} (implementazione scientifica)")

        seed = random.randint(1, 10000)
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax.axis("off")
        ax.set_facecolor('black')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        im = ax.imshow(np.zeros((size[1], size[0], 3)), aspect="equal")

        def animate(frame):
            current_time = frame / fps
            current_intensity = intensity
            current_size_factor = element_size_factor
            current_num_elements_factor = num_elements_factor
            
            if use_keyframes:
                if keyframes_intensity:
                    interpolated_intensity = interpolate_value(current_time, keyframes_intensity)
                    if interpolated_intensity is not None:
                        current_intensity = interpolated_intensity
                if keyframes_size:
                    interpolated_size = interpolate_value(current_time, keyframes_size)
                    if interpolated_size is not None:
                        current_size_factor = interpolated_size
                if keyframes_elements:
                    interpolated_elements = interpolate_value(current_time, keyframes_elements)
                    if interpolated_elements is not None:
                        current_num_elements_factor = interpolated_elements

            current_illusion_type = illusion_type

            colored = generate_illusion_frame(
                size[0], size[1], frame, audio_features,
                current_intensity, current_illusion_type, seed, current_size_factor, current_num_elements_factor
            )
            im.set_array(colored)
            return [im]


        anim = FuncAnimation(fig, animate, frames=n_frames, blit=True, interval=1000/fps)
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Loop507'), bitrate=1800)
        anim.save(tmp_video.name, writer=writer)
        plt.close(fig)

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video = ffmpeg.input(tmp_video.name)
        audio = ffmpeg.input(tmp_audio.name)
        final = ffmpeg.output(video, audio, output_file.name, vcodec="libx264", acodec="aac", strict="experimental")
        ffmpeg.run(final, overwrite_output=True, quiet=True)

        if video_title.strip():
            pos_x = "(w-text_w)/2" if horizontal_position == "Centro" else "20" if horizontal_position == "Sinistra" else "w-text_w-20"
            pos_y = "20" if vertical_position == "Sopra" else "h-text_h-20" if vertical_position == "Sotto" else "(h-text_h)/2"

            candidate_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc"
            ]
            fontfile = next((p for p in candidate_fonts if os.path.exists(p)), None)
            text_escaped = escape_drawtext(video_title)
            drawtext_args = f"text='{text_escaped}':fontcolor=white:fontsize={font_size}:x={pos_x}:y={pos_y}"
            if fontfile:
                drawtext_args += f":fontfile={fontfile}"

            titled_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            (
                ffmpeg
                .input(output_file.name)
                .output(
                    titled_file.name,
                    vf=f"drawtext={drawtext_args}",
                    vcodec="libx264", acodec="aac", strict="experimental"
                )
                .run(overwrite_output=True, quiet=True)
            )
            os.replace(titled_file.name, output_file.name)

        with open(output_file.name, "rb") as f:
            st.download_button(
                "📥 Scarica Video Illusorio Scientifico",
                f,
                file_name=f"vjing_{illusion_type.lower().replace(' ', '_')}_output.mp4",
                mime="video/mp4",
            )

        try:
            os.remove(tmp_audio.name)
            os.remove(tmp_video.name)
            os.remove(output_file.name)
        except Exception:
            pass

        st.success("✨ Video generato con successo! Implementazioni neuropsicologiche accurate.")
        st.info(
            f"""
            🧬 **Implementazione Scientifica Utilizzata:**
            - **{illusion_type}**: Basato su ricerca neuropsicologica
            - **Sincronizzazione Audio**: BPM→velocità transizioni, Bassi→movimenti globali, Medi→deformazioni, Alti→micro-dettagli
            - **Algoritmi**: Mather & Takeuchi (Motion), Retinal Slip (Y-Junctions), Phi Motion Effects
            """
        )
