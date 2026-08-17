import streamlit as st
import librosa
import numpy as np
import cv2
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
keyframes_rotation = {}  # NUOVO: Keyframe per la velocità di rotazione

if use_keyframes:
    st.sidebar.caption("Definisci i keyframe (tempo_in_secondi:valore).")
    st.sidebar.info("Esempio:\n0:1.0\n10:1.5\n20:0.8")

    intensity_str = st.sidebar.text_area("Keyframes Intensità", height=100)
    size_str = st.sidebar.text_area("Keyframes Dimensione", height=100)
    elements_str = st.sidebar.text_area("Keyframes Numero Elementi", height=100)
    rotation_str = st.sidebar.text_area("Keyframes Velocità Rotazione", height=100) # NUOVO

    # Valori di fallback se non si usano i keyframe
    intensity = 1.0
    element_size_factor = 1.0
    num_elements_factor = 1.0
    rotation_speed_factor = 1.0 # NUOVO

    def parse_keyframes(keyframe_string):
        keyframes_dict = {}
        for kf_line in keyframe_string.split('\n'):
            kf_line = kf_line.strip()
            if kf_line:
                try:
                    time_str, value_str = kf_line.split(':')
                    time = float(time_str.strip())
                    value = float(value_str.strip())
                    keyframes_dict[time] = value
                except ValueError:
                    st.sidebar.warning(f"Formato keyframe non valido: '{line}'. Ignorato.")
        return keyframes_dict

    keyframes_intensity = parse_keyframes(intensity_str)
    keyframes_size = parse_keyframes(size_str)
    keyframes_elements = parse_keyframes(elements_str)
    keyframes_rotation = parse_keyframes(rotation_str) # NUOVO
else:
    st.sidebar.subheader("🎨 Controlli Illusione")
    intensity = st.sidebar.slider("🔥 Intensità effetti", 0.1, 2.0, 1.0, 0.1)
    element_size_factor = st.sidebar.slider("📏 Densità/Dimensione", 0.5, 2.0, 1.0, 0.1)
    num_elements_factor = st.sidebar.slider("🔢 Fattore Elementi", 0.1, 2.0, 1.0, 0.1)
    rotation_speed_factor = st.sidebar.slider("🔄 Velocità Rotazione", 0.0, 2.0, 1.0, 0.1) # NUOVO


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

def draw_triangle_up(img, cx, cy, half_w, h, val):
    h_img, w_img = img.shape
    rr, cc = polygon([cy - h, cy, cy], [cx, cx - half_w, cx + half_w])
    valid = (rr >= 0) & (rr < h_img) & (cc >= 0) & (cc < w_img)
    img[rr[valid], cc[valid]] = val

def draw_triangle_down(img, cx, cy, half_w, h, val):
    h_img, w_img = img.shape
    rr, cc = polygon([cy + h, cy, cy], [cx, cx - half_w, cx + half_w])
    valid = (rr >= 0) & (rr < h_img) & (cc >= 0) & (cc < w_img)
    img[rr[valid], cc[valid]] = val

def draw_four_stroke_cell(img, cx, cy, half, state, style, min_radius, max_radius):
    """
    Cella a 4 fasi (phi / reversed phi) come nel riferimento:
    stato 0/1 = sfondo chiaro (cerchio piccolo -> grande), stato 2/3 = sfondo scuro (reversed phi).
    style 'outline' = Mather (line-type/edge-type), 'filled' = Takeuchi (mixed-type).
    """
    h_img, w_img = img.shape
    bg_white = state in (0, 1)
    bgval = 1.0 if bg_white else 0.0
    fgval = 0.0 if bg_white else 1.0
    fill_rect(img, cx - half, cy - half, cx + half, cy + half, bgval)
    radius = int(min_radius if state % 2 == 0 else max_radius)
    radius = max(1, min(radius, half))
    rr, cc = disk((cy, cx), radius, shape=(h_img, w_img))
    img[rr, cc] = fgval
    if style == "outline" and radius > 2:
        rr2, cc2 = disk((cy, cx), radius - 2, shape=(h_img, w_img))
        img[rr2, cc2] = bgval

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

def illusory_tilt_line_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor):
    """
    ILLUSORY TILT - Line-type (Kitaoka).
    Griglia di celle bowtie: triangolo superiore e inferiore a contrasto invertito,
    separati da una linea centrale. L'alternanza di polarita' a scacchiera,
    con sfasamento riga per riga, genera l'illusione di inclinazione della linea.
    """
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_cell = 90.0
    cell = int(base_cell / num_elements_factor * element_size_factor + bass_val * 20 * intensity)
    cell = max(10, cell)
    half_w = int(cell * 0.42)
    half_h = int(cell * 0.42)
    line_width = max(1, int(1 + high_val * 4 * intensity))

    row_shift = int(frame * 0.3 * rotation_speed_factor * (0.3 + mid_val))

    for row_idx, cy in enumerate(range(cell // 2, height, cell)):
        phase = (row_idx + row_shift) % 2
        for col_idx, cx in enumerate(range(cell // 2, width, cell)):
            top_white = (col_idx + phase) % 2 == 0
            top_val = 1.0 if top_white else 0.0
            bottom_val = 1.0 - top_val
            draw_triangle_down(img, cx, cy, half_w, half_h, top_val)
            draw_triangle_up(img, cx, cy, half_w, half_h, bottom_val)
            rr, cc = line(max(0, cy - line_width // 2), max(0, cx - half_w),
                          max(0, cy - line_width // 2), min(width - 1, cx + half_w))
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 1.0 - top_val
            for t in range(line_width):
                yy = min(height - 1, cy + t)
                img[yy, max(0, cx - half_w):min(width, cx + half_w)] = 1.0 - top_val
    return img

def illusory_tilt_mixed_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor):
    """
    ILLUSORY TILT - Mixed-type (lines & edges).
    Stessa griglia bowtie del line-type, ma meta' delle celle mostra la linea
    centrale e meta' mostra solo il bordo di contrasto (edge), a scacchiera:
    combinazione "linee & edge" come nel pannello centrale del riferimento.
    """
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_cell = 90.0
    cell = int(base_cell / num_elements_factor * element_size_factor + bass_val * 20 * intensity)
    cell = max(10, cell)
    half_w = int(cell * 0.42)
    half_h = int(cell * 0.42)
    line_width = max(1, int(1 + high_val * 4 * intensity))
    row_shift = int(frame * 0.3 * rotation_speed_factor * (0.3 + mid_val))

    for row_idx, cy in enumerate(range(cell // 2, height, cell)):
        phase = (row_idx + row_shift) % 2
        for col_idx, cx in enumerate(range(cell // 2, width, cell)):
            top_white = (col_idx + phase) % 2 == 0
            top_val = 1.0 if top_white else 0.0
            bottom_val = 1.0 - top_val
            draw_triangle_down(img, cx, cy, half_w, half_h, top_val)
            draw_triangle_up(img, cx, cy, half_w, half_h, bottom_val)
            has_line = (row_idx + col_idx) % 2 == 0
            if has_line:
                for t in range(line_width):
                    yy = min(height - 1, cy + t)
                    img[yy, max(0, cx - half_w):min(width, cx + half_w)] = 1.0 - top_val
    return img

def illusory_tilt_edge_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor):
    """
    ILLUSORY TILT - Edge-type.
    Stessa griglia bowtie, senza linea: solo il bordo di contrasto tra i due
    triangoli genera l'inclinazione percepita ("=" geometrico nel riferimento).
    """
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    base_cell = 90.0
    cell = int(base_cell / num_elements_factor * element_size_factor + mid_val * 25 * intensity)
    cell = max(10, cell)
    half_w = int(cell * 0.42)
    half_h = int(cell * 0.42)
    row_shift = int(frame * 0.3 * rotation_speed_factor * (0.3 + bass_val))

    for row_idx, cy in enumerate(range(cell // 2, height, cell)):
        phase = (row_idx + row_shift) % 2
        for col_idx, cx in enumerate(range(cell // 2, width, cell)):
            top_white = (col_idx + phase) % 2 == 0
            top_val = 1.0 if top_white else 0.0
            bottom_val = 1.0 - top_val
            draw_triangle_down(img, cx, cy, half_w, half_h, top_val)
            draw_triangle_up(img, cx, cy, half_w, half_h, bottom_val)
    return img

def illusory_motion_mather_line(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor):
    """
    ILLUSORY MOTION - Line-type / Edge-type <Mather's type>.
    Four-stroke apparent motion (phi / reversed phi) con cerchi a CONTORNO
    (Mather & Murdoch, 1999). Ogni cella cicla tra 4 stati (bianco piccolo ->
    bianco grande -> nero piccolo -> nero grande); lo sfasamento per posizione
    genera l'onda di moto illusorio che attraversa lo schermo.
    """
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    tempo_factor = audio_features["tempo"] / 120.0

    base_cell = 110.0
    cell = int(base_cell / num_elements_factor * element_size_factor)
    cell = max(20, cell)
    half = cell // 2
    min_radius = half * 0.35
    max_radius = half * 0.85 * (0.6 + 0.4 * bass_val * intensity)

    stroke_len = max(2, int(10 / (tempo_factor * rotation_speed_factor + 0.05)))
    global_cycle = frame // stroke_len

    for row_idx, cy in enumerate(range(half, height, cell)):
        for col_idx, cx in enumerate(range(half, width, cell)):
            state = (global_cycle + row_idx + col_idx) % 4
            draw_four_stroke_cell(img, cx, cy, half, state, "outline", min_radius, max_radius)
    return img

def illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor):
    """
    ILLUSORY MOTION - Mixed-type <Takeuchi's type> (1997, cafe wall motion analogue).
    Come la variante Mather ma con cerchi PIENI (edge stimuli anziche' line
    stimuli): la polarita' si inverte in modo netto ad ogni fase, rinforzando
    la componente "mixed" linee/edge del fenomeno.
    """
    img = np.zeros((height, width), dtype=float)
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    tempo_factor = audio_features["tempo"] / 120.0

    base_cell = 90.0
    cell = int(base_cell / num_elements_factor * element_size_factor + mid_val * 25 * intensity)
    cell = max(15, cell)
    half = cell // 2
    min_radius = half * 0.3
    max_radius = half * 0.9 * (0.6 + 0.4 * high_val * intensity)

    stroke_len = max(2, int(8 / (tempo_factor * rotation_speed_factor + 0.05)))
    global_cycle = frame // stroke_len

    for row_idx, cy in enumerate(range(half, height, cell)):
        for col_idx, cx in enumerate(range(half, width, cell)):
            state = (global_cycle + row_idx - col_idx) % 4
            draw_four_stroke_cell(img, cx, cy, half, state, "filled", min_radius, max_radius)
    return img

def y_junctions_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor): # AGGIORNATO
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    
    base_square_size = 50.0
    square_size = int(base_square_size / num_elements_factor * element_size_factor + bass_val * 40 * intensity)
    square_size = max(1, square_size)
    lateral_shift = int((frame * 0.5 * mid_val * intensity * rotation_speed_factor) % max(1, square_size)) # AGGIORNATO

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

def drifting_spines_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor):
    """
    DRIFTING SPINES ILLUSION.
    Texture densa di piccoli marcatori a farfalla (bowtie), come nel
    riferimento: ogni riga e' traslata orizzontalmente rispetto alla
    precedente (drift), producendo "retinal slip" e moto illusorio laterale.
    """
    img = np.zeros((height, width), dtype=float)
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    tempo_factor = audio_features["tempo"] / 120.0

    base_spacing = 26.0
    spacing = int(base_spacing / num_elements_factor * element_size_factor + bass_val * 8 * intensity)
    spacing = max(6, spacing)
    marker_half = max(2, int(spacing * 0.35))

    drift_speed = max(0.01, tempo_factor * intensity * rotation_speed_factor)
    drift_offset = (frame * drift_speed * 3) % spacing

    for row_idx, y in enumerate(range(spacing // 2, height, spacing)):
        row_shift = int(drift_offset * (1 if row_idx % 2 == 0 else -1))
        for col_idx, x0 in enumerate(range(spacing // 2, width + spacing, spacing)):
            x = x0 + row_shift
            if -marker_half <= x < width + marker_half:
                val = 1.0 if (row_idx + col_idx) % 2 == 0 else 0.0
                draw_triangle_up(img, x, y, marker_half, marker_half, val)
                draw_triangle_down(img, x, y, marker_half, marker_half, 1.0 - val)

    for x in range(0, width, max(2, spacing // 2)):
        hy = int(height // 2 + 40 * np.sin(x * 0.05 * rotation_speed_factor + drift_offset * 0.1))
        if 0 <= hy < height:
            radius = max(1, int(2 + high_val * 3 * intensity))
            rr, cc = disk((hy, x), radius, shape=(height, width))
            img[rr, cc] = 0.7
    return img

def spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor): # AGGIORNATO
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    cx, cy = width // 2, height // 2
    max_radius = min(width, height) // 2
    spiral_tightness = 0.1 * element_size_factor + bass_val * 0.2 * intensity
    rotation_speed = frame * 0.05 * rotation_speed_factor + mid_val * 0.1 # AGGIORNATO
    
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

def zollner_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor): # AGGIORNATO
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    base_spacing = 70.0
    line_spacing = int(base_spacing / num_elements_factor * element_size_factor + bass_val * 20 * intensity)
    line_spacing = max(1, line_spacing)
    
    oblique_angle = np.radians(45 + mid_val * 45)
    
    horizontal_shift = int(high_val * 10 * rotation_speed_factor) # AGGIORNATO

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

def generate_illusion_frame(width, height, frame, audio_features, intensity, illusion_type, seed, element_size_factor, num_elements_factor, rotation_speed_factor): # AGGIORNATO
    np.random.seed(seed + frame)

    if illusion_type == "Illusory Tilt (Line)":
        img = illusory_tilt_line_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    elif illusion_type == "Illusory Tilt (Mixed)":
        img = illusory_tilt_mixed_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    elif illusion_type == "Illusory Tilt (Edge)":
        img = illusory_tilt_edge_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    elif illusion_type == "Illusory Motion (Mather)":
        img = illusory_motion_mather_line(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    elif illusion_type == "Illusory Motion (Takeuchi)":
        img = illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    elif illusion_type == "Y-Junctions":
        img = y_junctions_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    elif illusion_type == "Drifting Spines":
        img = drifting_spines_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    elif illusion_type == "Spiral Illusion":
        img = spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    elif illusion_type == "Zollner Illusion":
        img = zollner_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
    else:
        img = spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor)
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

    with st.spinner("🎧 Analisi audio (BPM, bande di frequenza)..."):
        audio_features = analyze_audio(tmp_audio.name, duration, fps)
    tempo_display = audio_features["tempo"] if isinstance(audio_features["tempo"], (int, float)) else 120.0
    st.info(f"🎯 BPM rilevato: {tempo_display:.1f}")
    st.info(f"🧬 Illusione selezionata: {illusion_type} (implementazione scientifica)")

    seed = random.randint(1, 10000)

    # ---------------------------------
    # RENDERING DIRETTO VIA OPENCV
    # (niente più matplotlib Agg per-frame: si scrive direttamente il buffer
    # numpy sul VideoWriter, molto più veloce di FuncAnimation+FFMpegWriter)
    # ---------------------------------
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(tmp_video.name, fourcc, fps, size)

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    update_every = max(1, n_frames // 100)

    for frame in range(n_frames):
        current_time = frame / fps
        current_intensity = intensity
        current_size_factor = element_size_factor
        current_num_elements_factor = num_elements_factor
        current_rotation_speed_factor = rotation_speed_factor

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
            if keyframes_rotation:
                interpolated_rotation = interpolate_value(current_time, keyframes_rotation)
                if interpolated_rotation is not None:
                    current_rotation_speed_factor = interpolated_rotation

        colored = generate_illusion_frame(
            size[0], size[1], frame, audio_features,
            current_intensity, illusion_type, seed, current_size_factor,
            current_num_elements_factor, current_rotation_speed_factor
        )
        frame_uint8 = (np.clip(colored, 0.0, 1.0) * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

        if frame % update_every == 0 or frame == n_frames - 1:
            progress_bar.progress((frame + 1) / n_frames)
            status_text.text(f"🎨 Rendering frame {frame + 1}/{n_frames}")

    video_writer.release()
    progress_bar.progress(1.0)
    status_text.text("🎬 Frame completati, mux audio in corso...")

    # ---------------------------------
    # MUX AUDIO + TITOLO IN UN UNICO PASSAGGIO FFMPEG
    # (prima erano due processi ffmpeg separati: mux e poi drawtext)
    # ---------------------------------
    with st.spinner("🔊 Unione video + audio (e titolo, se presente)..."):
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_stream = ffmpeg.input(tmp_video.name)
        audio_stream = ffmpeg.input(tmp_audio.name)

        output_kwargs = dict(vcodec="libx264", acodec="aac", strict="experimental")

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
            output_kwargs["vf"] = f"drawtext={drawtext_args}"

        final = ffmpeg.output(video_stream, audio_stream, output_file.name, **output_kwargs)
        ffmpeg.run(final, overwrite_output=True, quiet=True)

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
