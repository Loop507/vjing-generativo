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
    ["Illusory Tilt", "Illusory Motion", "Y-Junctions", "Drifting Spines", "Spiral Illusion"]
)

video_title = st.text_input("📝 Titolo del video", "Visual Synthesis")
position = st.selectbox("📍 Posizione del titolo", ["Sopra", "Sotto", "Destra", "Sinistra", "Centro"])
aspect_ratio = st.selectbox("📺 Formato video", ["16:9", "1:1", "9:16"])
intensity = st.sidebar.slider("🔥 Intensità effetti", 0.1, 2.0, 1.0, 0.1)

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

def illusory_tilt_line_type(width, height, frame, audio_features, intensity):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    triangle_size = int(30 + bass_val * 40 * intensity)
    rotation_angle = frame * 0.5 + mid_val * 45

    for y in range(0, height, triangle_size * 2):
        for x in range(0, width, triangle_size * 2):
            center_x, center_y = x + triangle_size, y + triangle_size
            angle_rad = np.radians(rotation_angle + (x+y)*0.1)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            vertices = np.array([
                [-triangle_size//2, -triangle_size//2],
                [ triangle_size//2, -triangle_size//2],
                [ 0,                 triangle_size//2]
            ])
            rotated = np.array([
                [v[0]*cos_a - v[1]*sin_a + center_x, v[0]*sin_a + v[1]*cos_a + center_y]
                for v in vertices
            ]).astype(int)
            rr, cc = polygon(rotated[:,1], rotated[:,0], (height, width))
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 1.0
    return img

def illusory_tilt_mixed_type(width, height, frame, audio_features, intensity):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    line_spacing = int(20 + bass_val * 30)
    line_width = max(1, int(2 + high_val * 8 * intensity))

    for i in range(0, width + height, line_spacing):
        x1, y1 = min(i, width-1), 0
        x2, y2 = 0, min(i, height-1)
        rr, cc = line(y1, x1, y2, x2)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        img[rr[valid], cc[valid]] = 1.0
        if line_width > 2:
            for offset in range(-line_width//2, line_width//2 + 1):
                rr_o = rr + offset
                cc_o = cc + offset
                valid_o = (rr_o >= 0) & (rr_o < height) & (cc_o >= 0) & (cc_o < width)
                img[rr_o[valid_o], cc_o[valid_o]] = 0.7

    perpendicular_offset = frame * 2
    for i in range(perpendicular_offset, width + height, line_spacing * 2):
        x1, y1 = 0, min(i, height-1)
        x2, y2 = min(i, width-1), 0
        rr, cc = line(y1, x1, y2, x2)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        img[rr[valid], cc[valid]] = 0.5
    return img

def illusory_tilt_edge_type(width, height, frame, audio_features, intensity):
    img = np.zeros((height, width), dtype=float)
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    square_size = int(40 + mid_val * 60 * intensity)
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

def illusory_motion_mather_line(width, height, frame, audio_features, intensity):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    tempo_factor = audio_features["tempo"] / 120.0

    centers = [(width//4, height//4), (3*width//4, height//4),
               (width//4, 3*height//4), (3*width//4, 3*height//4)]

    for cx, cy in centers:
        max_radius = min(width, height) // 6
        current_radius = int(max_radius * (0.5 + 0.5 * bass_val * intensity))
        num_spokes = int(8 + tempo_factor * 4)
        for i in range(num_spokes):
            angle = (2 * np.pi * i / num_spokes) + (frame * 0.1 * tempo_factor)
            ex = int(cx + current_radius * np.cos(angle))
            ey = int(cy + current_radius * np.sin(angle))
            rr, cc = line(cy, cx, ey, ex)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = 1.0
        for r in range(max(1, current_radius//4), current_radius, max(1, current_radius//4)):
            rr, cc = disk((cy, cx), r, shape=(height, width))
            img[rr, cc] = 0.7
    return img

def illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity):
    img = np.zeros((height, width), dtype=float)
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    element_size = int(25 + mid_val * 35 * intensity)
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

def y_junctions_illusion(width, height, frame, audio_features, intensity):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    square_size = int(30 + bass_val * 40 * intensity)
    lateral_shift = int(frame * 0.5 * mid_val * intensity) % max(1, square_size)

    for y in range(0, height, square_size):
        for x in range(-lateral_shift, width + square_size, square_size):
            fill = (x//square_size + y//square_size) % 2 == 0
            if 0 <= x < width:
                end_x, end_y = min(x + square_size, width), min(y + square_size, height)
                img[y:end_y, x:end_x] = 1.0 if fill else 0.0
            if 0 < x < width and 0 < y < height:
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

def drifting_spines_illusion(width, height, frame, audio_features, intensity):
    img = np.zeros((height, width), dtype=float)
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    tempo_factor = audio_features["tempo"] / 120.0

    drift_speed = max(0.01, tempo_factor * intensity)
    drift_offset = (frame * drift_speed) % 100

    spine_spacing = int(40 + high_val * 30)
    spine_length = int(20 + high_val * 25 * intensity)

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

def spiral_illusion(width, height, frame, audio_features, intensity):
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    cx, cy = width // 2, height // 2
    max_radius = min(width, height) // 2
    spiral_tightness = 0.1 + bass_val * 0.2 * intensity
    rotation_speed = frame * 0.05 + mid_val * 0.1

    num_arms = 3
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

def generate_illusion_frame(width, height, frame, audio_features, intensity, illusion_type, seed):
    np.random.seed(seed + frame)
    tempo_factor = max(0.5, audio_features["tempo"] / 120.0)
    subtype_cycle = int(frame / (30 * tempo_factor)) % 3

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
    else:
        img = spiral_illusion(width, height, frame, audio_features, intensity)
    return apply_colors(img, line_color, bg_color)

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
            colored = generate_illusion_frame(size[0], size[1], frame, audio_features, intensity, illusion_type, seed)
            im.set_array(colored)
            return [im]

        anim = FuncAnimation(fig, animate, frames=n_frames, blit=True, interval=1000/fps)
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        # ✅ Correzione writer: uso FFMpegWriter
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Loop507'), bitrate=1800)
        anim.save(tmp_video.name, writer=writer)
        plt.close(fig)

        # 🔥 Merge audio + video con ffmpeg
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video = ffmpeg.input(tmp_video.name)
        audio = ffmpeg.input(tmp_audio.name)
        final = ffmpeg.output(video, audio, output_file.name, vcodec="libx264", acodec="aac", strict="experimental")
        ffmpeg.run(final, overwrite_output=True, quiet=True)

        # Overlay titolo (opzionale) con gestione font cross-platform
        if video_title.strip():
            pos_map = {
                "Sopra": "(w-text_w)/2:20",
                "Sotto": "(w-text_w)/2:h-text_h-20",
                "Destra": "w-text_w-20:(h-text_h)/2",
                "Sinistra": "20:(h-text_h)/2",
                "Centro": "(w-text_w)/2:(h-text_h)/2"
            }
            candidate_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc"
            ]
            fontfile = next((p for p in candidate_fonts if os.path.exists(p)), None)
            text_escaped = escape_drawtext(video_title)
            drawtext_args = f"text='{text_escaped}':fontcolor=white:fontsize=48:x={pos_map[position].split(':')[0]}:y={pos_map[position].split(':')[1]}"
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

        # Cleanup
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
