import streamlit as st
import librosa
import numpy as np
import cv2
from PIL import Image
import io
import tempfile
import os
import random
import ffmpeg
from skimage.draw import line, polygon, disk


# ---------------------------------
# FUNZIONI E COSTANTI (definite prima di qualsiasi codice UI,
# cosi' ogni blocco Streamlit puo' chiamarle senza problemi di ordine)
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

def make_bowtie_tiles(width, height, cell, half_w, half_h, row_shift):
    """
    Costruisce, con operazioni numpy vettorizzate (niente loop per-cella),
    le maschere booleane tassellate del pattern bowtie e la griglia colore
    (top_val_pixel) gia' espansa a risoluzione pixel. Il template della
    singola cella viene disegnato UNA sola volta per frame (con le funzioni
    di disegno esistenti) e poi ripetuto con np.tile: e' il pattern che
    sostituisce il doppio loop Python for-riga/for-colonna.
    """
    cell = max(2, cell)
    n_rows = height // cell + 2
    n_cols = width // cell + 2

    template = np.zeros((cell, cell), dtype=float)
    tcx, tcy = cell // 2, cell // 2
    draw_triangle_down(template, tcx, tcy, half_w, half_h, 1.0)
    top_mask = template > 0
    template[:] = 0.0
    draw_triangle_up(template, tcx, tcy, half_w, half_h, 1.0)
    bottom_mask = template > 0

    top_mask_tiled = np.tile(top_mask, (n_rows, n_cols))[:height, :width]
    bottom_mask_tiled = np.tile(bottom_mask, (n_rows, n_cols))[:height, :width]

    row_idx = np.arange(n_rows)[:, None]
    col_idx = np.arange(n_cols)[None, :]
    phase = (row_idx + row_shift) % 2
    top_val_cells = ((col_idx + phase) % 2 == 0).astype(float)
    top_val_pixel = np.kron(top_val_cells, np.ones((cell, cell)))[:height, :width]

    return top_mask_tiled, bottom_mask_tiled, top_val_pixel

def escape_drawtext(text: str) -> str:
    # Minima escape per drawtext ffmpeg
    return (
        text.replace("\\", "\\\\")   # \  -> \\\\
            .replace(":", "\\:")     # :  -> \:
            .replace("'", "\\'")     # '  -> \'
    )

ILLUSION_SCIENCE = {
    "Illusory Tilt (Line)": {
        "it": "Kitaoka :: Line-type. Griglia bowtie a contrasto invertito con linea centrale; l'alternanza di polarita' a scacchiera genera l'inclinazione percepita.",
        "en": "Kitaoka :: Line-type. Contrast-reversed bowtie grid with a central line; checkerboard polarity alternation drives the perceived tilt.",
        "tags": ["illusorytilt", "linetype", "kitaoka"],
    },
    "Illusory Tilt (Mixed)": {
        "it": "Kitaoka :: Mixed-type. Come il line-type ma meta' celle con linea e meta' solo a bordo di contrasto, a scacchiera.",
        "en": "Kitaoka :: Mixed-type. Same grid as line-type, half the cells carry a center line, half rely on contrast edges only, checkerboard-distributed.",
        "tags": ["illusorytilt", "mixedtype", "kitaoka"],
    },
    "Illusory Tilt (Edge)": {
        "it": "Kitaoka :: Edge-type. Solo bordo di contrasto tra triangoli, nessuna linea: inclinazione percepita puramente da edge.",
        "en": "Kitaoka :: Edge-type. Contrast edge only between triangles, no line: perceived tilt from edge information alone.",
        "tags": ["illusorytilt", "edgetype", "kitaoka"],
    },
    "Illusory Motion (Mather)": {
        "it": "Mather & Murdoch (1999) :: four-stroke apparent motion, cerchi a contorno (line stimuli), phi / reversed phi.",
        "en": "Mather & Murdoch (1999) :: four-stroke apparent motion, outline circles (line stimuli), phi / reversed phi.",
        "tags": ["illusorymotion", "mather", "phimotion"],
    },
    "Illusory Motion (Takeuchi)": {
        "it": "Takeuchi (1997) :: motion analogue del cafe wall, cerchi pieni (edge stimuli), fase mixed-type.",
        "en": "Takeuchi (1997) :: motion analogue of the cafe wall illusion, filled circles (edge stimuli), mixed-type phase.",
        "tags": ["illusorymotion", "takeuchi", "cafewall"],
    },
    "Y-Junctions": {
        "it": "Retinal slip su reticolo a scacchiera con marcatori a Y-junction alle intersezioni.",
        "en": "Retinal slip over a checkerboard lattice with Y-junction markers at the intersections.",
        "tags": ["yjunctions", "retinalslip"],
    },
    "Drifting Spines": {
        "it": "Texture densa di marcatori a farfalla (bowtie) con drift orizzontale per riga, retinal slip laterale.",
        "en": "Dense bowtie-marker texture with per-row horizontal drift, lateral retinal slip.",
        "tags": ["driftingspines", "retinalslip"],
    },
    "Spiral Illusion": {
        "it": "Spirale generativa modulata dalle bande di frequenza audio.",
        "en": "Generative spiral modulated by audio frequency bands.",
        "tags": ["spiral", "generativeart"],
    },
    "Zollner Illusion": {
        "it": "Illusione di Zollner: linee parallele apparentemente inclinate da segmenti trasversali.",
        "en": "Zollner illusion: parallel lines appear tilted due to crossing transversal segments.",
        "tags": ["zollner", "opticalillusion"],
    },
    "Cafe Wall": {
        "it": "Fraser (1908) / Gregory & Heard (1979) :: Cafe Wall. Righe di quadrati sfalsati con mortar line che appare inclinata.",
        "en": "Fraser (1908) / Gregory & Heard (1979) :: Cafe Wall. Offset square rows with a mortar line that appears tilted.",
        "tags": ["cafewall", "kitaoka"],
    },
    "Checkered": {
        "it": "Kitaoka (1998) / Lipps (1897) :: Checkered illusion. Scacchiera a bande sfasate, confine orizzontale percepito inclinato.",
        "en": "Kitaoka (1998) / Lipps (1897) :: Checkered illusion. Banded, phase-shifted checkerboard with a perceptually tilted horizontal border.",
        "tags": ["checkered", "kitaoka"],
    },
    "Shifted Edges": {
        "it": "Kitaoka, Pinna & Brelstaff (2001/2004) :: Illusion of shifted edges. Bordo a zig-zag tra bande che appare inclinato.",
        "en": "Kitaoka, Pinna & Brelstaff (2001/2004) :: Illusion of shifted edges. Zig-zag boundary between bands that appears tilted.",
        "tags": ["shiftededges", "kitaoka"],
    },
    "Fraser Twisted Cords": {
        "it": "Fraser (1908) :: Twisted cords. Corde ritorte bianco/nero su sfondo grigio, righe orizzontali percepite inclinate.",
        "en": "Fraser (1908) :: Twisted cords. Black/white twisted cords on a gray field, horizontal rows perceived as tilted.",
        "tags": ["fraser", "twistedcords"],
    },
    "Rotating Snakes": {
        "it": "Kitaoka & Ashida (2003) :: Rotating Snakes / Fraser-Wilcox. Gradino di luminanza a 4 livelli asimmetrico che genera rotazione illusoria spontanea.",
        "en": "Kitaoka & Ashida (2003) :: Rotating Snakes / Fraser-Wilcox. Asymmetric 4-level luminance step generating spontaneous illusory rotation.",
        "tags": ["rotatingsnakes", "fraserwilcox", "kitaoka"],
    },
    "Ouchi-Spillmann": {
        "it": "Ouchi (1977) / Spillmann :: Disco a scacchiera orizzontale su sfondo a scacchiera verticale, il centro sembra scivolare.",
        "en": "Ouchi (1977) / Spillmann :: Horizontally-checked disk on a vertically-checked field, the center appears to slide.",
        "tags": ["ouchispillmann"],
    },
    "Pinna-Brelstaff": {
        "it": "Pinna & Brelstaff (2000) :: Anelli di rettangoli obliqui a tilt invertito; lo zoom pulsato genera rotazione illusoria opposta tra anelli.",
        "en": "Pinna & Brelstaff (2000) :: Rings of oblique rectangles with inverted tilt; pulsed zoom generates opposite illusory rotation between rings.",
        "tags": ["pinnabrelstaff"],
    },
    "Hermann Grid": {
        "it": "Hermann (1870) :: Griglia chiara su sfondo scuro, macchie grigie fantasma alle intersezioni per inibizione laterale.",
        "en": "Hermann (1870) :: Light grid on dark background, ghost gray blobs at intersections from lateral inhibition.",
        "tags": ["hermanngrid"],
    },
    "Scintillating Grid": {
        "it": "Lingelbach & Schrauf (1994/1997) :: Dischi bianchi su griglia grigia, scintillano scuri se non fissati direttamente.",
        "en": "Lingelbach & Schrauf (1994/1997) :: White discs on a gray grid, scintillate dark when not directly fixated.",
        "tags": ["scintillatinggrid"],
    },
    "Kanizsa Triangle": {
        "it": "Kanizsa (1955) :: Contorni illusori. Terzetti di Pac-Man generano un triangolo bianco percepito che non esiste nei dati.",
        "en": "Kanizsa (1955) :: Illusory contours. Pac-Man triplets generate a perceived white triangle absent from the actual image data.",
        "tags": ["kanizsa", "illusorycontours"],
    },
    "Adelson Checkershadow": {
        "it": "Adelson (1995) :: Costanza di luminosita'. Un'ombra che scorre altera la luminosita' percepita di caselle identiche.",
        "en": "Adelson (1995) :: Lightness constancy. A sweeping shadow alters the perceived brightness of identically-valued squares.",
        "tags": ["checkershadow", "adelson"],
    },
}

def build_loop507_report(illusion_type, duration, fps, n_frames, size, bpm,
                          seed, intensity, size_factor, elements_factor, rotation_factor,
                          use_keyframes, video_title, report_number=0):
    science = ILLUSION_SCIENCE.get(illusion_type, {"it": "-", "en": "-", "tags": []})
    base_tags = [
        "vjing", "creativecoding", "generativeart", "reactiveaudio",
        "algorithmicmusic", "pythonart", "audiovisualart", "synesthesia",
        "proceduralaudio", "digitalartists", "abstractmotion", "visualalchemy",
    ]
    all_tags = base_tags + [t for t in science["tags"] if t not in base_tags]
    hashtags = " ".join(f"#{t}" for t in all_tags)
    kf_note_it = "Sequenza keyframe attiva (parametri interpolati nel tempo)." if use_keyframes else "Parametri statici (nessun keyframe)."
    kf_note_en = "Keyframe sequence active (parameters interpolated over time)." if use_keyframes else "Static parameters (no keyframes)."
    title_line = f'Titolo sovraimpresso: "{video_title.strip()}"' if video_title.strip() else "Nessun titolo sovraimpresso."
    title_line_en = f'Overlay Title: "{video_title.strip()}"' if video_title.strip() else "No Overlay Title."

    report = f"""VERSIONE ITALIANA :::

[VJING GENERATIVO] // VOL _ {report_number:02d} // 

Ogni fotogramma e' matematica pura che insegue il suono, nessuna rete neurale nel mezzo.

:: REPORT DI GENERAZIONE ::

Illusione     :: {illusion_type}
Base tecnica  :: {science['it']}
Formato       :: {size[0]}x{size[1]}px, {fps}fps, {n_frames} frame ({duration:.2f}s)
BPM rilevato  :: {bpm:.1f}
Seed          :: {seed}
Parametri     :: intensita={intensity:.2f} | dimensione={size_factor:.2f} | elementi={elements_factor:.2f} | rotazione={rotation_factor:.2f}
Keyframe      :: {kf_note_it}
Titolo        :: {title_line}
Sync audio    :: bassi->dimensione/ampiezza celle | medi->sfasamento/drift | acuti->spessore linee/microdettagli

Regia e Algoritmo: Loop507

{hashtags}

------

VERSIONE INGLESE:

[GENERATIVE VJING] // VOL _ {report_number:02d} //

Each frame is pure mathematics tracking the sound, no neural network in between.

:: GENERATION REPORT ::

Illusion :: {illusion_type}
Technical Base :: {science['en']}
Format :: {size[0]}x{size[1]}px, {fps}fps, {n_frames} frames ({duration:.2f}s)
Detected BPM :: {bpm:.1f}
Seed :: {seed}
Parameters :: intensity={intensity:.2f} | size={size_factor:.2f} | elements={elements_factor:.2f} | rotation={rotation_factor:.2f}
Keyframe :: {kf_note_en}
Title :: {title_line_en}
Audio Sync :: Bass -> Cell Size/Amplitude | Mid -> Phase Shift/Drift | Treble -> Line Thickness/Microdetails

Director and Algorithm: Loop507

{hashtags}
"""
    return report

def illusory_tilt_line_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    ILLUSORY TILT - Line-type (Kitaoka).
    Griglia di celle bowtie: triangolo superiore e inferiore a contrasto invertito,
    separati da una linea centrale. L'alternanza di polarita' a scacchiera,
    con sfasamento riga per riga, genera l'illusione di inclinazione della linea.
    Vettorizzato: niente doppio loop per-cella, solo un loop leggero sulle righe.
    pixel_scale riscala TUTTE le quantita' in pixel assoluti (base + termini
    audio-reattivi): serve per rendere l'anteprima a bassa risoluzione una
    vera miniatura del video finale, non solo la componente "base".
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_cell = 90.0
    cell = int((base_cell / num_elements_factor * element_size_factor + bass_val * 20 * intensity) * pixel_scale)
    cell = max(max(2, int(10 * pixel_scale)), cell)
    half_w = int(cell * 0.42)
    half_h = int(cell * 0.42)
    line_width = max(1, int((1 + high_val * 4 * intensity) * pixel_scale))
    row_shift = int(frame * 0.3 * rotation_speed_factor * (0.3 + mid_val))

    top_mask, bottom_mask, top_val_pixel = make_bowtie_tiles(width, height, cell, half_w, half_h, row_shift)
    bottom_val_pixel = 1.0 - top_val_pixel
    img = np.where(top_mask, top_val_pixel, np.where(bottom_mask, bottom_val_pixel, 0.0))

    for cy in range(cell // 2, height, cell):
        y0 = max(0, cy - line_width // 2)
        y1 = min(height, y0 + line_width)
        if y0 < height:
            img[y0:y1, :] = bottom_val_pixel[cy if cy < height else height - 1, :][np.newaxis, :]
    return img

def illusory_tilt_mixed_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    ILLUSORY TILT - Mixed-type (lines & edges).
    Stessa griglia bowtie del line-type, ma meta' delle celle mostra la linea
    centrale e meta' mostra solo il bordo di contrasto (edge), a scacchiera:
    combinazione "linee & edge" come nel pannello centrale del riferimento.
    Vettorizzato: niente doppio loop per-cella.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_cell = 90.0
    cell = int((base_cell / num_elements_factor * element_size_factor + bass_val * 20 * intensity) * pixel_scale)
    cell = max(max(2, int(10 * pixel_scale)), cell)
    half_w = int(cell * 0.42)
    half_h = int(cell * 0.42)
    line_width = max(1, int((1 + high_val * 4 * intensity) * pixel_scale))
    row_shift = int(frame * 0.3 * rotation_speed_factor * (0.3 + mid_val))

    top_mask, bottom_mask, top_val_pixel = make_bowtie_tiles(width, height, cell, half_w, half_h, row_shift)
    bottom_val_pixel = 1.0 - top_val_pixel
    img = np.where(top_mask, top_val_pixel, np.where(bottom_mask, bottom_val_pixel, 0.0))

    n_rows = height // cell + 2
    n_cols = width // cell + 2
    row_idx = np.arange(n_rows)[:, None]
    col_idx = np.arange(n_cols)[None, :]
    has_line_cells = ((row_idx + col_idx) % 2 == 0).astype(float)
    has_line_pixel = np.kron(has_line_cells, np.ones((cell, cell)))[:height, :width] > 0.5

    for cy in range(cell // 2, height, cell):
        y0 = max(0, cy - line_width // 2)
        y1 = min(height, y0 + line_width)
        if y0 < height:
            row_line = has_line_pixel[cy if cy < height else height - 1, :]
            row_val = bottom_val_pixel[cy if cy < height else height - 1, :]
            img[y0:y1, :] = np.where(row_line, row_val, img[y0:y1, :])
    return img

def illusory_tilt_edge_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    ILLUSORY TILT - Edge-type.
    Stessa griglia bowtie, senza linea: solo il bordo di contrasto tra i due
    triangoli genera l'inclinazione percepita ("=" geometrico nel riferimento).
    Completamente vettorizzato: nessun loop Python, solo operazioni numpy.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    base_cell = 90.0
    cell = int((base_cell / num_elements_factor * element_size_factor + mid_val * 25 * intensity) * pixel_scale)
    cell = max(max(2, int(10 * pixel_scale)), cell)
    half_w = int(cell * 0.42)
    half_h = int(cell * 0.42)
    row_shift = int(frame * 0.3 * rotation_speed_factor * (0.3 + bass_val))

    top_mask, bottom_mask, top_val_pixel = make_bowtie_tiles(width, height, cell, half_w, half_h, row_shift)
    bottom_val_pixel = 1.0 - top_val_pixel
    img = np.where(top_mask, top_val_pixel, np.where(bottom_mask, bottom_val_pixel, 0.0))
    return img

def illusory_motion_mather_line(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
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
    cell = int(base_cell / num_elements_factor * element_size_factor * pixel_scale)
    cell = max(max(4, int(20 * pixel_scale)), cell)
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

def illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
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
    cell = int((base_cell / num_elements_factor * element_size_factor + mid_val * 25 * intensity) * pixel_scale)
    cell = max(max(4, int(15 * pixel_scale)), cell)
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

def y_junctions_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0): # AGGIORNATO
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    base_square_size = 50.0
    square_size = int((base_square_size / num_elements_factor * element_size_factor + bass_val * 40 * intensity) * pixel_scale)
    square_size = max(max(1, int(1 * pixel_scale)), square_size)
    lateral_shift = int((frame * 0.5 * mid_val * intensity * rotation_speed_factor) % max(1, square_size)) # AGGIORNATO
    marker_arm = max(1, int(5 * pixel_scale))
    marker_arm_short = max(1, int(3 * pixel_scale))

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
                    rr, cc = line(jy-marker_arm, jx+d, jy+marker_arm, jx+d)
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid], cc[valid]] = 0.5
                    rr, cc = line(jy+d, jx-marker_arm, jy+d, jx+marker_arm)
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid], cc[valid]] = 0.5
                    rr, cc = line(jy-marker_arm_short, jx-marker_arm_short+d, jy+marker_arm_short, jx+marker_arm_short+d)
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    img[rr[valid], cc[valid]] = 0.5
    return img

def drifting_spines_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    DRIFTING SPINES ILLUSION.
    Texture densa di piccoli marcatori a farfalla (bowtie), come nel
    riferimento: ogni riga e' traslata orizzontalmente rispetto alla
    precedente (drift), producendo "retinal slip" e moto illusorio laterale.
    Vettorizzato: griglia costruita con make_bowtie_tiles, il drift per riga
    e' applicato con np.roll (un loop leggero sulle righe, non piu' sulle celle).
    """
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    tempo_factor = audio_features["tempo"] / 120.0

    base_spacing = 26.0
    spacing = int((base_spacing / num_elements_factor * element_size_factor + bass_val * 8 * intensity) * pixel_scale)
    spacing = max(max(2, int(6 * pixel_scale)), spacing)
    marker_half = max(1, int(spacing * 0.35))

    drift_speed = max(0.01, tempo_factor * intensity * rotation_speed_factor)
    drift_offset = (frame * drift_speed * 3) % spacing

    top_mask, bottom_mask, top_val_pixel = make_bowtie_tiles(width, height, spacing, marker_half, marker_half, 0)
    bottom_val_pixel = 1.0 - top_val_pixel
    base_img = np.where(top_mask, top_val_pixel, np.where(bottom_mask, bottom_val_pixel, 0.0))

    img = np.zeros((height, width), dtype=float)
    shift_int = int(round(drift_offset))
    for row_idx, y0 in enumerate(range(0, height, spacing)):
        y1 = min(height, y0 + spacing)
        shift = shift_int if row_idx % 2 == 0 else -shift_int
        img[y0:y1, :] = np.roll(base_img[y0:y1, :], shift, axis=1)

    for x in range(0, width, max(2, spacing // 2)):
        hy = int(height // 2 + 40 * pixel_scale * np.sin(x * 0.05 * rotation_speed_factor + drift_offset * 0.1))
        if 0 <= hy < height:
            radius = max(1, int((2 + high_val * 3 * intensity) * pixel_scale))
            rr, cc = disk((hy, x), radius, shape=(height, width))
            img[rr, cc] = 0.7
    return img

def spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0): # AGGIORNATO
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
                radius = max(1, int((2 + bass_val * 3) * pixel_scale))
                rr, cc = disk((y, x), radius, shape=(height, width))
                img[rr, cc] = intensity_val
    return img

def zollner_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0): # AGGIORNATO
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]
    
    base_spacing = 70.0
    line_spacing = int((base_spacing / num_elements_factor * element_size_factor + bass_val * 20 * intensity) * pixel_scale)
    line_spacing = max(max(1, int(1 * pixel_scale)), line_spacing)
    
    oblique_angle = np.radians(45 + mid_val * 45)
    
    horizontal_shift = int(high_val * 10 * rotation_speed_factor * pixel_scale) # AGGIORNATO

    for x in range(0, width + line_spacing, line_spacing):
        x_shifted = x + horizontal_shift
        rr, cc = line(0, x_shifted, height - 1, x_shifted)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        img[rr[valid], cc[valid]] = 1.0

        for y in range(0, height, max(1, int(line_spacing / 2))):
            length = int(10 * element_size_factor * pixel_scale)
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

def cafe_wall_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    CAFE WALL ILLUSION (Fraser 1908; Gregory & Heard 1979).
    File di quadrati neri/bianchi sfalsati di mezzo periodo tra righe
    adiacenti, separate da una sottile linea grigia ("mortar"): la mortar
    line appare inclinata anche se e' perfettamente orizzontale.
    Completamente vettorizzato (meshgrid + modulo), nessun loop Python.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_square = 60.0
    square_size = int((base_square / num_elements_factor * element_size_factor + bass_val * 20 * intensity) * pixel_scale)
    square_size = max(max(3, int(6 * pixel_scale)), square_size)
    mortar_width = max(1, int((1 + high_val * 3 * intensity) * pixel_scale))
    row_height = square_size + mortar_width
    drift = int(frame * 0.6 * rotation_speed_factor * (0.2 + mid_val))

    yv, xv = np.mgrid[0:height, 0:width]
    row_idx = yv // row_height
    within_row_y = yv % row_height
    is_mortar = within_row_y >= square_size
    row_shift = (square_size // 2) * (row_idx % 2) + drift
    col_pattern = ((xv + row_shift) // square_size) % 2
    img = np.where(is_mortar, 0.5, col_pattern.astype(float))
    return img

def checkered_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    CHECKERED / ENHANCED CHECKERED ILLUSION (Kitaoka 1998; Lipps 1897).
    Scacchiera classica divisa in bande orizzontali; ogni banda e' sfasata
    di un quarto di cella rispetto alla precedente, cosi' che il confine
    orizzontale tra bande appaia ripetutamente inclinato lungo lo schermo.
    Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    base_cell = 45.0
    cell = int((base_cell / num_elements_factor * element_size_factor + bass_val * 15 * intensity) * pixel_scale)
    cell = max(max(3, int(6 * pixel_scale)), cell)
    rows_per_band = max(1, int(3 * num_elements_factor))
    band_height = cell * rows_per_band
    drift = int(frame * 0.5 * rotation_speed_factor * (0.2 + mid_val))

    yv, xv = np.mgrid[0:height, 0:width]
    band_idx = yv // max(1, band_height)
    shift = (cell // 4) * (band_idx % 4) + drift
    col_idx = (xv + shift) // cell
    row_idx_local = yv // cell
    img = ((col_idx + row_idx_local) % 2).astype(float)
    return img

def shifted_edges_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    ILLUSION OF SHIFTED EDGES (Kitaoka, Pinna & Brelstaff, 2001/2004).
    Bande orizzontali bianco/nero il cui confine e' spostato in verticale a
    zig-zag (colonne alterne), producendo un confine percepito come
    inclinato pur essendo orizzontale in media. Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    n_bands = max(2, int(6 * num_elements_factor))
    band_h = max(4, int(height / n_bands))
    block_w = int((40.0 * element_size_factor + mid_val * 30 * intensity) * pixel_scale)
    block_w = max(max(2, int(4 * pixel_scale)), block_w)
    shift_amount = max(1, int((2 + bass_val * (band_h * 0.35) * intensity) * pixel_scale))
    drift = int(frame * 0.4 * rotation_speed_factor * (0.2 + high_val))

    yv, xv = np.mgrid[0:height, 0:width]
    band_idx = yv // band_h
    local_y = yv - band_idx * band_h
    col_group = (xv + drift) // block_w
    sign = np.where(col_group % 2 == 0, 1, -1)
    boundary_local = band_h // 2 + shift_amount * sign
    top_val = (band_idx % 2 == 0).astype(float)
    bottom_val = 1.0 - top_val
    img = np.where(local_y < boundary_local, top_val, bottom_val)
    return img

def fraser_twisted_cords_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    FRASER TWISTED CORDS (Fraser, 1908).
    Righe di "corde ritorte": segmenti diagonali bianco/nero alternati su
    sfondo grigio, che fanno apparire inclinate righe in realta' orizzontali.
    Template di un periodo costruito una volta per frame con formule
    vettorizzate, poi tassellato con np.tile; drift orizzontale via np.roll
    per riga (stesso pattern di drifting_spines_illusion).
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_period = 40.0
    period = int((base_period / num_elements_factor * element_size_factor) * pixel_scale)
    period = max(max(4, int(8 * pixel_scale)), period)
    base_band_h = 30.0
    band_h = int((base_band_h * element_size_factor) * pixel_scale)
    band_h = max(max(4, int(8 * pixel_scale)), band_h)
    tilt_px = int((period * 0.5) * (0.4 + bass_val * intensity))
    thickness = max(1, int((2 + high_val * 3 * intensity) * pixel_scale))
    drift_speed = max(0.01, mid_val * intensity * rotation_speed_factor)
    drift_offset = (frame * drift_speed * 4) % period

    yy, xx = np.mgrid[0:band_h, 0:period]
    line1_x = (tilt_px * yy) / max(1, band_h)
    mask1 = np.abs(xx - line1_x) < max(1, thickness) / 2.0
    line2_x = period / 2.0 + (tilt_px * yy) / max(1, band_h)
    line2_x = line2_x % period
    mask2 = np.abs(xx - line2_x) < max(1, thickness) / 2.0

    template = np.full((band_h, period), 0.5, dtype=float)
    template[mask1] = 1.0
    template[mask2] = 0.0

    n_rows = height // band_h + 2
    n_cols = width // period + 2
    base_img = np.tile(template, (n_rows, n_cols))[:height, :width]

    img = np.zeros((height, width), dtype=float)
    shift_int = int(round(drift_offset))
    for row_idx, y0 in enumerate(range(0, height, band_h)):
        y1 = min(height, y0 + band_h)
        shift = shift_int if row_idx % 2 == 0 else -shift_int
        img[y0:y1, :] = np.roll(base_img[y0:y1, :], shift, axis=1)
    return img

def rotating_snakes_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    ROTATING SNAKES / FRASER-WILCOX ILLUSION (Kitaoka & Ashida, 2003; Fraser
    & Wilcox, 1979). Anelli concentrici con un gradino di luminanza a 4
    livelli asimmetrico (nero, grigio scuro, bianco, grigio chiaro) che
    attiva i rilevatori di direzione/moto della corteccia visiva, generando
    la percezione di rotazione spontanea pur essendo il pattern statico.
    Anelli adiacenti hanno il "dente di sega" invertito (direzione opposta),
    come nella figura classica multi-anello di Kitaoka.
    Completamente vettorizzato in coordinate polari.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    cx, cy = width / 2.0, height / 2.0
    base_ring_width = 45.0
    ring_width = max(max(3, int(6 * pixel_scale)), int((base_ring_width * element_size_factor + bass_val * 15 * intensity) * pixel_scale))
    n_segments = max(4, int(14 * num_elements_factor))

    yv, xv = np.mgrid[0:height, 0:width]
    dx = xv - cx
    dy = yv - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    ring_idx = (r // ring_width).astype(int)
    direction = np.where(ring_idx % 2 == 0, 1.0, -1.0)
    spin = frame * 0.06 * rotation_speed_factor * (0.3 + mid_val) + high_val * 0.15
    segment_angle = 2 * np.pi / n_segments
    local_theta = (theta * direction + spin) % segment_angle
    frac = local_theta / segment_angle

    # Gradino asimmetrico a 4 livelli (proporzioni tipiche Fraser-Wilcox)
    b0, b1, b2 = 0.12, 0.5, 0.62
    img = np.select(
        [frac < b0, frac < b1, frac < b2],
        [0.0, 0.35, 1.0],
        default=0.68,
    )
    return img

def ouchi_spillmann_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    OUCHI-SPILLMANN ILLUSION (Ouchi, 1977; Spillmann, 2013).
    Disco centrale con scacchiera "orizzontale" (rettangoli larghi e bassi)
    incastonato in uno sfondo con scacchiera "verticale" (rettangoli alti e
    stretti): il centro appare scivolare/tremolare rispetto allo sfondo.
    Un piccolo jitter di posizione per frame esagera l'effetto per il video
    (l'illusione originale dipende dai micro-movimenti oculari).
    Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    cx, cy = width / 2.0, height / 2.0
    base_radius = min(width, height) * 0.28
    center_radius = max(max(4, int(8 * pixel_scale)), int((base_radius * element_size_factor + bass_val * 20 * intensity) * pixel_scale))

    base_block = 26.0
    block_short = max(max(2, int(4 * pixel_scale)), int((base_block / num_elements_factor) * pixel_scale))
    block_long = max(block_short * 2, int(block_short * (2.2 + mid_val)))

    jitter = int(3 * pixel_scale * intensity * np.sin(frame * 0.5 * rotation_speed_factor + high_val * 3))

    yv, xv = np.mgrid[0:height, 0:width]
    dx = xv - cx
    dy = yv - cy
    is_center = (dx * dx + dy * dy) <= (center_radius * center_radius)

    # scacchiera centrale: rettangoli larghi (block_long) e bassi (block_short)
    center_pattern = (((xv + jitter) // block_long) + (yv // block_short)) % 2

    # scacchiera esterna: rettangoli alti (block_long) e stretti (block_short)
    outer_pattern = ((xv // block_short) + ((yv + jitter) // block_long)) % 2

    img = np.where(is_center, center_pattern, outer_pattern).astype(float)
    return img

def pinna_brelstaff_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    PINNA-BRELSTAFF ILLUSION (Pinna & Brelstaff, 2000).
    Anelli concentrici di piccoli rettangoli obliqui, con tilt invertito tra
    anello e anello. Nel fenomeno reale, avvicinandosi/allontanandosi dalla
    figura gli anelli sembrano ruotare in direzioni opposte: qui lo zoom
    reale e' animato nel tempo (pulsato dai bassi), che e' esattamente il
    trigger fisico del fenomeno originale.
    """
    img = np.zeros((height, width), dtype=float)
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    cx, cy = width / 2.0, height / 2.0
    n_rings = max(2, int(3 * num_elements_factor))
    base_ring_gap = 55.0
    ring_gap = max(max(4, int(8 * pixel_scale)), int((base_ring_gap * element_size_factor) * pixel_scale))

    zoom = 1.0 + 0.35 * intensity * np.sin(frame * 0.08 * rotation_speed_factor * (0.4 + bass_val))
    n_per_ring = max(6, int(16 * num_elements_factor))
    rect_w = max(1, int((ring_gap * 0.45) * pixel_scale))
    rect_h = max(1, int((ring_gap * 0.9) * pixel_scale))
    tilt = np.radians(35 + mid_val * 15)

    for ring_i in range(1, n_rings + 1):
        radius = ring_i * ring_gap * zoom
        ring_tilt = tilt if ring_i % 2 == 0 else -tilt
        val = 1.0 if ring_i % 2 == 0 else 0.68
        for j in range(n_per_ring):
            angle = 2 * np.pi * j / n_per_ring + high_val * 0.3
            ex, ey = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
            cos_t, sin_t = np.cos(ring_tilt), np.sin(ring_tilt)
            local = np.array([
                [-rect_w / 2, -rect_h / 2], [rect_w / 2, -rect_h / 2],
                [rect_w / 2, rect_h / 2], [-rect_w / 2, rect_h / 2],
            ])
            rotated = np.array([
                [p[0] * cos_t - p[1] * sin_t + ex, p[0] * sin_t + p[1] * cos_t + ey]
                for p in local
            ]).astype(int)
            rr, cc = polygon(rotated[:, 1], rotated[:, 0], (height, width))
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            img[rr[valid], cc[valid]] = val
    return img

def hermann_grid_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    HERMANN GRID ILLUSION (Hermann, 1870).
    Griglia di barre chiare su sfondo scuro: alle intersezioni appaiono
    macchie grigie fantasma (inibizione laterale retinica), che spariscono
    se si fissa direttamente l'incrocio. Spaziatura e spessore audio-reattivi
    modulano quanto le macchie fantasma risultano percettivamente forti.
    Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_cell = 55.0
    cell = max(max(4, int(8 * pixel_scale)), int((base_cell / num_elements_factor * element_size_factor + bass_val * 15 * intensity) * pixel_scale))
    bar_width = max(1, int((cell * 0.18 + high_val * 4 * intensity) * pixel_scale))
    drift = int(frame * 0.3 * rotation_speed_factor)

    yv, xv = np.mgrid[0:height, 0:width]
    local_x = (xv + drift) % cell
    local_y = (yv + drift) % cell
    img = ((local_x < bar_width) | (local_y < bar_width)).astype(float)
    return img

def scintillating_grid_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    SCINTILLATING GRID ILLUSION (Lingelbach & Schrauf, 1994; Schrauf, Lingelbach
    & Wist, 1997). Griglia grigia su sfondo nero con dischi bianchi alle
    intersezioni: i dischi sembrano "scintillare" scuri quando non fissati
    direttamente. Qui il flicker per-disco e' animato esplicitamente (fase
    diversa per ogni intersezione) per rendere l'effetto visibile in video.
    Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_cell = 60.0
    cell = max(max(6, int(10 * pixel_scale)), int((base_cell / num_elements_factor * element_size_factor + bass_val * 15 * intensity) * pixel_scale))
    bar_width = max(1, int(cell * 0.14 * pixel_scale))
    base_radius = cell * 0.28

    yv, xv = np.mgrid[0:height, 0:width]
    local_x = xv % cell
    local_y = yv % cell
    dx = np.minimum(local_x, cell - local_x)
    dy = np.minimum(local_y, cell - local_y)

    cell_row = yv // cell
    cell_col = xv // cell
    flicker_speed = 0.15 * rotation_speed_factor * (0.3 + high_val)
    phase = np.sin(frame * flicker_speed + (cell_row * 7 + cell_col * 13).astype(float))
    radius = base_radius * (0.7 + 0.3 * (0.5 + 0.5 * phase) * (0.5 + mid_val))

    dist2 = dx * dx + dy * dy
    disc_mask = dist2 < (radius * radius)
    grid_mask = (local_x < bar_width) | (local_y < bar_width)
    img = np.where(disc_mask, 1.0, np.where(grid_mask, 0.5, 0.0))
    return img

def kanizsa_triangle_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    KANIZSA TRIANGLE ILLUSION (Kanizsa, 1955).
    Terzetti di "Pac-Man" con lo spicchio mancante rivolto verso il centro
    del gruppo: il cervello completa i bordi mancanti percependo un
    triangolo bianco illusorio che non esiste nei dati dell'immagine.
    Il gruppo intero ruota nel tempo (audio-reattivo); il template di una
    singola cella viene costruito una volta per frame e tassellato con
    np.tile, stesso pattern di make_bowtie_tiles.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    base_tile = 130.0
    tile = max(max(10, int(20 * pixel_scale)), int((base_tile / num_elements_factor * element_size_factor) * pixel_scale))
    pac_radius = tile * 0.22
    orbit_radius = tile * 0.26
    wedge_half_angle = np.radians(28 + bass_val * 20 * intensity)
    rotation = frame * 0.04 * rotation_speed_factor * (0.3 + mid_val)

    tcx, tcy = tile / 2.0, tile / 2.0
    yy, xx = np.mgrid[0:tile, 0:tile]
    pac_mask = np.zeros((tile, tile), dtype=bool)
    for k in range(3):
        angle_center = rotation + 2 * np.pi * k / 3
        pcx = tcx + orbit_radius * np.cos(angle_center)
        pcy = tcy + orbit_radius * np.sin(angle_center)
        dx = xx - pcx
        dy = yy - pcy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        cut_dir = angle_center + np.pi  # lo spicchio guarda verso il centro del gruppo
        ang_diff = np.mod(theta - cut_dir + np.pi, 2 * np.pi) - np.pi
        within_wedge = np.abs(ang_diff) < wedge_half_angle
        pac_mask |= (r < pac_radius) & (~within_wedge)

    template = pac_mask.astype(float)
    n_rows = height // tile + 2
    n_cols = width // tile + 2
    img = np.tile(template, (n_rows, n_cols))[:height, :width]
    return img

def adelson_checkershadow_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    ADELSON CHECKERSHADOW ILLUSION (Adelson, 1995).
    Scacchiera a valori di grigio fissi, attraversata da una banda d'ombra
    morbida che ne altera la luminosita' percepita: caselle di identico
    valore fisico appaiono diverse a seconda che siano dentro o fuori
    l'ombra (costanza di luminosita'). La banda scorre nel tempo, pilotata
    dai bassi. Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    base_cell = 50.0
    cell = max(max(4, int(8 * pixel_scale)), int((base_cell / num_elements_factor * element_size_factor) * pixel_scale))

    yv, xv = np.mgrid[0:height, 0:width]
    checker = ((xv // cell + yv // cell) % 2).astype(float)
    base_val = 0.22 + 0.56 * checker

    angle = np.radians(25 + mid_val * 30)
    diag = width * np.cos(angle) + height * np.sin(angle)
    band_center = ((frame * (0.8 + bass_val * 2.5) * rotation_speed_factor) % (diag * 1.6)) - diag * 0.3
    band_width = max(40.0, diag * (0.28 + high_val * 0.12))

    dist_along = xv * np.cos(angle) + yv * np.sin(angle) - band_center
    shadow_factor = 1.0 - 0.45 * intensity * np.exp(-(dist_along ** 2) / (2 * (band_width / 2) ** 2))

    img = base_val * shadow_factor
    return img

def generate_illusion_frame(width, height, frame, audio_features, intensity, illusion_type, seed, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0): # AGGIORNATO
    np.random.seed(seed + frame)

    if illusion_type == "Illusory Tilt (Line)":
        img = illusory_tilt_line_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Illusory Tilt (Mixed)":
        img = illusory_tilt_mixed_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Illusory Tilt (Edge)":
        img = illusory_tilt_edge_type(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Illusory Motion (Mather)":
        img = illusory_motion_mather_line(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Illusory Motion (Takeuchi)":
        img = illusory_motion_takeuchi_mixed(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Y-Junctions":
        img = y_junctions_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Drifting Spines":
        img = drifting_spines_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Spiral Illusion":
        img = spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Zollner Illusion":
        img = zollner_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Cafe Wall":
        img = cafe_wall_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Checkered":
        img = checkered_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Shifted Edges":
        img = shifted_edges_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Fraser Twisted Cords":
        img = fraser_twisted_cords_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Rotating Snakes":
        img = rotating_snakes_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Ouchi-Spillmann":
        img = ouchi_spillmann_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Pinna-Brelstaff":
        img = pinna_brelstaff_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Hermann Grid":
        img = hermann_grid_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Scintillating Grid":
        img = scintillating_grid_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Kanizsa Triangle":
        img = kanizsa_triangle_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Adelson Checkershadow":
        img = adelson_checkershadow_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    else:
        img = spiral_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    return apply_colors(img, line_color, bg_color)

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
# INTERFACCIA STREAMLIT E LOGICA APPLICATIVA
# ---------------------------------

st.set_page_config(page_title="VJing Generativo", layout="wide")

st.title("🎵 VJing Generativo - Illusioni Ottiche Scientifiche")

st.caption("by Loop507 | Arte cinetica sincronizzata al suono con implementazioni neuropsicologiche accurate")

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
        "Y-Junctions", "Drifting Spines", "Spiral Illusion", "Zollner Illusion",
        "Cafe Wall", "Checkered", "Shifted Edges", "Fraser Twisted Cords",
        "Rotating Snakes", "Ouchi-Spillmann", "Pinna-Brelstaff",
        "Hermann Grid", "Scintillating Grid", "Kanizsa Triangle", "Adelson Checkershadow",
    ]
)

# ---------------------------------
# GRIGLIA DI MINIATURE :: una foto piccola per ogni effetto, cosi' si
# riconosce subito il pattern senza doverselo ricordare a memoria.
# Cachate (per nome + colori correnti) cosi' non vengono rigenerate ad
# ogni rerun dello script, solo quando cambiano davvero i colori.
# ---------------------------------
@st.cache_data(show_spinner=False)
def _generate_illusion_thumbnail(illusion_name, line_hex, bg_hex, thumb_w=110, thumb_h=64):
    neutral_audio = {
        "tempo": 120.0,
        "bass": np.full(10, 0.5), "mid": np.full(10, 0.5), "high": np.full(10, 0.5),
    }
    thumb_pixel_scale = thumb_w / 1280.0
    img = generate_illusion_frame(
        thumb_w, thumb_h, 6, neutral_audio, 1.0, illusion_name, 7,
        1.0, 1.0, 1.0, pixel_scale=thumb_pixel_scale,
    )
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

with st.expander("🖼️ Anteprima di tutti gli effetti disponibili", expanded=False):
    _thumb_names = [
        "Illusory Tilt (Line)", "Illusory Tilt (Mixed)", "Illusory Tilt (Edge)",
        "Illusory Motion (Mather)", "Illusory Motion (Takeuchi)",
        "Y-Junctions", "Drifting Spines", "Spiral Illusion", "Zollner Illusion",
        "Cafe Wall", "Checkered", "Shifted Edges", "Fraser Twisted Cords",
        "Rotating Snakes", "Ouchi-Spillmann", "Pinna-Brelstaff",
        "Hermann Grid", "Scintillating Grid", "Kanizsa Triangle", "Adelson Checkershadow",
    ]
    _thumb_cols = st.columns(4)
    for _i, _name in enumerate(_thumb_names):
        _thumb = _generate_illusion_thumbnail(_name, line_color, bg_color)
        with _thumb_cols[_i % 4]:
            st.image(_thumb, caption=_name, use_container_width=True)

st.sidebar.subheader("🎥 Sequenza Keyframe (avanzato)")

use_keyframes = st.sidebar.checkbox("Usa Sequenza Keyframe", value=False)

keyframes_intensity = {}

keyframes_size = {}

keyframes_elements = {}

keyframes_rotation = {}

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
                    st.sidebar.warning(f"Formato keyframe non valido: '{kf_line}'. Ignorato.")
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
    rotation_speed_factor = st.sidebar.slider("🔄 Velocità Rotazione", 0.0, 2.0, 1.0, 0.1)

st.sidebar.subheader("📝 Titolo Video")

video_title = st.text_input("Testo del titolo", "")

font_size = st.sidebar.slider("Grandezza carattere", 20, 100, 48, 2)

vertical_position = st.sidebar.selectbox("Posizione verticale", ["Sopra", "Sotto", "Centro"])

horizontal_position = st.sidebar.selectbox("Posizione orizzontale", ["Sinistra", "Destra", "Centro"])

aspect_ratio = st.selectbox("📺 Formato video", ["16:9", "1:1", "9:16"])

st.subheader("🔍 Anteprima rapida animata (bassa risoluzione)")

PREVIEW_MAX_DIM = 220

PREVIEW_FPS = 8

PREVIEW_DURATION_SEC = 2.0

PREVIEW_N_FRAMES = max(4, int(PREVIEW_FPS * PREVIEW_DURATION_SEC))

if st.button("👁️ Genera anteprima"):
    # Stessa mappatura risoluzione usata nel render completo piu' sotto:
    # la preview deve essere una VERA miniatura proporzionale, non un
    # pattern a griglia diversa solo perche' i pixel assoluti sono pochi.
    if aspect_ratio == "16:9": full_target_size = (1280, 720)
    elif aspect_ratio == "1:1": full_target_size = (720, 720)
    else: full_target_size = (720, 1280)

    if aspect_ratio == "16:9":
        preview_size = (PREVIEW_MAX_DIM, int(PREVIEW_MAX_DIM * 9 / 16))
    elif aspect_ratio == "1:1":
        preview_size = (PREVIEW_MAX_DIM, PREVIEW_MAX_DIM)
    else:
        preview_size = (int(PREVIEW_MAX_DIM * 9 / 16), PREVIEW_MAX_DIM)

    preview_scale = preview_size[0] / full_target_size[0]

    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in (".wav", ".mp3"): ext = ".wav"
        tmp_preview_audio = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_preview_audio.write(uploaded_file.getvalue())
        tmp_preview_audio.close()
        y_prev, sr_prev = librosa.load(tmp_preview_audio.name, sr=None, duration=PREVIEW_DURATION_SEC)
        duration_prev = float(librosa.get_duration(y=y_prev, sr=sr_prev))
        preview_features = analyze_audio(tmp_preview_audio.name, max(duration_prev, 0.5), PREVIEW_FPS)
        os.remove(tmp_preview_audio.name)
    else:
        st.caption("Nessun audio caricato: anteprima con valori audio neutri.")
        preview_features = {
            "tempo": 120.0,
            "bass": np.full(PREVIEW_N_FRAMES, 0.5),
            "mid": np.full(PREVIEW_N_FRAMES, 0.5),
            "high": np.full(PREVIEW_N_FRAMES, 0.5),
        }

    preview_intensity = intensity
    preview_size_factor = element_size_factor
    preview_elements_factor = num_elements_factor
    preview_rotation_factor = rotation_speed_factor
    if use_keyframes:
        if keyframes_intensity:
            v = interpolate_value(0.0, keyframes_intensity)
            if v is not None: preview_intensity = v
        if keyframes_size:
            v = interpolate_value(0.0, keyframes_size)
            if v is not None: preview_size_factor = v
        if keyframes_elements:
            v = interpolate_value(0.0, keyframes_elements)
            if v is not None: preview_elements_factor = v
        if keyframes_rotation:
            v = interpolate_value(0.0, keyframes_rotation)
            if v is not None: preview_rotation_factor = v

    # pixel_scale riscala TUTTE le quantita' in pixel assoluti (base +
    # termini audio-reattivi) dentro le funzioni di illusione: senza,
    # a bassa risoluzione la griglia sarebbe piu' "grossa" (meno celle)
    # invece di essere una miniatura fedele del video finale.
    preview_pixel_scale = preview_scale

    preview_seed = random.randint(1, 10000)
    FULL_RENDER_FPS = 30  # deve combaciare con l'fps del render completo piu' sotto
    with st.spinner(f"Generazione anteprima ({PREVIEW_N_FRAMES} frame, {preview_size[0]}×{preview_size[1]}px)..."):
        preview_frames = []
        for pf in range(PREVIEW_N_FRAMES):
            # mappa il frame dell'anteprima allo stesso ritmo temporale reale
            # del video finale (altrimenti il moto sembra rallentato/statico)
            real_frame_equiv = int(pf * FULL_RENDER_FPS / PREVIEW_FPS)
            frame_img = generate_illusion_frame(
                preview_size[0], preview_size[1], real_frame_equiv, preview_features,
                preview_intensity, illusion_type, preview_seed,
                preview_size_factor, preview_elements_factor, preview_rotation_factor,
                pixel_scale=preview_pixel_scale,
            )
            frame_uint8 = (np.clip(frame_img, 0.0, 1.0) * 255).astype(np.uint8)
            preview_frames.append(Image.fromarray(frame_uint8))

        gif_buffer = io.BytesIO()
        preview_frames[0].save(
            gif_buffer, format="GIF", save_all=True,
            append_images=preview_frames[1:], duration=int(1000 / PREVIEW_FPS),
            loop=0,
        )
        gif_buffer.seek(0)

    st.image(
        gif_buffer.getvalue(),
        caption=f"Anteprima animata {preview_size[0]}×{preview_size[1]}px, {PREVIEW_N_FRAMES} frame @ {PREVIEW_FPS}fps — {illusion_type}",
        width=preview_size[0] * 2,
    )
    st.caption("Anteprima a bassa risoluzione/fps: il video finale sara' generato alla risoluzione e framerate pieni (30fps).")

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

    # Leggo i byte in memoria e li salvo in session_state: un download_button
    # (compreso quello del report) fa ripartire da capo lo script, e senza
    # session_state tutto cio' che era dentro "if st.button(...)" sparirebbe
    # (bottone di generazione tornato "non premuto"), video incluso.
    with open(output_file.name, "rb") as f:
        video_bytes = f.read()

    report_counter = st.session_state.get("loop507_report_counter", 0)
    report_text = build_loop507_report(
        illusion_type=illusion_type, duration=duration, fps=fps, n_frames=n_frames,
        size=size, bpm=tempo_display, seed=seed,
        intensity=intensity, size_factor=element_size_factor,
        elements_factor=num_elements_factor, rotation_factor=rotation_speed_factor,
        use_keyframes=use_keyframes, video_title=video_title, report_number=report_counter,
    )
    st.session_state["loop507_report_counter"] = report_counter + 1
    safe_name = illusion_type.lower().replace(" ", "_").replace("(", "").replace(")", "")

    st.session_state["loop507_video_bytes"] = video_bytes
    st.session_state["loop507_video_filename"] = f"vjing_{safe_name}_output.mp4"
    st.session_state["loop507_report_text"] = report_text
    st.session_state["loop507_report_filename"] = f"loop507_report_{safe_name}.txt"

    try:
        os.remove(tmp_audio.name)
        os.remove(tmp_video.name)
        os.remove(output_file.name)
    except Exception:
        pass

if "loop507_video_bytes" in st.session_state:
    st.success("✨ Video generato con successo! Implementazioni neuropsicologiche accurate.")
    st.download_button(
        "📥 Scarica Video Illusorio Scientifico",
        st.session_state["loop507_video_bytes"],
        file_name=st.session_state["loop507_video_filename"],
        mime="video/mp4",
        key="download_video_btn",
    )
    st.text(st.session_state["loop507_report_text"])
    st.download_button(
        "📄 Scarica Report Bilingue (.txt)",
        st.session_state["loop507_report_text"],
        file_name=st.session_state["loop507_report_filename"],
        mime="text/plain",
        key="download_report_btn",
    )
