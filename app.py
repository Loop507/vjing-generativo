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
    "Motion Silencing": {
        "it": "Suchow & Alvarez (2011, Current Biology) :: Motion Silencing. Un anello di elementi che oscillano in luminanza smette di apparire mutevole quando ruota velocemente ('silencing of awareness to visual change').",
        "en": "Suchow & Alvarez (2011, Current Biology) :: Motion Silencing. A ring of luminance-oscillating elements stops appearing to change once it rotates fast enough ('silencing of awareness to visual change').",
        "tags": ["motionsilencing", "suchowalvarez"],
    },
    "Enigma Illusion": {
        "it": "Leviant (1996) :: Enigma. Anelli concentrici con texture radiale a denti di sega; il moto serpeggiante illusorio e' legato ai micro-movimenti oculari fissazionali (microsaccadi), qui simulati come jitter angolare.",
        "en": "Leviant (1996) :: Enigma. Concentric rings with a sawtooth radial texture; the illusory serpentine motion is tied to fixational eye movements (microsaccades), here simulated as angular jitter.",
        "tags": ["enigma", "leviant", "microsaccades"],
    },
    "Barberpole Illusion": {
        "it": "Wallach (1935); Wuerger, Shapley & Rubin (1996) :: Barberpole / aperture problem. Strisce diagonali viste attraverso un'apertura allungata sembrano scorrere lungo l'asse lungo dell'apertura, non nella direzione fisica reale.",
        "en": "Wallach (1935); Wuerger, Shapley & Rubin (1996) :: Barberpole / aperture problem. Diagonal stripes seen through an elongated aperture appear to travel along the aperture's long axis rather than their true physical direction.",
        "tags": ["barberpole", "apertureproblem"],
    },
    "White's Illusion": {
        "it": "White (1979) :: Lightness illusion. Toppe grigie identiche su una griglia a bande bianche/nere appaiono di luminosita' diversa, in direzione opposta a quella predetta dal semplice contrasto simultaneo.",
        "en": "White (1979) :: Lightness illusion. Identical gray patches on a black/white striped grating appear to differ in brightness, in the opposite direction predicted by simple simultaneous contrast.",
        "tags": ["whitesillusion", "lightness"],
    },
    "Poggendorff Illusion": {
        "it": "Poggendorff (1860) :: Due segmenti diagonali collineari, interrotti da due bande verticali occludenti, appaiono disallineati pur essendo perfettamente collineari.",
        "en": "Poggendorff (1860) :: Two collinear diagonal segments, interrupted by two occluding vertical bands, appear misaligned despite being perfectly collinear.",
        "tags": ["poggendorff", "geometricillusion"],
    },
    "Rubber Pencil Illusion": {
        "it": "Pomerantz (1983); Macknik & Martinez-Conde (2008, Nat. Rev. Neurosci.) :: Un'asta rigida oscillata rapidamente appare flettersi come gomma, per la risposta differenziale dei neuroni end-stopped (V1/MT) tra estremita' e centro dello stimolo in moto.",
        "en": "Pomerantz (1983); Macknik & Martinez-Conde (2008, Nat. Rev. Neurosci.) :: A rigid bar oscillated rapidly appears to flex like rubber, due to the differential response of end-stopped neurons (V1/MT) between the stimulus' endpoints and its center during motion.",
        "tags": ["rubberpencil", "endstoppedneurons"],
    },
    "Wagon Wheel Illusion": {
        "it": "Purves, Paydarfar & Andrews (1996, PNAS) :: Un disco a raggi che ruota sopra una soglia critica di velocita' viene percepito invertire il senso di rotazione, anche in luce continua (campionamento discreto della percezione).",
        "en": "Purves, Paydarfar & Andrews (1996, PNAS) :: A spoked disk rotating past a critical speed threshold is perceived to reverse its direction of rotation, even under continuous illumination (discrete sampling of perception).",
        "tags": ["wagonwheel", "purves"],
    },
    "Motion Aftereffect": {
        "it": "Addams (1834); Mather, Verstraten & Anstis (1998, 'The Motion Aftereffect') :: Dopo un adattamento prolungato a un moto direzionale, un pattern statico appare muoversi nella direzione opposta.",
        "en": "Addams (1834); Mather, Verstraten & Anstis (1998, 'The Motion Aftereffect') :: After prolonged adaptation to directional motion, a static pattern appears to move in the opposite direction.",
        "tags": ["motionaftereffect", "waterfallillusion"],
    },
    "Flash-Lag Effect": {
        "it": "Nijhawan (1994, Nature) :: Un oggetto in moto continuo e un flash statico co-localizzato nello stesso istante vengono percepiti disallineati: l'oggetto in moto appare in anticipo.",
        "en": "Nijhawan (1994, Nature) :: A continuously moving object and a co-located static flash presented at the same instant are perceived as misaligned: the moving object appears to lead.",
        "tags": ["flashlag", "nijhawan"],
    },
    "Line Motion Illusion": {
        "it": "Hikosaka, Miyauchi & Shimojo (1993, Vision Research) :: Un cue attenzionale a un'estremita', seguito dall'apparizione istantanea di una linea statica, genera la percezione che la linea si stia disegnando in movimento.",
        "en": "Hikosaka, Miyauchi & Shimojo (1993, Vision Research) :: An attentional cue at one end, followed by the instantaneous appearance of a static line, generates the perception of the line drawing itself in motion.",
        "tags": ["linemotion", "hikosaka"],
    },
    "Motion-Induced Blindness": {
        "it": "Bonneh, Cooperman & Sagi (2001, Nature) :: Bersagli statici salienti circondati da un pattern globale in movimento scompaiono e ricompaiono periodicamente dalla consapevolezza durante la fissazione.",
        "en": "Bonneh, Cooperman & Sagi (2001, Nature) :: Salient static targets surrounded by a moving global pattern periodically disappear and reappear from awareness during fixation.",
        "tags": ["motioninducedblindness", "bonneh"],
    },
    "Troxler Fading": {
        "it": "Troxler (1804) :: Bersagli periferici statici su un campo omogeneo scompaiono dalla consapevolezza durante la fissazione prolungata del centro (adattamento neurale periferico).",
        "en": "Troxler (1804) :: Static peripheral targets on a homogeneous field fade from awareness during prolonged central fixation (peripheral neural adaptation).",
        "tags": ["troxlerfading"],
    },
    "McCollough Effect": {
        "it": "McCollough (1965, Science) :: Adattamento a griglie con colore contingente all'orientamento; griglie acromatiche successive appaiono tinte del colore associato a quell'orientamento.",
        "en": "McCollough (1965, Science) :: Adaptation to orientation-contingent colored gratings; subsequent achromatic gratings appear tinted with the color associated with that orientation.",
        "tags": ["mccollougheffect", "colorcontingent"],
    },
    "Phi/Beta Movement": {
        "it": "Wertheimer (1912) :: Due stimoli vicini mostrati in alternanza generano una sensazione di moto la cui natura dipende dall'intervallo interstimolo (ISI): a ISI brevi si vede un oggetto spostarsi (beta movement, base percettiva del cinema); a ISI cortissimi tende al 'puro phi' (senso di moto senza oggetto). L'ISI qui e' pilotato direttamente dal BPM.",
        "en": "Wertheimer (1912) :: Two nearby stimuli shown in alternation generate a sensation of motion whose nature depends on the interstimulus interval (ISI): short ISIs yield a perceived moving object (beta movement, the perceptual basis of cinema); very short ISIs tend toward 'pure phi' (a sense of movement without an object). ISI here is driven directly by the track's BPM.",
        "tags": ["phiphenomenon", "betamovement", "wertheimer"],
    },
    "Ternus Illusion": {
        "it": "Ternus (1926); Pantle & Picciano (1976) :: Tre elementi mostrati in due fotogrammi separati da un breve ISI generano una percezione di moto ambigua: ISI corti (<~30-50ms) danno 'element motion' (un elemento salta all'altro capo), ISI piu' lunghi danno 'group motion' (tutti scorrono insieme). L'ISI qui e' derivato dal BPM, permettendo di attraversare dal vivo la soglia percettiva.",
        "en": "Ternus (1926); Pantle & Picciano (1976) :: Three elements shown in two frames separated by a brief ISI produce an ambiguous motion percept: short ISIs (<~30-50ms) give 'element motion' (one element jumps to the other end), longer ISIs give 'group motion' (all elements shift together). ISI here is derived from the BPM, letting the threshold be crossed live.",
        "tags": ["ternus", "elementmotion", "groupmotion"],
    },
    "Brucke-Bartley Effect": {
        "it": "Brucke (1864); Bartley (1938) :: Una luce che lampeggia tra 1 e 17 Hz, con picco a 8-10 Hz (banda alfa), viene percepita piu' luminosa di una luce continua di identica luminanza media. La frequenza di flicker qui e' derivata direttamente dal BPM.",
        "en": "Brucke (1864); Bartley (1938) :: Light flickering between 1 and 17 Hz, peaking around 8-10 Hz (alpha band), is perceived as brighter than steady light of identical mean luminance. Flicker frequency here is derived directly from the BPM.",
        "tags": ["bruckebartley", "brightnessenhancement"],
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

def motion_silencing_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    MOTION SILENCING ILLUSION (Suchow & Alvarez, 2011, Current Biology).
    Anello di elementi che oscillano in luminanza secondo una sinusoide con
    fase diversa per ciascun elemento. Facendo ruotare l'anello, il
    cervello smette di percepire il flicker luminoso pur essendo ancora
    presente nei dati ('silencing of awareness to visual change'):
    l'effetto dipende dalla velocita' di rotazione, qui pilotata dai bassi,
    e la frequenza del flicker e' pilotata dagli alti. Un piccolo marker
    di fissazione centrale aiuta l'occhio a restare fermo, come nel
    paradigma sperimentale originale.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    cx, cy = width / 2.0, height / 2.0
    base_radius = min(width, height) * 0.32
    ring_radius = base_radius * element_size_factor
    n_dots = max(8, int(28 * num_elements_factor))
    base_dot_r = max(2, int(9 * pixel_scale))
    dot_radius = max(2, int((base_dot_r + bass_val * 4 * intensity) * pixel_scale))

    spin = frame * (0.03 + bass_val * 0.12) * rotation_speed_factor
    flicker_speed = 0.35 + high_val * 0.9

    img = np.zeros((height, width), dtype=float)
    for i in range(n_dots):
        angle = spin + i * (2 * np.pi / n_dots)
        dx_c = cx + ring_radius * np.cos(angle)
        dy_c = cy + ring_radius * np.sin(angle)
        phase = i * (2 * np.pi / n_dots) * 3.0  # un terzo dei dot in controfase
        val = np.clip(0.15 + 0.75 * (0.5 + 0.5 * np.sin(frame * flicker_speed + phase)) * intensity, 0.0, 1.0)
        rr, cc = disk((dy_c, dx_c), dot_radius, shape=(height, width))
        img[rr, cc] = val
    fix_r = max(1, int(3 * pixel_scale))
    rr, cc = disk((cy, cx), fix_r, shape=(height, width))
    img[rr, cc] = 0.9
    return img

def enigma_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    ENIGMA ILLUSION (Leviant, 1996). Anelli concentrici attraversati da una
    fine texture radiale a denti di sega; nel fenomeno reale il moto
    "serpeggiante" illusorio lungo gli anelli e' legato ai micro-movimenti
    oculari involontari (microsaccadi) durante la fissazione centrale. Qui
    le microsaccadi sono simulate con un piccolo jitter angolare oscillante
    nel tempo, la cui ampiezza e' pilotata dagli alti. Completamente
    vettorizzato in coordinate polari.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    cx, cy = width / 2.0, height / 2.0
    yv, xv = np.mgrid[0:height, 0:width]
    dx = xv - cx
    dy = yv - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    base_ring_width = 40.0
    ring_width = max(max(3, int(6 * pixel_scale)), int((base_ring_width * element_size_factor + bass_val * 12 * intensity) * pixel_scale))
    ring_idx = (r // ring_width).astype(int)
    direction = np.where(ring_idx % 2 == 0, 1.0, -1.0)

    microsaccade = 0.06 * (0.3 + high_val) * rotation_speed_factor * np.sin(frame * 2.7)

    n_teeth = max(6, int(40 * num_elements_factor))
    tooth_angle = 2 * np.pi / n_teeth
    local_theta = (theta * direction + microsaccade * direction + ring_idx * 0.35) % tooth_angle
    frac = local_theta / tooth_angle

    img = np.where(frac < 0.5, frac * 2, (1 - frac) * 1.2 + 0.15)
    return img

def barberpole_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    BARBERPOLE ILLUSION / APERTURE PROBLEM (Wallach, 1935; Wuerger, Shapley
    & Rubin, 1996). Strisce diagonali che scorrono sempre nella stessa
    direzione fisica (ortogonale al proprio orientamento), visibili solo
    attraverso un'apertura rettangolare allungata: il sistema visivo
    risolve l'ambiguita' del moto locale (aperture problem) percependo lo
    scorrimento lungo l'asse lungo dell'apertura invece che nella direzione
    fisica reale delle strisce. L'apertura ruota lentamente nel tempo
    (pilotata dai medi) per mostrare come la direzione percepita "segua"
    sempre l'orientamento dell'apertura.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    cx, cy = width / 2.0, height / 2.0
    yv, xv = np.mgrid[0:height, 0:width]
    dx = xv - cx
    dy = yv - cy

    aperture_angle = np.radians(30 + mid_val * 40 + frame * 0.15 * rotation_speed_factor)
    rot_x = dx * np.cos(-aperture_angle) - dy * np.sin(-aperture_angle)
    rot_y = dx * np.sin(-aperture_angle) + dy * np.cos(-aperture_angle)
    aperture_half_w = min(width, height) * 0.14 * element_size_factor
    aperture_half_h = min(width, height) * 0.34 * element_size_factor
    inside_aperture = (np.abs(rot_x) <= aperture_half_w) & (np.abs(rot_y) <= aperture_half_h)

    stripe_angle = np.radians(60)
    period = max(8, int((26 / num_elements_factor) * pixel_scale))
    shift = frame * (2 + bass_val * 6 * intensity) * pixel_scale
    proj = dx * np.cos(stripe_angle) + dy * np.sin(stripe_angle)
    stripe_val = (((proj + shift) // period) % 2).astype(float)

    bg_val = 0.12
    img = np.where(inside_aperture, stripe_val, bg_val)
    return img

def whites_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    WHITE'S ILLUSION (White, 1979). Griglia a bande verticali bianche/nere;
    toppe grigie identiche vengono posizionate su una riga sopra le bande
    bianche e su un'altra riga sopra le bande nere. Fisicamente sono lo
    stesso grigio, ma appaiono di luminosita' diversa -- in direzione
    OPPOSTA a quella prevista dal semplice contrasto simultaneo (le toppe
    sulle bande nere non sembrano piu' chiare come da contrasto, ma quasi
    assimilate). La griglia scorre pilotata dai bassi, le toppe pulsano
    con i medi. Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    yv, xv = np.mgrid[0:height, 0:width]

    base_stripe_w = 34.0
    stripe_w = max(max(3, int(6 * pixel_scale)), int((base_stripe_w / num_elements_factor * element_size_factor) * pixel_scale))
    shift = frame * (1.2 + bass_val * 3.0) * pixel_scale
    col_idx = ((xv + shift) // stripe_w).astype(int)
    grating = col_idx % 2  # 0 = banda scura, 1 = banda chiara
    base = np.where(grating == 1, 0.85, 0.15)

    patch_h = height * (0.14 + mid_val * 0.05)
    upper_band = (yv >= height * 0.28) & (yv <= height * 0.28 + patch_h)
    lower_band = (yv >= height * 0.62) & (yv <= height * 0.62 + patch_h)

    patch_on_light = upper_band & (grating == 1) & (col_idx % 4 == 1)
    patch_on_dark = lower_band & (grating == 0) & (col_idx % 4 == 2)

    patch_gray = 0.5 * intensity + 0.25 * (1 - intensity)
    img = np.where(patch_on_light | patch_on_dark, patch_gray, base)
    return img

def poggendorff_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    POGGENDORFF ILLUSION (Poggendorff, 1860). Piu' segmenti diagonali,
    ognuno perfettamente collineare, vengono interrotti da due bande
    verticali occludenti: pur essendo geometricamente allineati, i tratti
    visibili ai lati delle bande appaiono disallineati. Le linee sono
    disposte a piu' corsie orizzontali (per una texture generativa piena
    di schermo), l'angolo e la posizione delle bande sono pilotati
    dall'audio. Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    cx = width / 2.0
    yv, xv = np.mgrid[0:height, 0:width]

    base_lane_h = 46.0
    lane_h = max(max(6, int(10 * pixel_scale)), int((base_lane_h / num_elements_factor * element_size_factor) * pixel_scale))
    lane_idx = (yv // lane_h).astype(int)
    lane_center = (lane_idx + 0.5) * lane_h

    angle = np.radians(35 + mid_val * 20)
    slope = np.tan(angle) * (0.5 + high_val * 0.5)
    line_y = lane_center + slope * (xv - cx)

    thickness = max(1, int((2 + bass_val * 2 * intensity) * pixel_scale))
    diag_line = np.abs(yv - line_y) < thickness

    band_w = max(4, int((width * (0.05 + bass_val * 0.03))))
    band_center_1 = width * (0.32 + 0.03 * np.sin(frame * 0.05 * rotation_speed_factor))
    band_center_2 = width * (0.68 + 0.03 * np.sin(frame * 0.05 * rotation_speed_factor + np.pi))
    in_band = (np.abs(xv - band_center_1) < band_w / 2) | (np.abs(xv - band_center_2) < band_w / 2)

    img = np.where(in_band, 0.55, np.where(diag_line, 1.0, 0.05))
    return img

def rubber_pencil_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    RUBBER PENCIL ILLUSION (Pomerantz, 1983; recensione neurale in Macknik &
    Martinez-Conde, 2008, Nature Reviews Neuroscience). Un'asta rigida
    oscillata rapidamente da un'estremita' appare flettersi come gomma: i
    neuroni end-stopped di V1/MT rispondono diversamente tra le estremita'
    e il centro dello stimolo in moto, causando una dislocazione spaziale
    apparente. Qui il fenomeno e' rappresentato direttamente come un'onda
    la cui ampiezza e fase crescono dall'estremita' "tenuta" (ferma) verso
    quella "libera" (che sferza), pilotata dal ritmo audio. Piu' corsie
    alternano il lato di presa per una texture generativa a schermo
    intero. Completamente vettorizzato.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    yv, xv = np.mgrid[0:height, 0:width]

    base_lane_h = 60.0
    lane_h = max(max(8, int(14 * pixel_scale)), int((base_lane_h / num_elements_factor * element_size_factor) * pixel_scale))
    lane_idx = (yv // lane_h).astype(int)
    lane_center = (lane_idx + 0.5) * lane_h
    held_left = (lane_idx % 2 == 0)

    xn = np.where(held_left, xv / float(width), 1.0 - xv / float(width))

    amplitude = lane_h * 0.32 * element_size_factor * intensity
    phase_lag = 5.0 + high_val * 7.0
    omega = (0.9 + bass_val * 2.2) * rotation_speed_factor

    offset = amplitude * xn * np.sin(omega * frame * 0.15 - phase_lag * xn)
    target_y = lane_center + offset

    thickness = max(2, int((5 + bass_val * 3) * pixel_scale))
    bar_mask = np.abs(yv - target_y) < thickness

    img = np.where(bar_mask, 1.0, 0.06)
    return img

def wagon_wheel_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    CONTINUOUS WAGON WHEEL ILLUSION (Purves, Paydarfar & Andrews, 1996,
    PNAS). Un disco a raggi equidistanti ruota: superata una soglia
    critica di velocita' angolare, il sistema visivo percepisce inversioni
    spontanee del senso di rotazione, anche in luce continua. Qui la
    velocita' e' pilotata dal BPM e dai bassi: l'aliasing percettivo (lo
    stesso principio del campionamento discreto dei fotogrammi cinema)
    emerge dal rendering stesso quando la velocita' dei raggi supera la
    soglia di Nyquist angolare rispetto al frame rate -- non e' un trucco
    aggiunto, e' il fenomeno descritto nel paper.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]

    cx, cy = width / 2.0, height / 2.0
    yv, xv = np.mgrid[0:height, 0:width]
    dx = xv - cx
    dy = yv - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    n_spokes = max(4, int(10 * num_elements_factor))
    radius_max = min(width, height) * 0.42 * element_size_factor

    angular_step = (0.15 + bass_val * 0.9) * rotation_speed_factor * intensity
    spin = frame * angular_step

    spoke_angle = 2 * np.pi / n_spokes
    local_theta = (theta - spin) % spoke_angle
    frac = local_theta / spoke_angle
    spoke_width = 0.08 + 0.04 * bass_val
    is_spoke = (frac < spoke_width) & (r < radius_max)
    rim = np.abs(r - radius_max) < max(2, int(3 * pixel_scale))
    hub = r < max(3, int(8 * pixel_scale))

    img = np.where(is_spoke | rim | hub, 1.0, 0.05)
    return img

def motion_aftereffect_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    MOTION AFTEREFFECT / WATERFALL ILLUSION (Addams, 1834; rassegna moderna
    in Mather, Verstraten & Anstis, "The Motion Aftereffect", 1998). Dopo
    un adattamento prolungato a un moto direzionale, un pattern statico
    appare muoversi nella direzione opposta (desensibilizzazione
    asimmetrica dei rilevatori di direzione). Qui si riproduce il
    paradigma sperimentale: una texture a bande scorre per una fase di
    "adattamento" pilotata dal ritmo, poi si arresta di colpo per una
    fase di "test" -- l'aftereffect vero si manifesta nell'osservatore
    subito dopo l'arresto.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]

    period_frames = max(30, int(90 / max(0.1, rotation_speed_factor)))
    cycle = frame % period_frames
    adapt_len = int(period_frames * 0.75)
    is_adapt = cycle < adapt_len

    stripe_w = max(4, int(24 * element_size_factor * pixel_scale))
    speed = (3 + bass_val * 6) * intensity * pixel_scale
    shift = cycle * speed if is_adapt else adapt_len * speed

    yv, xv = np.mgrid[0:height, 0:width]
    img = (((yv + shift) // stripe_w) % 2).astype(float)
    return img

def flash_lag_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    FLASH-LAG EFFECT (Nijhawan, 1994, Nature). Un oggetto in moto continuo
    e un flash statico, presentati nello stesso punto nello stesso
    istante, vengono percepiti disallineati: l'oggetto in moto appare "in
    anticipo" rispetto al flash. Qui un punto orbita a velocita' pilotata
    dal BPM/bassi, e un anello lampeggia in sincronia col beat esattamente
    nella posizione istantanea del punto in moto -- stesso paradigma
    sperimentale del paper originale.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    high_val = audio_features["high"][frame % len(audio_features["high"])]

    cx, cy = width / 2.0, height / 2.0
    orbit_r = min(width, height) * 0.3 * element_size_factor
    ang_speed = (0.05 + bass_val * 0.15) * rotation_speed_factor * intensity
    angle = frame * ang_speed
    mx, my = cx + orbit_r * np.cos(angle), cy + orbit_r * np.sin(angle)

    yv, xv = np.mgrid[0:height, 0:width]
    dot_r = max(3, int(10 * pixel_scale))
    dist_sq = (xv - mx) ** 2 + (yv - my) ** 2
    moving_dot = dist_sq <= dot_r ** 2

    flash_period = max(6, int(24 / (0.3 + high_val)))
    is_flash = (frame % flash_period) < 3
    flash_ring = (dist_sq <= (dot_r + 6) ** 2) & (dist_sq > (dot_r + 2) ** 2) if is_flash else np.zeros_like(moving_dot)

    orbit_path = np.abs(np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2) - orbit_r) < 1
    img = np.where(orbit_path, 0.15, 0.0)
    img = np.where(flash_ring, 0.6, img)
    img = np.where(moving_dot, 1.0, img)
    return img

def line_motion_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    LINE MOTION ILLUSION (Hikosaka, Miyauchi & Shimojo, 1993, Vision
    Research). Un breve cue attenzionale a un'estremita', seguito
    dall'apparizione istantanea di una linea statica per intero, genera
    la percezione che la linea si stia "disegnando" in movimento dal capo
    cued verso l'altro. Il ciclo cue+linea si ripete a un ritmo pilotato
    dal beat, alternando l'estremita' del cue a ogni ciclo.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    mid_val = audio_features["mid"][frame % len(audio_features["mid"])]

    cycle_len = max(10, int(40 / (0.5 + bass_val) * rotation_speed_factor))
    cue_len = max(2, int(cycle_len * 0.12))
    cycle_pos = frame % cycle_len
    cycle_num = frame // cycle_len
    cued_left = (cycle_num % 2 == 0)

    n_lines = max(3, int(6 * num_elements_factor))
    lane_h = height / n_lines
    yv, xv = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width), dtype=float)
    line_len_frac = 0.6 + mid_val * 0.2
    thickness = max(2, int(4 * pixel_scale))
    x0 = int(width * (1 - line_len_frac) / 2)
    x1 = int(width * (1 + line_len_frac) / 2)

    for i in range(n_lines):
        y_c = int((i + 0.5) * lane_h)
        lane_mask = np.abs(yv - y_c) < thickness
        if cycle_pos < cue_len:
            cue_x = x0 if cued_left else x1
            cue_mask = lane_mask & (np.abs(xv - cue_x) < max(3, int(6 * pixel_scale)))
            img = np.where(cue_mask, 1.0, img)
        else:
            full_line = lane_mask & (xv >= x0) & (xv <= x1)
            img = np.where(full_line, 1.0, img)
    return img

def motion_induced_blindness_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    MOTION-INDUCED BLINDNESS (Bonneh, Cooperman & Sagi, 2001, Nature).
    Bersagli statici salienti, circondati da un pattern globale in
    movimento (una griglia rotante a crocette), scompaiono e ricompaiono
    periodicamente dalla consapevolezza percettiva durante la fissazione
    prolungata del centro. Qui si riproduce lo stimolo sperimentale
    fedele: griglia rotante pilotata dal ritmo, tre bersagli statici fissi
    e un marker di fissazione centrale -- la scomparsa percettiva accade
    nell'osservatore, non nel rendering.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]

    cx, cy = width / 2.0, height / 2.0
    yv, xv = np.mgrid[0:height, 0:width]
    dx = xv - cx
    dy = yv - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    n_cross = max(8, int(24 * num_elements_factor))
    spin = frame * (0.03 + bass_val * 0.08) * rotation_speed_factor
    seg_angle = 2 * np.pi / n_cross
    local_theta = (theta - spin) % seg_angle
    ring_mask = (r > min(width, height) * 0.12) & (r < min(width, height) * 0.46)
    cross_mask = ((local_theta < seg_angle * 0.12) | (local_theta > seg_angle * 0.88)) & ring_mask

    target_r = min(width, height) * 0.28 * element_size_factor
    target_positions = [(-1.0, 0.0), (0.5, 0.87), (0.5, -0.87)]
    dot_r = max(2, int(6 * pixel_scale))
    target_mask = np.zeros((height, width), dtype=bool)
    for ox, oy in target_positions:
        px, py = cx + ox * target_r, cy + oy * target_r
        target_mask |= ((xv - px) ** 2 + (yv - py) ** 2) <= dot_r ** 2

    fix_r = max(1, int(3 * pixel_scale))
    fix_mask = ((xv - cx) ** 2 + (yv - cy) ** 2) <= fix_r ** 2

    img = np.where(cross_mask, 0.5, 0.02)
    img = np.where(target_mask, 1.0, img)
    img = np.where(fix_mask, 0.8, img)
    return img

def troxler_fading_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    TROXLER FADING (Troxler, 1804). Bersagli periferici statici su un
    campo omogeneo scompaiono dalla consapevolezza durante la fissazione
    prolungata del centro (adattamento neurale periferico). L'illusione
    richiede un campo il piu' possibile stabile: qui il pattern resta
    quasi statico, con solo una lentissima modulazione di luminosita'
    pilotata dai bassi (variazioni troppo rapide o ampie impedirebbero
    l'adattamento e quindi la sparizione percepita).
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]

    cx, cy = width / 2.0, height / 2.0
    slow_phase = frame * 0.01 * (0.5 + bass_val * 0.3) * intensity
    bg_val = 0.4 + 0.05 * np.sin(slow_phase)

    n_targets = max(4, int(6 * num_elements_factor))
    target_r = min(width, height) * 0.38 * element_size_factor
    dot_r = max(3, int(9 * pixel_scale))
    img = np.full((height, width), bg_val, dtype=float)
    for i in range(n_targets):
        ang = 2 * np.pi * i / n_targets
        px, py = cx + target_r * np.cos(ang), cy + target_r * np.sin(ang)
        rr, cc = disk((py, px), dot_r, shape=(height, width))
        img[rr, cc] = 0.75

    fix_r = max(1, int(3 * pixel_scale))
    rr, cc = disk((cy, cx), fix_r, shape=(height, width))
    img[rr, cc] = 0.95
    return img

def mccollough_effect_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    McCOLLOUGH EFFECT (McCollough, 1965, Science). Un adattamento
    prolungato a griglie bianco/nero colorate in modo diverso per
    orientamento (arancio sulle orizzontali, ciano sulle verticali)
    produce un dopoimmagine contingente all'orientamento. Qui si
    riproduce lo stimolo sperimentale originale, che richiede colori
    specifici per funzionare e quindi IGNORA la palette utente: fasi
    alternate di griglia arancio-orizzontale e ciano-verticale, il cui
    ritmo di alternanza e' pilotato dal beat. RITORNA DIRETTAMENTE
    UN'IMMAGINE RGB (bypassa apply_colors).
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    period = max(20, int(70 / (0.4 + bass_val)))
    phase_horizontal = (frame % (period * 2)) < period

    base_stripe = max(4, int(18 * element_size_factor * pixel_scale))
    yv, xv = np.mgrid[0:height, 0:width]

    rgb = np.zeros((height, width, 3), dtype=float)
    if phase_horizontal:
        stripe = ((yv // base_stripe) % 2).astype(float)
        rgb[:, :, 0] = stripe * 0.95
        rgb[:, :, 1] = stripe * 0.55
        rgb[:, :, 2] = stripe * 0.05
    else:
        stripe = ((xv // base_stripe) % 2).astype(float)
        rgb[:, :, 0] = stripe * 0.0
        rgb[:, :, 1] = stripe * 0.75
        rgb[:, :, 2] = stripe * 0.95
    return rgb

def phi_beta_movement_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    PHI PHENOMENON / BETA MOVEMENT (Wertheimer, 1912). Due stimoli vicini,
    mostrati in alternanza, generano una sensazione di moto la cui natura
    dipende criticamente dall'intervallo interstimolo (ISI): a ISI brevi
    si percepisce un oggetto che si sposta (beta movement, la base
    percettiva del cinema), a ISI cortissimi la sensazione tende al
    "puro phi" (senso di movimento senza un oggetto definito). Qui l'ISI
    e' derivato direttamente dal BPM del brano (una frazione del battito),
    pilotando l'alternanza di una griglia di coppie di dischi.
    NOTA: la conversione BPM->frame assume un frame rate di output
    tipico (~24-30 fps); non e' uno strumento psicofisico di laboratorio.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    tempo = audio_features.get("tempo", 120.0)

    beats_per_frame_unit = max(1, int((60.0 / max(1.0, tempo)) * 6.0 / max(0.1, rotation_speed_factor) / max(0.1, intensity)))
    flip = (frame // beats_per_frame_unit) % 2 == 0

    n_pairs_x = max(2, int(4 * num_elements_factor))
    n_pairs_y = max(1, int(2 * num_elements_factor))
    cell_w = width / n_pairs_x
    cell_h = height / n_pairs_y
    dot_r = max(3, int(cell_w * 0.12 * element_size_factor * pixel_scale))
    offset = cell_w * 0.22 * (0.6 + bass_val * 0.6)

    img = np.zeros((height, width), dtype=float)
    for iy in range(n_pairs_y):
        for ix in range(n_pairs_x):
            cx = (ix + 0.5) * cell_w
            cy = (iy + 0.5) * cell_h
            px = cx - offset if flip else cx + offset
            rr, cc = disk((cy, px), dot_r, shape=(height, width))
            img[rr, cc] = 1.0
    return img

def ternus_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    TERNUS ILLUSION (Ternus, 1926; Pantle & Picciano, 1976). Tre elementi
    identici, mostrati in due fotogrammi separati da un breve intervallo
    vuoto (ISI), generano una percezione di moto ambigua: a ISI corti
    (<~30-50ms) si vede solo l'elemento estremo "saltare" all'altro capo
    (element motion), a ISI piu' lunghi tutti gli elementi sembrano
    scorrere insieme (group motion). Qui la durata dell'ISI (in numero di
    fotogrammi) e' derivata dal BPM e dai bassi, permettendo di
    attraversare dal vivo la soglia percettiva element/group.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]

    isi_frames = max(1, int(2 + bass_val * 10 * rotation_speed_factor * intensity))
    frame_a_len = max(3, int(10 / max(0.1, rotation_speed_factor)))
    total_cycle = frame_a_len * 2 + isi_frames
    pos = frame % total_cycle

    n_slots = max(4, int(5 * num_elements_factor))
    slot_w = width / (n_slots + 1)
    dot_r = max(4, int(slot_w * 0.28 * element_size_factor * pixel_scale))
    cy = height / 2.0

    if pos < frame_a_len:
        active_slots = [1, 2, 3]
    elif pos < frame_a_len + isi_frames:
        active_slots = []
    else:
        active_slots = [2, 3, 4]

    img = np.zeros((height, width), dtype=float)
    for s in active_slots:
        px = s * slot_w
        rr, cc = disk((cy, px), dot_r, shape=(height, width))
        img[rr, cc] = 1.0
    return img

def brucke_bartley_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale=1.0):
    """
    BRUCKE-BARTLEY EFFECT (Brucke, 1864; Bartley, 1938). Una luce che
    lampeggia a frequenze comprese tra 1 e 17 Hz, con un picco intorno
    agli 8-10 Hz (banda alfa), viene percepita PIU' LUMINOSA di una luce
    continua di identica luminanza media -- un potenziamento paradossale
    della luminosita' percepita legato alle oscillazioni corticali alfa.
    La frequenza di flicker qui e' derivata direttamente dal BPM (una
    frazione del battito) e dai bassi: variando la velocita' l'utente
    attraversa dal vivo la banda 8-10 Hz dove l'effetto e' massimo. Un
    anello di riferimento a luminanza costante circonda il disco
    lampeggiante per il confronto diretto.
    NOTA: la conversione Hz->fase assume un frame rate di output tipico
    (~30 fps); non e' uno strumento psicofisico di laboratorio.
    """
    bass_val = audio_features["bass"][frame % len(audio_features["bass"])]
    tempo = audio_features.get("tempo", 120.0)

    flicker_hz = (tempo / 60.0) * (0.5 + bass_val * 1.5) * rotation_speed_factor * intensity
    flicker_hz = max(0.5, min(20.0, flicker_hz))
    phase_per_frame = 2 * np.pi * flicker_hz / 30.0
    flicker_val = 0.5 + 0.5 * np.sin(frame * phase_per_frame)

    cx, cy = width / 2.0, height / 2.0
    yv, xv = np.mgrid[0:height, 0:width]
    r = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)
    inner_r = min(width, height) * 0.32 * element_size_factor
    outer_r = min(width, height) * 0.46 * element_size_factor

    inner_disc = r < inner_r
    ref_ring = (r >= inner_r) & (r < outer_r)

    img = np.where(inner_disc, flicker_val, np.where(ref_ring, 0.5, 0.02))
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
    elif illusion_type == "Motion Silencing":
        img = motion_silencing_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Enigma Illusion":
        img = enigma_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Barberpole Illusion":
        img = barberpole_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "White's Illusion":
        img = whites_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Poggendorff Illusion":
        img = poggendorff_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Rubber Pencil Illusion":
        img = rubber_pencil_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Wagon Wheel Illusion":
        img = wagon_wheel_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Motion Aftereffect":
        img = motion_aftereffect_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Flash-Lag Effect":
        img = flash_lag_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Line Motion Illusion":
        img = line_motion_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Motion-Induced Blindness":
        img = motion_induced_blindness_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Troxler Fading":
        img = troxler_fading_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "McCollough Effect":
        # Richiede colori specifici (arancio/ciano) per funzionare: ritorna
        # gia' un'immagine RGB e bypassa la palette utente / apply_colors.
        return mccollough_effect_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Phi/Beta Movement":
        img = phi_beta_movement_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Ternus Illusion":
        img = ternus_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
    elif illusion_type == "Brucke-Bartley Effect":
        img = brucke_bartley_illusion(width, height, frame, audio_features, intensity, element_size_factor, num_elements_factor, rotation_speed_factor, pixel_scale)
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
        "Motion Silencing", "Enigma Illusion", "Barberpole Illusion",
        "White's Illusion", "Poggendorff Illusion", "Rubber Pencil Illusion",
        "Wagon Wheel Illusion", "Motion Aftereffect", "Flash-Lag Effect",
        "Line Motion Illusion", "Motion-Induced Blindness", "Troxler Fading",
        "McCollough Effect",
        "Phi/Beta Movement", "Ternus Illusion", "Brucke-Bartley Effect",
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
        "Motion Silencing", "Enigma Illusion", "Barberpole Illusion",
        "White's Illusion", "Poggendorff Illusion", "Rubber Pencil Illusion",
        "Wagon Wheel Illusion", "Motion Aftereffect", "Flash-Lag Effect",
        "Line Motion Illusion", "Motion-Induced Blindness", "Troxler Fading",
        "McCollough Effect",
        "Phi/Beta Movement", "Ternus Illusion", "Brucke-Bartley Effect",
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

    # ---------------------------------
    # ANTEPRIMA VIDEO A 380p
    # Downscale veloce (ffmpeg scale + preset veryfast) del file gia'
    # generato: non serve ri-renderizzare i frame, solo transcodificare.
    # Il lato corto del video viene portato a 380px mantenendo l'aspect ratio.
    # ---------------------------------
    with st.spinner("🔎 Preparazione anteprima 380p..."):
        preview_short_side = 380
        if size[0] <= size[1]:
            preview_w = preview_short_side
            preview_h = int(round(size[1] * preview_short_side / size[0] / 2) * 2)
        else:
            preview_h = preview_short_side
            preview_w = int(round(size[0] * preview_short_side / size[1] / 2) * 2)

        preview_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        (
            ffmpeg
            .input(output_file.name)
            .output(
                preview_output_file.name,
                vf=f"scale={preview_w}:{preview_h}",
                vcodec="libx264", acodec="aac", preset="veryfast", crf=30,
            )
            .run(overwrite_output=True, quiet=True)
        )
        with open(preview_output_file.name, "rb") as pf:
            preview_video_bytes = pf.read()

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
    st.session_state["loop507_preview_video_bytes"] = preview_video_bytes
    st.session_state["loop507_report_text"] = report_text
    st.session_state["loop507_report_filename"] = f"loop507_report_{safe_name}.txt"

    try:
        os.remove(tmp_audio.name)
        os.remove(tmp_video.name)
        os.remove(output_file.name)
        os.remove(preview_output_file.name)
    except Exception:
        pass

if "loop507_video_bytes" in st.session_state:
    st.success("✨ Video generato con successo! Implementazioni neuropsicologiche accurate.")
    _preview_col, _ = st.columns([1, 1])
    with _preview_col:
        st.video(st.session_state["loop507_preview_video_bytes"])
    st.caption("Anteprima a 380p — il download qui sotto e' alla risoluzione piena scelta in fase di generazione.")
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
