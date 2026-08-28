#!/usr/bin/env python3
"""Analisi biomeccanica di video di pedalata (bike fitting) con MediaPipe Pose.

Uso:
    python bike_analysis.py --input video.mp4 --side right
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MIN_HALF_CYCLE_FRAMES_RELIABLE = 8  # sotto questa soglia il campionamento dei punti morti è inaffidabile
KNEE_FLEXION_IDEAL_MIN = 25.0
KNEE_FLEXION_IDEAL_MAX = 35.0
SMOOTHING_WINDOW_S = 0.3  # finestra del filtro Savitzky-Golay, per attenuare il jitter di tracking frame-per-frame
SMOOTHING_POLYORDER = 2

LANDMARK_NAMES = {
    "left": {
        "shoulder": "LEFT_SHOULDER",
        "elbow": "LEFT_ELBOW",
        "wrist": "LEFT_WRIST",
        "hip": "LEFT_HIP",
        "knee": "LEFT_KNEE",
        "ankle": "LEFT_ANKLE",
        "foot_index": "LEFT_FOOT_INDEX",
    },
    "right": {
        "shoulder": "RIGHT_SHOULDER",
        "elbow": "RIGHT_ELBOW",
        "wrist": "RIGHT_WRIST",
        "hip": "RIGHT_HIP",
        "knee": "RIGHT_KNEE",
        "ankle": "RIGHT_ANKLE",
        "foot_index": "RIGHT_FOOT_INDEX",
    },
}

SKELETON_CONNECTIONS = [
    ("shoulder", "elbow"),
    ("elbow", "wrist"),
    ("shoulder", "hip"),
    ("hip", "knee"),
    ("knee", "ankle"),
    ("ankle", "foot_index"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analisi biomeccanica di pedalata da video laterale.")
    parser.add_argument("--input", required=True, help="Percorso del video di input (mp4)")
    parser.add_argument("--output-dir", default="output", help="Cartella in cui salvare video, CSV e grafico (default: %(default)s)")
    parser.add_argument("--output", default=None, help="Percorso del video annotato in output (default: <output-dir>/<input>_annotated.mp4)")
    parser.add_argument("--csv", default=None, help="Percorso del CSV di output (default: <output-dir>/<input>_angles.csv)")
    parser.add_argument("--plot", default=None, help="Percorso del grafico PNG (default: <output-dir>/<input>_knee_rom.png)")
    parser.add_argument("--side", choices=["left", "right"], default="right", help="Lato del ciclista ripreso")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5, help="Soglia minima di confidenza/visibilità dei keypoint")
    parser.add_argument("--smoothing-window", type=float, default=SMOOTHING_WINDOW_S,
                         help="Finestra temporale (s) del filtro Savitzky-Golay per attenuare il jitter di tracking (default: %(default)s)")
    parser.add_argument("--no-smoothing", action="store_true", help="Disattiva lo smoothing e usa gli angoli grezzi (solo interpolati)")
    return parser.parse_args()


def extract_pose_landmarks(frame, pose_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb)
    if result.pose_landmarks is None:
        return None
    return result.pose_landmarks.landmark


def get_side_keypoints(landmarks, side, frame_w, frame_h, min_confidence):
    if landmarks is None:
        return {name: None for name in LANDMARK_NAMES[side]}

    pose_landmark = mp.solutions.pose.PoseLandmark
    kpts = {}
    for key, lm_name in LANDMARK_NAMES[side].items():
        lm = landmarks[pose_landmark[lm_name].value]
        if lm.visibility is not None and lm.visibility < min_confidence:
            kpts[key] = None
        else:
            kpts[key] = (lm.x * frame_w, lm.y * frame_h)
    return kpts


def angle_3pts(a, b, c):
    if a is None or b is None or c is None:
        return np.nan
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def angle_vs_horizontal(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    p1, p2 = np.array(p1), np.array(p2)
    v = p2 - p1
    if np.linalg.norm(v) == 0:
        return np.nan
    angle = np.degrees(np.arctan2(abs(v[1]), abs(v[0])))
    return float(angle)


def compute_frame_angles(kpts):
    knee = angle_3pts(kpts.get("hip"), kpts.get("knee"), kpts.get("ankle"))
    trunk = angle_vs_horizontal(kpts.get("shoulder"), kpts.get("hip"))
    ankle = angle_3pts(kpts.get("knee"), kpts.get("ankle"), kpts.get("foot_index"))
    return {"angolo_ginocchio": knee, "angolo_busto": trunk, "angolo_caviglia": ankle}


def draw_skeleton(frame, kpts):
    for a_key, b_key in SKELETON_CONNECTIONS:
        a, b = kpts.get(a_key), kpts.get(b_key)
        if a is not None and b is not None:
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 2)
    for point in kpts.values():
        if point is not None:
            cv2.circle(frame, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)


def draw_angle_text(frame, kpts, angles):
    def put(label, value, anchor_key):
        anchor = kpts.get(anchor_key)
        if anchor is None or np.isnan(value):
            return
        text = f"{label}: {value:.1f}deg"
        pos = (int(anchor[0]) + 10, int(anchor[1]))
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    put("Ginocchio", angles["angolo_ginocchio"], "knee")
    put("Busto", angles["angolo_busto"], "hip")
    put("Caviglia", angles["angolo_caviglia"], "ankle")


def draw_hud(frame, frame_idx, angles, running_min, running_max):
    lines = [
        f"Frame: {frame_idx}",
        f"Ginocchio min/max: {running_min:.1f}deg / {running_max:.1f}deg" if not np.isnan(running_min) else "Ginocchio min/max: N/D",
    ]
    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 25


def process_video(input_path, output_path, side, min_detection_confidence):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {input_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # applica la rotazione dei video verticali (es. smartphone)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # CAP_PROP_FRAME_WIDTH/HEIGHT riportano le dimensioni del flusso codificato
    # ignorando l'eventuale rotazione (es. video verticali da smartphone), che invece
    # viene già applicata ai frame restituiti da cap.read(). Si usa quindi la shape
    # reale del primo frame per evitare un video di output con dimensioni errate.
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError(f"Il video non contiene frame leggibili: {input_path}")
    height, width = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    rows = []
    running_min, running_max = np.nan, np.nan

    with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                       min_tracking_confidence=min_detection_confidence) as pose_model:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = extract_pose_landmarks(frame, pose_model)
            kpts = get_side_keypoints(landmarks, side, width, height, min_detection_confidence)
            angles = compute_frame_angles(kpts)

            knee_val = angles["angolo_ginocchio"]
            if not np.isnan(knee_val):
                running_min = knee_val if np.isnan(running_min) else min(running_min, knee_val)
                running_max = knee_val if np.isnan(running_max) else max(running_max, knee_val)

            draw_skeleton(frame, kpts)
            draw_angle_text(frame, kpts, angles)
            draw_hud(frame, frame_idx, angles, running_min, running_max)
            writer.write(frame)

            rows.append({
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "angolo_ginocchio": knee_val,
                "angolo_busto": angles["angolo_busto"],
                "angolo_caviglia": angles["angolo_caviglia"],
            })
            frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    return df, fps


def interpolate_series(df, cols):
    df = df.copy()
    for col in cols:
        df[f"{col}_was_missing"] = df[col].isna()
        df[col] = df[col].interpolate(limit_direction="both")
    return df


def smooth_series(df, cols, fps, window_s=SMOOTHING_WINDOW_S, polyorder=SMOOTHING_POLYORDER):
    """Attenua il jitter frame-per-frame del tracking (Savitzky-Golay) senza appiattire i picchi reali."""
    df = df.copy()
    window = int(round(window_s * fps))
    window += 1 - (window % 2)  # forza lunghezza dispari, richiesta da savgol_filter
    window = max(window, polyorder + 1 + (1 - (polyorder + 1) % 2))  # >= polyorder+1 e dispari

    for col in cols:
        values = df[col].to_numpy()
        if len(values) <= window:
            continue  # serie troppo corta per il filtro, si mantengono i valori interpolati
        df[col] = savgol_filter(values, window_length=window, polyorder=polyorder)
    return df


def detect_pedal_cycles(knee_angle_series, fps, min_prominence=5.0, min_distance_s=0.25):
    values = knee_angle_series.to_numpy()
    valid = ~np.isnan(values)
    if valid.sum() < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    min_distance = max(1, int(round(min_distance_s * fps)))
    peaks_idx, _ = find_peaks(values, prominence=min_prominence, distance=min_distance)
    valleys_idx, _ = find_peaks(-values, prominence=min_prominence, distance=min_distance)
    return peaks_idx, valleys_idx


def estimate_cycle_reliability(peaks_idx, valleys_idx, fps):
    extrema = np.sort(np.concatenate([peaks_idx, valleys_idx]))
    if len(extrema) < 2:
        return {
            "n_cycles_detected": 0,
            "mean_half_cycle_frames": np.nan,
            "reliable": False,
            "message": "Numero insufficiente di picchi/valli rilevati per stimare i cicli di pedalata.",
        }

    half_cycle_frames = np.diff(extrema)
    mean_half_cycle = float(np.mean(half_cycle_frames))
    reliable = mean_half_cycle >= MIN_HALF_CYCLE_FRAMES_RELIABLE

    n_cycles = len(peaks_idx)
    if reliable:
        message = (f"Frame rate ({fps:.1f} fps) sufficiente: in media {mean_half_cycle:.1f} frame "
                    f"per mezzo ciclo (soglia minima consigliata: {MIN_HALF_CYCLE_FRAMES_RELIABLE}).")
    else:
        message = (f"ATTENZIONE: frame rate ({fps:.1f} fps) probabilmente insufficiente per "
                    f"catturare con affidabilità i punti morti (PMS/PMI). Media di soli "
                    f"{mean_half_cycle:.1f} frame per mezzo ciclo (soglia minima consigliata: "
                    f"{MIN_HALF_CYCLE_FRAMES_RELIABLE}). Si consiglia di riprendere a fps più alto.")

    return {
        "n_cycles_detected": n_cycles,
        "mean_half_cycle_frames": mean_half_cycle,
        "reliable": reliable,
        "message": message,
    }


def summarize_knee_rom(knee_angle_series):
    values = knee_angle_series.to_numpy()
    if np.all(np.isnan(values)):
        return {"min": np.nan, "max": np.nan, "frame_min": None, "frame_max": None}

    frame_min = int(np.nanargmin(values))
    frame_max = int(np.nanargmax(values))
    return {
        "min": float(values[frame_min]),
        "max": float(values[frame_max]),
        "frame_min": frame_min,
        "frame_max": frame_max,
    }


def export_csv(df, path):
    export_cols = ["frame", "timestamp", "angolo_ginocchio", "angolo_busto", "angolo_caviglia"]
    df[export_cols].to_csv(path, index=False)


def export_plot(df, peaks_idx, valleys_idx, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["timestamp"], df["angolo_ginocchio"], label="Angolo ginocchio", color="tab:blue")

    if len(peaks_idx) > 0:
        ax.scatter(df["timestamp"].iloc[peaks_idx], df["angolo_ginocchio"].iloc[peaks_idx],
                   color="tab:red", marker="^", s=60, label="Massimi (estensione)", zorder=5)
    if len(valleys_idx) > 0:
        ax.scatter(df["timestamp"].iloc[valleys_idx], df["angolo_ginocchio"].iloc[valleys_idx],
                   color="tab:green", marker="v", s=60, label="Minimi (flessione)", zorder=5)

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Angolo ginocchio (°)")
    ax.set_title("Escursione angolare ginocchio durante la pedalata")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    input_path = args.input
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = args.output or os.path.join(output_dir, f"{input_stem}_annotated.mp4")
    csv_path = args.csv or os.path.join(output_dir, f"{input_stem}_angles.csv")
    plot_path = args.plot or os.path.join(output_dir, f"{input_stem}_knee_rom.png")

    print(f"[1/6] Elaborazione video: {input_path} (lato: {args.side})")
    df, fps = process_video(input_path, output_path, args.side, args.min_detection_confidence)
    print(f"      Video annotato salvato in: {output_path}")

    angle_cols = ["angolo_ginocchio", "angolo_busto", "angolo_caviglia"]

    print("[2/6] Interpolazione dati mancanti...")
    df_interp = interpolate_series(df, angle_cols)

    if args.no_smoothing:
        df_clean = df_interp
        print("[3/6] Smoothing disattivato (--no-smoothing).")
    else:
        print(f"[3/6] Attenuazione jitter di tracking (Savitzky-Golay, finestra {args.smoothing_window:.2f}s)...")
        df_clean = smooth_series(df_interp, angle_cols, fps, window_s=args.smoothing_window)

    print("[4/6] Rilevamento cicli di pedalata...")
    peaks_idx, valleys_idx = detect_pedal_cycles(df_clean["angolo_ginocchio"], fps)
    reliability = estimate_cycle_reliability(peaks_idx, valleys_idx, fps)
    print(f"      {reliability['message']}")

    rom = summarize_knee_rom(df_clean["angolo_ginocchio"])
    print(f"[5/6] Escursione ginocchio: min={rom['min']:.1f}° (frame {rom['frame_min']}), "
          f"max={rom['max']:.1f}° (frame {rom['frame_max']})")

    flexion_at_bottom = 180.0 - rom["max"] if not np.isnan(rom["max"]) else np.nan
    if not np.isnan(flexion_at_bottom):
        in_range = KNEE_FLEXION_IDEAL_MIN <= flexion_at_bottom <= KNEE_FLEXION_IDEAL_MAX
        status = "nel range ideale" if in_range else "FUORI dal range ideale"
        print(f"      Flessione stimata a fondo corsa: {flexion_at_bottom:.1f}° ({status}, "
              f"riferimento {KNEE_FLEXION_IDEAL_MIN}-{KNEE_FLEXION_IDEAL_MAX}°)")

    print("[6/6] Export CSV e grafico...")
    export_csv(df_clean, csv_path)
    export_plot(df_clean, peaks_idx, valleys_idx, plot_path)
    print(f"      CSV salvato in: {csv_path}")
    print(f"      Grafico salvato in: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        sys.exit(1)
