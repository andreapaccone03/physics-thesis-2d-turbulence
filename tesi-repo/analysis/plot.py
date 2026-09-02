import os
import shutil

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

DEFAULT_FFMPEG_PATH = (
    r"C:\Users\HP\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
)


def save_animation_mp4(ani, output_path, fps=20, dpi=150, bitrate=1800):
    ffmpeg_path = os.environ.get("FFMPEG_PATH")
    if not ffmpeg_path:
        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg_path and os.path.exists(DEFAULT_FFMPEG_PATH):
        ffmpeg_path = DEFAULT_FFMPEG_PATH

    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg non trovato. Installa ffmpeg e aggiungilo al PATH, "
            "oppure imposta FFMPEG_PATH con il percorso di ffmpeg.exe."
        )

    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
    ani.save(output_path, writer=writer, dpi=dpi)


def load_scalar_frames(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File non trovato: {filename}")

    times = []
    frames = []
    current_values = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# Time:"):
                if current_values:
                    frames.append(np.array(current_values, dtype=float))
                    current_values = []
                times.append(float(line.split(":")[1]))
                continue

            parts = line.split()
            if len(parts) >= 3:
                current_values.append(float(parts[2]))

    if current_values:
        frames.append(np.array(current_values, dtype=float))

    if not frames:
        raise ValueError(f"Nessun frame trovato in {filename}")

    return times, frames


def robust_limits(frames, lower=0.5, upper=99.5):
    values = np.concatenate(frames)
    vmin = np.percentile(values, lower)
    vmax = np.percentile(values, upper)

    if vmin == vmax:
        delta = 1.0 if vmax == 0 else 0.05 * abs(vmax)
        vmin -= delta
        vmax += delta

    return float(vmin), float(vmax)


def build_histograms(psi_frames, w_frames, bins=100):
    psi_min, psi_max = robust_limits(psi_frames)
    w_min, w_max = robust_limits(w_frames)

    histograms = []
    max_count = 1.0

    for psi_frame, w_frame in zip(psi_frames, w_frames):
        hist, _, _ = np.histogram2d(
            psi_frame,
            w_frame,
            bins=bins,
            range=[[psi_min, psi_max], [w_min, w_max]],
        )
        histograms.append(hist)
        max_count = max(max_count, float(hist.max()))

    return histograms, (psi_min, psi_max, w_min, w_max), max_count


def animate_phase_plot(
    psi_file="output/psi_evolution.dat",
    w_file="output/vorticity_evolution.dat",
    output_file="output/w_vs_psi_animation.mp4",
    bins=100,
    fps=20,
):
    times_psi, psi_frames = load_scalar_frames(psi_file)
    times_w, w_frames = load_scalar_frames(w_file)

    n_frames = min(len(psi_frames), len(w_frames), len(times_psi), len(times_w))
    if n_frames == 0:
        raise ValueError("Nessun frame disponibile per il phase plot.")

    if len(psi_frames) != len(w_frames):
        print(
            f"Attenzione: numero di frame diverso tra psi ({len(psi_frames)}) "
            f"e w ({len(w_frames)}). Uso i primi {n_frames}."
        )

    psi_frames = psi_frames[:n_frames]
    w_frames = w_frames[:n_frames]
    times = times_w[:n_frames]

    histograms, (psi_min, psi_max, w_min, w_max), max_count = build_histograms(
        psi_frames, w_frames, bins=bins
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    initial = np.ma.masked_less_equal(histograms[0].T, 0.0)

    im = ax.imshow(
        initial,
        origin="lower",
        extent=[psi_min, psi_max, w_min, w_max],
        aspect="auto",
        cmap="magma",
        norm=LogNorm(vmin=1.0, vmax=max_count),
        interpolation="nearest",
        animated=True,
    )

    fig.colorbar(im, ax=ax, label="Conteggio")
    ax.set_xlabel("psi")
    ax.set_ylabel("w")
    title = ax.set_title(f"Phase Plot w vs psi - t = {times[0]:.3f}")

    def update(frame_idx):
        hist = np.ma.masked_less_equal(histograms[frame_idx].T, 0.0)
        im.set_array(hist)
        title.set_text(f"Phase Plot w vs psi - t = {times[frame_idx]:.3f}")
        return im, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
    )

    plt.tight_layout()

    try:
        save_animation_mp4(ani, output_file, fps=fps, dpi=150, bitrate=1800)
        print(f"Video salvato come '{output_file}'")
    except RuntimeError as exc:
        print(exc)

    plt.show()


if __name__ == "__main__":
    animate_phase_plot()
