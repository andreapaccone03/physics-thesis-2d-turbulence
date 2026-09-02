import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import os
import shutil

DEFAULT_FFMPEG_PATH = (
    r"C:\Users\HP\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
)
ANIMATION_FPS = 15
ANIMATION_INTERVAL_MS = 1000.0 / ANIMATION_FPS
t_stamp = 2.5   # tempo di cui vuoi ottenere lo screenshot

VORTICITY_CMAP = LinearSegmentedColormap.from_list(
    "vorticity_dark_center",
    [
        (0.00, "#1d4ed8"),
        (0.35, "#60a5fa"),
        (0.50, "#ffffff"),
        (0.65, "#f87171"),
        (1.00, "#991b1b"),
    ],
)


def frame_vorticity_norm(frame):
    w_abs = max(np.percentile(np.abs(frame), 99.0), 1e-6)
    return TwoSlopeNorm(vmin=-w_abs, vcenter=0.0, vmax=w_abs)


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


def load_vorticity(filename):
    if not os.path.exists(filename):
        print(f"File {filename} non trovato.")
        return [], None, None, []

    frames = []
    times = []
    current_values = []
    xs_first = []
    ys_first = []
    collecting_grid = True

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Time:"):
                times.append(float(line.split(":")[1]))
                if current_values:
                    frames.append(np.array(current_values))
                    collecting_grid = False
                current_values = []
            else:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, w = float(parts[0]), float(parts[1]), float(parts[2])
                    current_values.append(w)
                    if collecting_grid:
                        xs_first.append(x)
                        ys_first.append(y)

    if current_values:
        frames.append(np.array(current_values))

    if not frames:
        return [], None, None, []

    xs_first = np.array(xs_first)
    ys_first = np.array(ys_first)
    X = np.unique(xs_first)
    Y = np.unique(ys_first)
    nx, ny = len(X), len(Y)

    reshaped_frames = []
    for frame in frames:
        if len(frame) == nx * ny:
            matrix = frame.reshape((nx, ny)).T
            reshaped_frames.append(matrix)
        else:
            print(f"Frame con dimensione inattesa: {len(frame)} invece di {nx*ny}")

    return times, X, Y, reshaped_frames


def animate_vorticity(filename="output/vorticity_evolution.dat"):
    times, X, Y, frames = load_vorticity(filename)
    if not frames:
        print("Nessun dato da mostrare.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    extent = [X.min(), X.max(), Y.min(), Y.max()]
    initial_norm = frame_vorticity_norm(frames[0])

    im = ax.imshow(
        frames[0],
        extent=extent,
        origin="lower",
        cmap=VORTICITY_CMAP,
        norm=initial_norm,
        animated=True,
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax, label="Vorticita")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"Vorticita - t = {times[0]:.3f}")

    def update(frame_idx):
        frame = frames[frame_idx]
        im.set_array(frame)
        im.set_norm(frame_vorticity_norm(frame))
        cbar.update_normal(im)
        title.set_text(f"Vorticita - t = {times[frame_idx]:.3f}")
        return im, title, cbar

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=ANIMATION_INTERVAL_MS,
        blit=False,
    )

    plt.tight_layout()

    try:
        save_animation_mp4(
            ani,
            "output/vorticity_animation.mp4",
            fps=ANIMATION_FPS,
            dpi=150,
            bitrate=1800,
        )
        print("Video salvato come 'output/vorticity_animation.mp4'")
    except RuntimeError as exc:
        print(exc)

    plt.show()

def save_vorticity_snapshot(filename="output/vorticity_evolution.dat", t_stamp=0.0):
    times, X, Y, frames = load_vorticity(filename)
    if not frames:
        print("Nessun dato disponibile.")
        return

    # Trova il frame con tempo più vicino a t_stamp
    idx = np.argmin(np.abs(np.array(times) - t_stamp))

    fig, ax = plt.subplots(figsize=(8, 6))

    extent = [X.min(), X.max(), Y.min(), Y.max()]

    im = ax.imshow(
        frames[idx],
        extent=extent,
        origin="lower",
        cmap=VORTICITY_CMAP,
        norm=frame_vorticity_norm(frames[idx]),
        aspect="equal",
    )

    fig.colorbar(im, ax=ax, label="Vorticita")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Vorticita - t = {times[idx]:.3f}")

    plt.tight_layout()

    output_name = f"output/vorticity_t_{times[idx]:.3f}.png"
    plt.savefig(output_name, dpi=300)
    print(f"Immagine salvata in '{output_name}'")

    plt.show()


if __name__ == "__main__":
    animate_vorticity()
    save_vorticity_snapshot(t_stamp=t_stamp)
