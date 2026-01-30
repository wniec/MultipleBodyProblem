from moviepy import VideoFileClip
from pathlib import Path

def to_gif_1000_frames(mp4_file, gif_file, target_frames=1000):
    with VideoFileClip(mp4_file) as clip:

        fps = clip.fps
        duration_needed = target_frames / fps

        final_duration = min(duration_needed, clip.duration)

        print(f"Original FPS: {clip.fps}. Clip duration: {clip.duration}")
        print(f"Target Duration: {final_duration:.4f} seconds for {target_frames} frames")

        cut_clip = clip.subclipped(0, final_duration)

        cut_clip.write_gif(gif_file, fps=fps)


if __name__ == "__main__":
    mp4_file = "multiplebodyproblem/my_animation.mp4"
    gif_file = mp4_file.split("/")[-1].split(".")[0]   # -> name_of_mp4_file.gif

    Path("gifs").mkdir(exist_ok=True)
    to_gif_1000_frames(mp4_file, f"gifs/{gif_file}.gif", target_frames=400)