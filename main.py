from pathlib import Path

from mypylibs.tracking import SCENES, process_video


def main():
    root = Path(__file__).resolve().parent
    inputs = root / "in"
    outputs = root / "out"
    outputs.mkdir(exist_ok=True)

    for case, scene in SCENES.items():
        video = inputs / f"{case}.mp4"
        if case == "04" and not video.exists():
            continue
        process_video(video, outputs, scene)


if __name__ == "__main__":
    main()
