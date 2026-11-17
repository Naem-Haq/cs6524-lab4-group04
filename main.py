from pathlib import Path

from mypylibs.optical_flow import write_flow_video


def main():
    inputs = Path("in")
    outputs = Path("out")
    outputs.mkdir(exist_ok=True)

    for case in ("01", "02", "03", "04"):
        source = inputs / f"{case}.mp4"
        if case == "04" and not source.exists():
            continue
        destination = outputs / f"{case}_of.mp4"
        count = write_flow_video(source, destination)
        print(f"{source.name}: wrote {count} frames to {destination}")


if __name__ == "__main__":
    main()
