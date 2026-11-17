from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2 as cv
import numpy as np


LK_OPTIONS = {
    "winSize": (27, 27),
    "maxLevel": 3,
    "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, .01),
}
WORK_WIDTH = 640
MOTION_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
ARROW_SCALE = 3


@dataclass(frozen=True)
class FrameFlow:
    start: np.ndarray
    end: np.ndarray

    @property
    def velocity(self):
        """Displacement in processing pixels per frame."""
        return self.end - self.start


def select_keypoints(previous, current):
    # Reserve some points for areas with motion so stationary texture cannot
    # use up the entire corner budget.
    difference = cv.absdiff(previous, current)
    _, changed = cv.threshold(difference, 12, 255, cv.THRESH_BINARY)
    nearby = cv.dilate(changed, MOTION_KERNEL)
    moving = cv.goodFeaturesToTrack(previous, 100, .006, 6, mask=nearby)
    background = cv.goodFeaturesToTrack(previous, 120, .01, 14, mask=cv.bitwise_not(nearby))
    points = [group for group in (moving, background) if group is not None]
    return np.concatenate(points) if points else None


def estimate_flow(previous, current, points, initial_shift=None):
    """Track previous-frame keypoints and keep consistent forward/backward matches."""
    empty = np.empty((0, 2), dtype=np.float32)
    if points is None or not len(points):
        return FrameFlow(empty, empty)

    guess = points + initial_shift if initial_shift is not None else None
    flags = cv.OPTFLOW_USE_INITIAL_FLOW if guess is not None else 0
    moved, forward_ok, _ = cv.calcOpticalFlowPyrLK(
        previous, current, points, guess, flags=flags, **LK_OPTIONS
    )
    if moved is None:
        return FrameFlow(empty, empty)
    positions = moved[:, 0]
    height, width = current.shape
    valid = (
        forward_ok.ravel().astype(bool)
        & (positions[:, 0] >= 0)
        & (positions[:, 0] < width)
        & (positions[:, 1] >= 0)
        & (positions[:, 1] < height)
    )
    if not valid.any():
        return FrameFlow(empty, empty)

    points, moved = points[valid], moved[valid]
    guess = points.copy() if flags else None
    returned, backward_ok, _ = cv.calcOpticalFlowPyrLK(
        current, previous, moved, guess, flags=flags, **LK_OPTIONS
    )
    if returned is None:
        return FrameFlow(empty, empty)
    error = np.linalg.norm((points - returned)[:, 0], axis=1)
    valid = backward_ok.ravel().astype(bool) & (error < 1.5)
    return FrameFlow(points[valid, 0], moved[valid, 0])


def bonus_ball(frame, gray):
    # The yellow-green ball can move more than its diameter between frames.
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (21, 75, 75), (48, 255, 255))
    height = gray.shape[0]
    mask[:round(.28 * height)] = 0
    mask[round(.84 * height):] = 0
    _, labels, stats, centres = cv.connectedComponentsWithStats(mask)
    areas = stats[:, cv.CC_STAT_AREA]
    candidates = np.flatnonzero((areas >= 45) & (areas <= 950))
    if not len(candidates):
        return None, None
    ball = candidates[np.argmax(areas[candidates])]
    ball_mask = (labels == ball).astype(np.uint8) * 255
    points = cv.goodFeaturesToTrack(gray, 35, .004, 3, mask=ball_mask, blockSize=3)
    return centres[ball].astype(np.float32), points


def draw_flow(frame, flows, index):
    for flow in flows:
        velocities = flow.velocity
        moving = np.linalg.norm(velocities, axis=1) >= .35
        for start, velocity in zip(flow.start[moving], velocities[moving]):
            first = tuple(np.rint(start).astype(int))
            last = tuple(np.rint(start + ARROW_SCALE * velocity).astype(int))
            cv.arrowedLine(frame, first, last, (20, 20, 20), 3, cv.LINE_AA, tipLength=.27)
            cv.arrowedLine(frame, first, last, (255, 237, 81), 1, cv.LINE_AA, tipLength=.27)

    caption = f"Lucas-Kanade optical flow | frame {index:03d} | arrows x{ARROW_SCALE}"
    cv.rectangle(frame, (0, 0), (frame.shape[1], 28), (22, 22, 22), cv.FILLED)
    cv.putText(frame, caption, (10, 19), cv.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv.LINE_AA)
    return frame


def write_flow_video(source, destination):
    capture = cv.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open {source}")
        fps = capture.get(cv.CAP_PROP_FPS)
        ok, frame = capture.read()
        if not ok or fps <= 0:
            raise OSError(f"Cannot read frames or frame rate from {source}")

        height, width = frame.shape[:2]
        process_width = min(WORK_WIDTH, width - width % 2)
        size = process_width, max(2, 2 * round(height * process_width / (2 * width)))
        # Publish the new video only after its writer has closed successfully.
        with TemporaryDirectory(dir=destination.parent) as directory:
            scratch = Path(directory) / destination.name
            writer = cv.VideoWriter(str(scratch), cv.VideoWriter_fourcc(*"mp4v"), fps, size)
            try:
                if not writer.isOpened():
                    raise OSError(f"Cannot write {destination}")
                previous = None
                previous_ball = previous_ball_points = None
                count = 0
                while ok:
                    resized = cv.resize(frame, size, interpolation=cv.INTER_AREA)
                    current = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
                    flows = []
                    if previous is not None:
                        points = select_keypoints(previous, current)
                        flows.append(estimate_flow(previous, current, points))
                    if source.stem == "04":
                        ball, ball_points = bonus_ball(resized, current)
                        if ball is not None and previous_ball is not None:
                            shift = ball - previous_ball
                            ball_flow = estimate_flow(previous, current, previous_ball_points, shift)
                            consistent = np.linalg.norm(ball_flow.velocity - shift, axis=1) <= 8
                            flows.append(FrameFlow(ball_flow.start[consistent], ball_flow.end[consistent]))
                        previous_ball, previous_ball_points = ball, ball_points
                    writer.write(draw_flow(resized, flows, count))
                    previous = current
                    count += 1
                    ok, frame = capture.read()
            finally:
                writer.release()
            scratch.replace(destination)
        return count
    finally:
        capture.release()
