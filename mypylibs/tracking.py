from dataclasses import dataclass

import cv2 as cv
import numpy as np

from mypylibs.kalman import PointKalman, measurement_noise
from mypylibs.optical_flow import estimate_flow
from mypylibs.visualization import FlowOverlay, TrajectoryOverlay


@dataclass(frozen=True)
class Scene:
    background_frame: int
    vertical_range: tuple[float, float]
    occluder: tuple[float, float]
    minimum_area: int
    hsv_min: tuple[int, int, int]
    hsv_max: tuple[int, int, int]
    difference: int
    target_half_width: int
    acceleration_noise: float
    gravity: float = 0


# The fixed camera lets us subtract an empty background and mask the occluder.
# Case 02 represents the pen by the center of its blue cap.
SCENES = {
    "01": Scene(50, (.12, .63), (.250, .702), 600, (4, 85, 55), (38, 255, 255), 23, 88, .7),
    "02": Scene(85, (.15, .44), (.449, .652), 32, (86, 58, 42), (136, 255, 255), 18, 6, .2),
    "03": Scene(10, (.33, .73), (.399, .652), 600, (8, 70, 47), (46, 255, 255), 19, 52, .8),
    "04": Scene(5, (.28, .84), (.253, .471), 52, (21, 75, 75), (48, 255, 255), 20, 15, 12, 5),
}

WORK_WIDTH = 640
KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
EMPTY_POINTS = np.empty((0, 2), dtype=np.float32)


def follow_points(previous, current, points, shift=None):
    """Track keypoints with Naem's forward/backward Lucas-Kanade estimate_flow()."""
    if previous is None or points is None or not len(points):
        return EMPTY_POINTS.copy(), EMPTY_POINTS.copy()
    flow = estimate_flow(previous, current, points, initial_shift=shift)
    return flow.start, flow.end


class Silhouette:
    def __init__(self, background, scene):
        self.scene = scene
        self.background = cv.GaussianBlur(background, (5, 5), 0)
        height, width = background.shape[:2]
        top, bottom = (round(height * value) for value in scene.vertical_range)
        self.left, self.right = (round(width * value) for value in scene.occluder)
        self.region = np.zeros((height, width), np.uint8)
        self.region[top:bottom, :self.left] = 255
        self.region[top:bottom, self.right:] = 255

    def detect(self, frame, expected, gap):
        difference = cv.absdiff(cv.GaussianBlur(frame, (5, 5), 0), self.background)
        changed = (np.max(difference, axis=2) >= self.scene.difference).astype(np.uint8) * 255
        colour = cv.inRange(cv.cvtColor(frame, cv.COLOR_BGR2HSV), self.scene.hsv_min, self.scene.hsv_max)
        foreground = changed & colour & self.region
        foreground = cv.morphologyEx(foreground, cv.MORPH_CLOSE, KERNEL, iterations=2)
        foreground = cv.morphologyEx(foreground, cv.MORPH_OPEN, KERNEL)
        _, labels, stats, centres = cv.connectedComponentsWithStats(foreground)
        candidates = np.flatnonzero(stats[1:, cv.CC_STAT_AREA] >= self.scene.minimum_area) + 1
        if not len(candidates):
            return None, None
        areas = stats[candidates, cv.CC_STAT_AREA]
        scores = np.log(areas)
        if expected is not None:
            distances = np.linalg.norm(centres[candidates] - expected, axis=1)
            # Widen the search during an occlusion, since the object may slow or bounce.
            scores -= distances / min(100, 20 + 5 * gap)
        best = candidates[np.argmax(scores)]
        centre = centres[best].astype(np.float32)
        x, _, width, _, _ = stats[best]
        right = x + width
        # When a silhouette is clipped, infer x from its unobstructed edge.
        half_width = self.scene.target_half_width
        if width < 2 * half_width:
            if x <= 1 or self.right <= x <= self.right + 1:
                centre[0] = right - half_width
            elif self.left - 1 <= right <= self.left or right >= frame.shape[1] - 1:
                centre[0] = x + half_width
        return centre, (labels == best).astype(np.uint8) * 255


class KalmanTrack:
    """Patrick's PointKalman wrapper: predict through gaps, clone on reacquisition."""

    def __init__(self, scene):
        self.accel_std = scene.acceleration_noise
        self.gravity = float(scene.gravity)
        # Bonus clip 04 is more precise when gravity is modelled; others are noisier.
        self.position_noise = 1.5 if scene.gravity else 3.0
        self.filter = None
        self.started = False
        self.gap = 0
        self.branch_filters = []
        self.reappearances = []

    @property
    def branches(self):
        """Prediction-only clone states for the trajectory overlay."""
        return [branch.state for branch in self.branch_filters]

    def _apply_gravity(self, track):
        # PointKalman has no control input; apply the same discrete gravity step
        # the silhouette tracker used: [0, 0.5 g, 0, g] on statePost.
        if not self.gravity:
            return
        track.kf.statePost[1, 0] += 0.5 * self.gravity
        track.kf.statePost[3, 0] += self.gravity

    def predict(self):
        for branch in self.branch_filters:
            branch.predict()
            self._apply_gravity(branch)
        if self.filter is None:
            return None
        self.filter.predict()
        self._apply_gravity(self.filter)
        return self.filter.state

    def update(self, prediction, measurement, velocity, flow_points, flow_velocities, index):
        if measurement is None:
            if prediction is not None:
                self.gap += 1
            return None

        if not self.started:
            initial_velocity = (0.0, 0.0) if velocity is None else tuple(velocity)
            self.filter = PointKalman(
                measurement,
                velocity=initial_velocity,
                measure_velocity=True,
                accel_std=self.accel_std,
                pos_std=self.position_noise,
            )
            self.started = True
            return self.filter.state

        if self.gap >= 3:
            error = float(np.linalg.norm(prediction[:2] - measurement))
            self.reappearances.append((index, self.gap, error))
            # Clone before correct so the twin keeps the prediction-only path.
            self.branch_filters.append(self.filter.clone())

        self.gap = 0
        if velocity is not None and len(flow_points) >= 2:
            noise = measurement_noise(
                points=flow_points,
                velocities=flow_velocities,
                measure_velocity=True,
                pos_std=self.position_noise,
            )
            return self.filter.correct(measurement, velocity=velocity, R=noise)

        # Silhouette-only fix: trust position, keep the predicted velocity soft.
        soft = max(8.0, 4.0 * self.position_noise)
        noise = measurement_noise(
            measure_velocity=True,
            pos_std=2.0 * self.position_noise,
            vel_std=soft,
        )
        return self.filter.correct(measurement, velocity=prediction[2:], R=noise)


def object_measurement(centre, mask, previous_centre, points_from, points_to):
    if centre is None or previous_centre is None or not len(points_to):
        return centre, None, EMPTY_POINTS.copy(), EMPTY_POINTS.copy()
    height, width = mask.shape
    x = np.clip(np.rint(points_to[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(points_to[:, 1]).astype(int), 0, height - 1)
    supported = cv.dilate(mask, KERNEL, iterations=2)[y, x] != 0
    points_from, points_to = points_from[supported], points_to[supported]
    if len(points_to) < 2:
        return centre, None, EMPTY_POINTS.copy(), EMPTY_POINTS.copy()
    displacement = points_to - points_from
    median = np.median(displacement, axis=0)
    consistent = np.linalg.norm(displacement - median, axis=1) <= 4
    if consistent.sum() < 2:
        return centre, None, EMPTY_POINTS.copy(), EMPTY_POINTS.copy()
    kept_from = points_from[consistent]
    kept_to = points_to[consistent]
    velocity = np.median(displacement[consistent], axis=0).astype(np.float32)
    flow_centre = previous_centre + velocity
    # Anchor the integrated LK displacement to the silhouette to limit drift.
    measurement = (.55 * flow_centre + .45 * centre).astype(np.float32)
    return measurement, velocity, kept_from, kept_to


def writer(path, fps, size):
    video = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not video.isOpened():
        video.release()
        raise OSError(f"Cannot write {path}")
    return video


def process_video(path, outputs, scene):
    source = cv.VideoCapture(str(path))
    flow_video = trajectory_video = None
    try:
        if not source.isOpened():
            raise OSError(f"Cannot open {path}")
        width = int(source.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(source.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = source.get(cv.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(f"Invalid video geometry or frame rate in {path}")
        size = WORK_WIDTH, 2 * round(height * WORK_WIDTH / (2 * width))
        source.set(cv.CAP_PROP_POS_FRAMES, scene.background_frame)
        ok, background = source.read()
        if not ok:
            raise OSError(f"Cannot read background frame {scene.background_frame} from {path}")
        detector = Silhouette(cv.resize(background, size, interpolation=cv.INTER_AREA), scene)
        source.set(cv.CAP_PROP_POS_FRAMES, 0)
        flow_video = writer(outputs / f"{path.stem}_of.mp4", fps, size)
        trajectory_video = writer(outputs / f"{path.stem}_tr.mp4", fps, size)
        track = KalmanTrack(scene)
        flow_overlay, trajectory_overlay = FlowOverlay(), TrajectoryOverlay()
        previous_gray = previous_measurement = previous_centre = features = None
        scene_features = None
        frames = 0

        while True:
            ok, frame = source.read()
            if not ok:
                break
            frame = cv.resize(frame, size, interpolation=cv.INTER_AREA)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            prediction = track.predict()
            centre, mask = detector.detect(
                frame, prediction[:2] if prediction is not None else None, track.gap
            )
            shift = centre - previous_centre if centre is not None and previous_centre is not None else None
            start, end = follow_points(previous_gray, gray, features, shift)
            measurement, velocity, start, end = object_measurement(
                centre, mask, previous_measurement, start, end
            )
            flow_velocities = end - start if len(end) else None
            corrected = track.update(
                prediction, measurement, velocity, start, flow_velocities, frames
            )
            scene_start, scene_end = follow_points(previous_gray, gray, scene_features)
            flow_video.write(flow_overlay.draw(frame, scene_start, scene_end, start, end, frames))
            trajectory_video.write(
                trajectory_overlay.draw(
                    frame, prediction, measurement, corrected, track.branches, frames
                )
            )
            scene_features = scene_end.reshape(-1, 1, 2) if len(scene_end) else None
            if frames % 8 == 0 or scene_features is None or len(scene_end) < 40:
                scene_features = cv.goodFeaturesToTrack(gray, 140, .015, 13)
            features = None
            if mask is not None:
                features = cv.goodFeaturesToTrack(gray, 65, .004, 3, mask=mask, blockSize=3)
            previous_gray, previous_measurement, previous_centre = gray, measurement, centre
            frames += 1

        if not frames or not track.started:
            raise ValueError(f"No target tracked in {path}")
        print(f"{path.name}: {frames} frames at {fps:g} fps")
        for index, gap, error in track.reappearances:
            print(
                f"  Frame {index}: reacquired after {gap} frames; "
                f"prediction error {error * width / WORK_WIDTH:.1f} source px"
            )
    finally:
        source.release()
        for video in (flow_video, trajectory_video):
            if video is not None:
                video.release()
