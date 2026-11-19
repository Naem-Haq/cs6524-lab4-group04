from collections import deque

import cv2 as cv
import numpy as np


MEASURED = (255, 234, 69)
PREDICTED = (30, 175, 255)
CORRECTED = (96, 240, 102)
UNCORRECTED = (232, 109, 249)
SCENE_FLOW = (210, 210, 210)
INK = (22, 22, 22)
FONT = cv.FONT_HERSHEY_SIMPLEX


def pixel(point):
    return tuple(int(round(value)) for value in point[:2])


def caption(frame, title, items):
    cv.rectangle(frame, (0, 0), (frame.shape[1], 49), INK, cv.FILLED)
    cv.putText(frame, title, (11, 18), FONT, .47, (255, 255, 255), 1, cv.LINE_AA)
    x = 12
    for label, colour in items:
        cv.line(frame, (x, 36), (x + 14, 36), colour, 2, cv.LINE_AA)
        cv.putText(frame, label, (x + 20, 40), FONT, .39, (245, 245, 245), 1, cv.LINE_AA)
        x += cv.getTextSize(label, FONT, .39, 1)[0][0] + 40


def arrow(frame, start, end, colour, thickness=1):
    first, last = pixel(start), pixel(end)
    if first != last:
        cv.arrowedLine(frame, first, last, INK, thickness + 2, cv.LINE_AA, tipLength=.24)
        cv.arrowedLine(frame, first, last, colour, thickness, cv.LINE_AA, tipLength=.24)


class FlowOverlay:
    def draw(self, frame, scene_start, scene_end, target_start, target_end, index):
        canvas = frame.copy()
        for start, end in zip(scene_start, scene_end):
            if np.linalg.norm(end - start) >= .5:
                arrow(canvas, start, start + 3 * (end - start), SCENE_FLOW)
        for start, end in zip(target_start, target_end):
            cv.circle(canvas, pixel(start), 2, MEASURED, cv.FILLED, cv.LINE_AA)
            arrow(canvas, start, start + 3 * (end - start), MEASURED)
        caption(canvas, f"Lucas-Kanade optical flow | frame {index:03d} | vectors x3",
                (("Scene", SCENE_FLOW), ("Target", MEASURED)))
        return canvas


def draw_trail(frame, history, colour, thickness, dotted=False):
    for (index, start), (_, end) in zip(history, list(history)[1:]):
        if start is None or end is None or (dotted and index % 2):
            continue
        cv.line(frame, pixel(start), pixel(end), colour, thickness, cv.LINE_AA)


class TrajectoryOverlay:
    def __init__(self):
        self.predicted = deque(maxlen=65)
        self.measured = deque(maxlen=65)
        self.corrected = deque(maxlen=65)
        self.counterfactuals = []

    def draw(self, frame, prediction, measurement, corrected, branches, index):
        canvas = frame.copy()
        was_missing = self.measured and self.measured[-1][1] is None
        prediction_on_trail = prediction if measurement is None or was_missing else None
        self.predicted.append((index, prediction_on_trail))
        self.measured.append((index, measurement))
        self.corrected.append((index, corrected))
        draw_trail(canvas, self.predicted, PREDICTED, 2, dotted=True)
        draw_trail(canvas, self.corrected, CORRECTED, 4)
        draw_trail(canvas, self.measured, MEASURED, 1)

        if len(branches) > len(self.counterfactuals):
            # The new branch starts at the same prediction as the returning live track.
            self.counterfactuals.append(deque(maxlen=65))
        for history, branch in zip(self.counterfactuals, branches):
            history.append((index, branch))
            draw_trail(canvas, history, UNCORRECTED, 2)
            cv.circle(canvas, pixel(branch), 5, UNCORRECTED, 2, cv.LINE_AA)

        for point, radius, colour, thickness in (
            (prediction, 9, PREDICTED, 2),
            (corrected, 6, CORRECTED, 2),
            (measurement, 3, MEASURED, cv.FILLED),
        ):
            if point is not None:
                cv.circle(canvas, pixel(point), radius, colour, thickness, cv.LINE_AA)

        state = corrected if corrected is not None else prediction
        if state is not None and index % 5 == 0:
            colour = CORRECTED if corrected is not None else PREDICTED
            arrow(canvas, state[:2], state[:2] + 5 * state[2:], colour, 2)
            for branch in branches:
                arrow(canvas, branch[:2], branch[:2] + 5 * branch[2:], UNCORRECTED, 2)

        status = "Visible" if measurement is not None else "Prediction only"
        if state is None:
            status = "Waiting for target"
        elif was_missing and measurement is not None and prediction is not None:
            status = "Reacquired"
        caption(canvas, f"Object trajectory | frame {index:03d} | {status} | velocity x5",
                (("Measured", MEASURED), ("Predicted", PREDICTED), ("Corrected", CORRECTED),
                 ("No reappearance", UNCORRECTED)))
        return canvas
