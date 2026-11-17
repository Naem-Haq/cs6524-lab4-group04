"""Constant-velocity Kalman filter for tracking one point in the image plane.

Units are pixels and frames (dt = 1), so velocity is pixels/frame.

State      x = [px, py, vx, vy]^T
Transition px' = px + vx*dt,  vx' = vx   (constant velocity, no control input -> no G/u)
Measurement z = [px, py]            (H is 2x4), or
            z = [px, py, vx, vy]    (H is 4x4) when measure_velocity=True,
            since optical flow gives us a velocity estimate as well.
"""

import cv2 as cv
import numpy as np


# --- Tunable defaults (pixels / frames) -------------------------------------
DT = 1.0
ACCEL_STD = 0.05      # sigma_a: how much the true velocity may wander per frame (Q)
POS_STD_FLOOR = 1.0   # lower bounds on the measurement std, so R never collapses to 0
VEL_STD_FLOOR = 0.3
INIT_VEL_STD = 5.0    # P0 velocity std when velocity is not yet known


def transition_matrix(dt=DT):
    """F for the constant-velocity model."""
    return np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def measurement_matrix(measure_velocity):
    """H: select position, or position and velocity, from the state."""
    rows = 4 if measure_velocity else 2
    return np.eye(rows, 4, dtype=np.float32)


def process_noise(accel_std=ACCEL_STD, dt=DT):
    """Q from the discrete white-noise-acceleration model (kalmanfilter.net).

    A random acceleration a ~ N(0, sigma_a^2) acting over one step moves the
    position by a*dt^2/2 and the velocity by a*dt, giving per axis
        Q_axis = sigma_a^2 * [[dt^4/4, dt^3/2],
                              [dt^3/2, dt^2  ]]
    """
    q = accel_std ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                   [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((4, 4))
    for axis in (0, 1):                      # x uses (0, 2), y uses (1, 3)
        idx = np.ix_([axis, axis + 2], [axis, axis + 2])
        Q[idx] = q
    return Q.astype(np.float32)


def measurement_noise(points=None, velocities=None, measure_velocity=False,
                      pos_std=None, vel_std=None):
    """R as a diagonal matrix, estimated from the optical-flow keypoints if given.

    For a centroid of N keypoints, the std of the mean is std(points)/sqrt(N),
    and the same holds for the mean velocity. Fixed stds can be passed instead.
    Floors stop R reaching 0 when only a handful of points agree.
    """
    if pos_std is None:
        pos_std = _std_of_mean(points, POS_STD_FLOOR)
    if vel_std is None:
        vel_std = _std_of_mean(velocities, VEL_STD_FLOOR)
    pos_std = np.maximum(np.broadcast_to(pos_std, (2,)), POS_STD_FLOOR)
    vel_std = np.maximum(np.broadcast_to(vel_std, (2,)), VEL_STD_FLOOR)
    stds = np.concatenate([pos_std, vel_std]) if measure_velocity else pos_std
    return np.diag(stds ** 2).astype(np.float32)


def _std_of_mean(samples, floor):
    if samples is None or len(samples) < 2:
        return np.array([floor, floor])
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, 2)
    return samples.std(axis=0, ddof=1) / np.sqrt(len(samples))


class PointKalman:
    """Wrapper around cv.KalmanFilter for one tracked point.

    predict() only        -> use during occlusion (OpenCV copies the prior into
                             the posterior, so repeated predicts extrapolate).
    predict() + correct() -> use whenever a measurement is available.
    clone()               -> make an independent copy, e.g. the
                             "what if it never came back" branch.
    """

    def __init__(self, position, velocity=(0.0, 0.0), measure_velocity=True,
                 accel_std=ACCEL_STD, pos_std=POS_STD_FLOOR, vel_std=INIT_VEL_STD,
                 R=None, dt=DT):
        self.measure_velocity = measure_velocity
        m = 4 if measure_velocity else 2
        kf = cv.KalmanFilter(4, m, 0, cv.CV_32F)
        kf.transitionMatrix = transition_matrix(dt)
        kf.measurementMatrix = measurement_matrix(measure_velocity)
        kf.processNoiseCov = process_noise(accel_std, dt)
        kf.measurementNoiseCov = (R if R is not None else
                                  measurement_noise(measure_velocity=measure_velocity))
        # P0: diagonal, from how uncertain the first measurement is.
        pos_var = np.broadcast_to(np.asarray(pos_std, dtype=np.float32) ** 2, (2,))
        vel_var = np.broadcast_to(np.asarray(vel_std, dtype=np.float32) ** 2, (2,))
        kf.errorCovPost = np.diag(np.concatenate([pos_var, vel_var])).astype(np.float32)
        kf.statePost = np.array([*position, *velocity], dtype=np.float32).reshape(4, 1)
        self.kf = kf

    # --- filter steps ------------------------------------------------------
    def predict(self):
        """Advance one frame; returns the predicted state [px, py, vx, vy]."""
        return self.kf.predict().ravel().copy()

    def correct(self, position, velocity=None, R=None):
        """Fuse a measurement; returns the corrected state.

        R lets you pass a per-frame measurement noise (e.g. from
        measurement_noise(points, velocities)) so frames where the optical
        flow is noisy are trusted less.
        """
        if R is not None:
            self.kf.measurementNoiseCov = np.asarray(R, dtype=np.float32)
        if self.measure_velocity:
            if velocity is None:
                raise ValueError("this filter measures velocity; pass velocity=")
            z = [*position, *velocity]
        else:
            z = list(position)
        z = np.array(z, dtype=np.float32).reshape(-1, 1)
        return self.kf.correct(z).ravel().copy()

    # --- inspection --------------------------------------------------------
    @property
    def state(self):
        return self.kf.statePost.ravel().copy()

    @property
    def position(self):
        return self.state[:2]

    @property
    def velocity(self):
        return self.state[2:]

    @property
    def covariance(self):
        return self.kf.errorCovPost.copy()

    @property
    def gain(self):
        """Kalman gain K from the last correct() (4 x m)."""
        return self.kf.gain.copy()

    def position_std(self):
        """1-sigma uncertainty of the position, handy for drawing an ellipse."""
        return np.sqrt(np.diag(self.kf.errorCovPost)[:2])

    def clone(self):
        """Independent deep copy (cv.KalmanFilter cannot be copied directly)."""
        twin = PointKalman.__new__(PointKalman)
        twin.measure_velocity = self.measure_velocity
        kf = cv.KalmanFilter(4, self.kf.measurementMatrix.shape[0], 0, cv.CV_32F)
        for name in ("transitionMatrix", "measurementMatrix", "processNoiseCov",
                     "measurementNoiseCov", "statePre", "statePost",
                     "errorCovPre", "errorCovPost", "gain"):
            setattr(kf, name, getattr(self.kf, name).copy())
        twin.kf = kf
        return twin
