import numpy as np
import pandas as pd
import skimage
import skimage.filters
from scipy.ndimage.filters import gaussian_filter1d


def get_angles(track, window_size):
    tnext = track.iloc[1:][['dx', 'dy']].as_matrix()
    tcur = track.iloc[:-1][['dx', 'dy']].as_matrix()

    eps = np.finfo(np.float32).eps
    # element-wise dot product + normalization => cos(theta)
    cos_theta = np.sum(tnext * tcur, axis=1)\
        / (np.linalg.norm(tcur, axis=1) * np.linalg.norm(tnext, axis=1) + eps)

    cos_theta_smooth = pd.Series(np.arccos(cos_theta)).rolling(window_size, center=True).mean()

    return cos_theta_smooth


def determine_threshold(cos_theta_smooth):
    vals = cos_theta_smooth[(1 - np.isnan(cos_theta_smooth)).astype(np.bool)]
    return skimage.filters.threshold_otsu(vals)


def detect_waggles(track, window_size=51):
    cos_theta_smooth = get_angles(track, window_size)
    threshold = determine_threshold(cos_theta_smooth)

    detected_waggles = cos_theta_smooth > threshold
    detected_waggles_median = detected_waggles.rolling(window_size, center=True).median()

    detected_waggles_median[np.isnan(detected_waggles_median)] = 0

    removed_nans = np.copy(detected_waggles_median)
    removed_nans[np.isnan(removed_nans)] = 0

    detected_waggles_median = removed_nans

    return cos_theta_smooth, detected_waggles, detected_waggles_median


def extract_waggles(track, detected_waggles_median):
    edges = detected_waggles_median[1:] - detected_waggles_median[:-1]

    starts = np.where(edges == 1)
    ends = np.where(edges == -1)
    intervals = np.array(list(zip(starts, ends))).T[:, :, 0]

    waggles = []

    for start, end in intervals:
        waggle_data = {}

        points = track[(track.index >= start) & (track.index <= end)][['x', 'y']].as_matrix()

        if len(points) == 0:
            continue

        deltas = points[1:] - points[:-1]
        deltas_x_s = pd.Series(deltas[:, 0]).rolling(21, center=True).mean().dropna()
        deltas_y_s = pd.Series(deltas[:, 1]).rolling(21, center=True).mean().dropna()

        deltas_s = np.stack((deltas_x_s, deltas_y_s), axis=-1)

        thetas = np.arctan2(deltas_s[:, 1], deltas_s[:, 0])

        hist, edges = np.histogram(thetas, bins=255, range=(-np.pi, np.pi))

        theta_bins = edges[:-1] + (edges[1:] - edges[:-1]) / 2
        smoothed_hist = gaussian_filter1d(hist.astype(np.float64), 10, mode='wrap')

        best_theta = theta_bins[np.argmax(smoothed_hist)]

        waggle_data['start_time_in_video'] = track.iloc[start].t / 30
        waggle_data['end_time_in_video'] = track.iloc[end].t / 30
        waggle_data['points'] = points
        waggle_data['theta_bins'] = theta_bins
        waggle_data['best_theta'] = best_theta
        waggle_data['smoothed_hist'] = smoothed_hist

        waggles.append(waggle_data)

    return waggles


def rad_to_deg(theta):
    theta_deg = theta / np.pi * 180
    if theta_deg < 0:
        theta_deg = 360 + theta_deg
    return theta_deg


def extract_most_likely_angle(waggles):
    theta_bins = waggles[0]['theta_bins']
    histograms = [w['smoothed_hist'] for w in waggles]

    median_thetas = np.median(histograms, axis=0)
    median_thetas /= np.max(median_thetas)

    return theta_bins[np.argmax(median_thetas)]
