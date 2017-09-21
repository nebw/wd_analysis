import itertools
import numpy as np
from pathos.multiprocessing import ProcessPool, cpu_count
import pandas as pd
import skimage
import skimage.filters
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import warnings
import matplotlib.pyplot as plt
from wda.ransac import ransac_thetas

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


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


def detect_waggles(track, window_size=51, threshold=1.55):
    cos_theta_smooth = get_angles(track, window_size)

    if threshold is None:
        threshold = determine_threshold(cos_theta_smooth)

    detected_waggles = cos_theta_smooth > threshold
    detected_waggles_median = detected_waggles.rolling(window_size, center=True).median()

    detected_waggles_median[np.isnan(detected_waggles_median)] = 0

    removed_nans = np.copy(detected_waggles_median)
    removed_nans[np.isnan(removed_nans)] = 0

    detected_waggles_median = removed_nans

    return cos_theta_smooth, detected_waggles, detected_waggles_median


def extract_waggle_angle(waggle_data):
    deltas = waggle_data['points'][1:] - waggle_data['points'][:-1]
    deltas_x_s = pd.Series(deltas[:, 0]).rolling(21, center=True).mean().dropna()
    deltas_y_s = pd.Series(deltas[:, 1]).rolling(21, center=True).mean().dropna()

    deltas_s = np.stack((deltas_x_s, deltas_y_s), axis=-1)

    thetas = np.arctan2(deltas_s[:, 1], deltas_s[:, 0])

    hist, edges = np.histogram(thetas, bins=255, range=(-np.pi, np.pi))

    theta_bins = edges[:-1] + (edges[1:] - edges[:-1]) / 2
    smoothed_hist = gaussian_filter1d(hist.astype(np.float64), 10, mode='wrap')

    best_theta = theta_bins[np.argmax(smoothed_hist)]

    waggle_data['theta_bins'] = theta_bins
    waggle_data['best_theta'] = best_theta
    waggle_data['smoothed_hist'] = smoothed_hist

    return waggle_data


def _regress(points, start_idx, end_idx):
    seq = points[start_idx:end_idx]
    times = np.array(list(range(len(seq))))
    inputs = seq
    targets = times
    inputs = StandardScaler().fit_transform(inputs.astype(np.float64))
    targets = StandardScaler().fit_transform(targets[:, None].astype(np.float64))

    pval = np.mean((
        sm.OLS(inputs[:, 0], targets).fit().mse_resid,
        sm.OLS(inputs[:, 1], targets).fit().mse_resid
    ))

    return start_idx, end_idx, pval


def regress(points):
    df = []

    # speed up computation for long waggles
    step = int(np.ceil(len(points) / 100))

    start_indices = range(0, points.shape[0] // 2 - 5, step)
    end_indices = range(points.shape[0] // 2 + 5, points.shape[0], step)
    indices = list(itertools.product(start_indices, end_indices))

    df = [_regress(points, s, e) for s, e in indices]
    df = pd.DataFrame(df, columns=['six', 'eix', 'stderr'])

    return df


def get_regression_argmin(x, y, model=lambda x, y: SVR().fit(x, y)):
    if x.ndim == 1:
        x = x[:, None]
    model = model(x, y)
    test_values = np.array(list(range(x.min(), x.max() + 1)))
    test_preds = model.predict(test_values[:, None])
    return test_values[np.argmin(test_preds)]


def detect_waggle_subsequence(waggle):
    points = waggle['points']
    times = waggle['times']

    regression_df = regress(points)
    regression_df['len'] = regression_df.eix - regression_df.six
    six = get_regression_argmin(regression_df.six, regression_df.stderr)
    eix = get_regression_argmin(regression_df.eix, regression_df.stderr)

    waggle['points'] = points[six:eix]
    waggle['times'] = times[six:eix]
    waggle['points_before'] = points[:six]
    waggle['points_after'] = points[eix:]

    return waggle


def detect_outliers(waggles):
    if len(waggles) > 2:
        waggle_lens = [len(w['points']) for w in waggles]
        waggle_thetas = [w['best_theta'] for w in waggles]

        len_outlier_model = EllipticEnvelope().fit(np.array(waggle_lens)[:, None])
        len_outliers = len_outlier_model.predict(np.array(waggle_lens)[:, None]) < 1

        for waggle, is_len_outlier in zip(waggles, len_outliers):
            waggle['is_len_outlier'] = is_len_outlier

        ransac_results = ransac_thetas(
            waggle_thetas, 3, len(waggle_thetas) * 5,
            np.pi / 4, int(len(waggle_thetas) * .7))

        if ransac_results is None:
            outlier_idxs = set()
        else:
            ransac_theta, inlier_idxs, outlier_idxs = ransac_results

        outlier_idxs = outlier_idxs if outlier_idxs is not None else set()

        for idx, waggle in enumerate(waggles):
            waggle['is_theta_outlier'] = idx in outlier_idxs
    else:
        for idx, waggle in enumerate(waggles):
            waggle['is_len_outlier'] = False
            waggle['is_theta_outlier'] = False

    return waggles


def detect_direction(waggle):
    all_points = np.concatenate((
        waggle['points_before'],
        waggle['points'],
        waggle['points_after']))

    six = waggle['points_before'].shape[0]
    eix = six + waggle['points'].shape[0]

    theta = -waggle['best_theta']

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    points = np.copy(all_points) - np.mean(all_points, axis=0)

    rotated_points = points @ R.T

    waggle_dir = 'right' if np.mean(rotated_points[eix:], axis=0)[1] < 0 else 'left'
    waggle['direction'] = waggle_dir

    return waggle


def _extract_waggle(track, start, end, debug=False):
    start = np.max((0, start - 25))
    end = np.min((len(track.index), end + 25))

    waggle_data = {}

    points = track[(track.index >= start) & (track.index <= end)][['x', 'y']].as_matrix()
    times = track[(track.index >= start) & (track.index <= end)][['t']].as_matrix()

    if len(points) == 0:
        return None

    waggle_data['points'] = points
    waggle_data['times'] = times
    waggle_data = detect_waggle_subsequence(waggle_data)
    waggle_data = extract_waggle_angle(waggle_data)
    waggle_data = detect_direction(waggle_data)
    waggle_data['distance_pixel'] = np.sqrt(np.sum((np.array(points[-1]) - np.array(points[0])) ** 2))

    if debug:
        points = waggle_data['points']
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], c='gray', alpha=.2, linestyle='--')
        plt.scatter(points[:, 0], points[:, 1], c='green')
        plt.scatter(waggle_data['points_before'][:, 0],
                    waggle_data['points_before'][:, 1], c='red')
        plt.scatter(waggle_data['points_after'][:, 0],
                    waggle_data['points_after'][:, 1], c='blue')
        plt.title(waggle_data['direction'])
        plt.show()

    waggle_data['start_time_in_video'] = times[0] / 30
    waggle_data['end_time_in_video'] = times[-1] / 30

    return waggle_data


def extract_waggles(track, detected_waggles_median, debug=False):
    edges = detected_waggles_median[1:] - detected_waggles_median[:-1]

    starts = np.where(edges == 1)
    ends = np.where(edges == -1)
    intervals = np.array(list(zip(starts, ends))).T[:, :, 0]

    if debug:
        waggles = [_extract_waggle(track, start, end, debug=True) for start, end in intervals]
    else:
        waggles = pool.uimap(lambda i: _extract_waggle(track, i[0], i[1]), intervals)
    waggles = list(filter(lambda t: t is not None, waggles))

    if len(waggles) > 0:
        waggles = detect_outliers(waggles)

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


pool = ProcessPool(nodes=cpu_count())
