import click
import itertools
from pathos.multiprocessing import ProcessPool, cpu_count
from tqdm import tqdm
import numpy as np
import os

from wda import analysis, io


def process_track(track, track_idx, fname, subfolder, verbose=False):
    results = []
    if verbose:
        print('Track {}'.format(track_idx + 1))
    cos_theta_smooth, detected_waggles, detected_waggles_median = \
        analysis.detect_waggles(track)

    waggles = analysis.extract_waggles(track, detected_waggles_median)
    if verbose:
        print('Number of detected waggle runs: {}'.format(len(waggles)))
    if len(waggles) == 0:
        return None

    dance_start_time = track.t.iloc[0] / 30
    dance_end_time = track.t.iloc[-1] / 30
    most_likely_dance_angle = analysis.extract_most_likely_angle(waggles)

    for waggle in waggles:
        results.append((
            fname,
            subfolder,
            track_idx,
            len(waggle['points']),
            waggle['is_len_outlier'],
            waggle['best_theta'],
            waggle['is_theta_outlier'],
            waggle['direction'],
            waggle['start_time_in_video'],
            waggle['end_time_in_video'],
            dance_start_time,
            dance_end_time,
            most_likely_dance_angle

        ))

    return results


def get_tracks(path, verbose=False):
    tracks = io.load_tracks(path)
    fname = os.path.splitext(os.path.split(path)[-1])[0]
    subfolder = path.split(os.sep)[-2]

    loaded_tracks = []
    for track_idx, track in enumerate(tracks):
        loaded_tracks.append((track, track_idx, fname, subfolder))

    return loaded_tracks


@click.command()
@click.option('--track', type=click.Path(exists=True, file_okay=True,
                                         dir_okay=False, readable=True),
              required=False, help='Path to track csv/json')
@click.option('--path', type=click.Path(exists=True, file_okay=False,
                                        dir_okay=True, readable=True),
              required=False, help='Path to track csv/json directory')
def main(track=None, path=None):
    pool = ProcessPool(nodes=cpu_count())

    if track is not None:
        tracks = [track]
    else:
        if path is None:
            raise click.BadParameter('either track or path must be given')
        tracks = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".json") or file.endswith(".csv"):
                    tracks.append(os.path.join(root, file))

    loaded_tracks = tqdm(pool.imap(get_tracks, tracks), 'Loading data', total=len(tracks))
    loaded_tracks = list(filter(lambda t: t is not None,
                                itertools.chain(*loaded_tracks)))

    results = tqdm(pool.uimap(lambda t: process_track(*t), loaded_tracks),
                   'Processing tracks', total=len(loaded_tracks))
    results = list(itertools.chain(*results))

    header = 'fname,subfolder,track_idx,waggle_len,is_len_outlier,waggle_theta,is_theta_outlier' +\
        'waggle_direction,waggle_start_time,waggle_end_time,dance_start_time,dance_end_time' +\
        'dance_angle'
    fmt = ['%s'] * len(header)
    np.savetxt('results.csv', np.array(results), delimiter=",", comments='',
               header=header, fmt=fmt)
