import json
import numpy as np
import pandas as pd

from wda import features


def is_float(obj):
    try:
        float(obj)
        return True
    except:
        return False


def load_csv_tracks(path, sep=None):
    data = pd.read_csv(path, sep=sep, header=None,
                       names=('t', 'id', 'x', 'y', 'behaviour'),
                       skiprows=1, usecols=range(5), engine='python')

    tracks = []
    for track_id in data.id.unique():
        track = data[data.id == track_id]
        # remove corrupt columns
        track = track[track.iloc[:, 2].apply(is_float) & track.iloc[:, 3].apply(is_float)]
        track[['x','y']] = track[['x','y']].apply(lambda c: c.astype(np.float64))

        tracks.append(track)

    return tracks


def load_json_tracks(path):
    data = json.load(open(path, 'r'))
    assert(len(data.keys()) == 1)
    data = pd.DataFrame(data[list(data.keys())[0]])

    tracks = []
    for track_id in data.Id.unique():
        track = data[data.Id == track_id]
        if len(track) < 100:
            continue

        track = track[['Center.x', 'Center.y', 'Frame', 'Id', 'State']]
        track.rename(
            index=str,
            inplace=True,
            columns={'Center.x': 'x',
                     'Center.y': 'y',
                     'Frame': 't',
                     'Id': 'id'})
        track[['x','y']] = track[['x','y']].apply(lambda c: c.astype(np.float64))
        track.reset_index(inplace=True)
        tracks.append(track)

    return tracks


def load_tracks(path, preprocess=True):
    if path.endswith('json'):
        tracks = load_json_tracks(path)
    elif path.endswith('csv'):
        tracks = load_csv_tracks(path)
    else:
        raise ValueError('File type not supported')

    if preprocess:
        tracks = [features.preprocess_track(track) for track in tracks]

    return tracks
