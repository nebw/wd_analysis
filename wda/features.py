import numpy as np
import pandas as pd


def preprocess_track(track, window_size=51):
    track['dt'] = track.t.diff()

    track['dx'] = track.x.diff() / track['dt']
    track['dy'] = track.y.diff() / track['dt']

    smoothed = track[['dx', 'dy']].rolling(window_size, center=True).mean()

    track['sdx'] = smoothed.dx
    track['sdy'] = smoothed.dy

    return track
