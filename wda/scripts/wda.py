import numpy as np
import click
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from wda import analysis, io, visualization


@click.command()
@click.option('--track', type=click.Path(exists=True, file_okay=True,
                                         dir_okay=False, readable=True),
              required=True, help='Path to track csv/json')
def main(track):
    tracks = io.load_tracks(track)
    fname = os.path.splitext(track)[0]

    print('Number of tracks in file: {}'.format(len(tracks)))
    print()
    for track_idx, track in enumerate(tracks):
        print('Track {}'.format(track_idx + 1))
        pp = PdfPages('{}-track{}-figures.pdf'.format(fname, track_idx))
        cos_theta_smooth, detected_waggles, detected_waggles_median = \
            analysis.detect_waggles(track)
        visualization.plot_features(cos_theta_smooth, detected_waggles, detected_waggles_median)
        pp.savefig(bbox_inches='tight')

        visualization.plot_track(track.iloc[:-1], Y=detected_waggles_median.astype(np.int32))
        pp.savefig(bbox_inches='tight')

        waggles = analysis.extract_waggles(track, detected_waggles_median)
        print('Number of detected waggle runs: {}'.format(len(waggles)))
        visualization.plot_waggles(waggles)
        pp.savefig(bbox_inches='tight')

        visualization.plot_angle_distribution(waggles)
        pp.savefig(bbox_inches='tight')
        pp.close()
