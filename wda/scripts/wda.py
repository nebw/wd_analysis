import numpy as np
import click
from json_tricks.np import dumps
import os
from matplotlib.backends.backend_pdf import PdfPages

from wda import analysis, io, visualization


@click.command()
@click.option('--track', type=click.Path(exists=True, file_okay=True,
                                         dir_okay=False, readable=True),
              required=True, help='Path to track csv/json')
def main(track):
    tracks = io.load_tracks(track)
    fname = os.path.splitext(os.path.split(track)[-1])[0]

    results = []
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

        open('{}-track{}-waggles.json'.format(fname, track_idx), 'w').write(dumps(waggles))

        results.append((track_idx, analysis.extract_most_likely_angle(waggles)))

    np.savetxt('{}-results.csv'.format(fname), np.array(results), delimiter=",",
               header='track_id,angle', fmt=['%d', '%s'])
