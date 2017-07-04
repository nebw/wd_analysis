import itertools
import numpy as np
import click
from json_tricks.np import dumps
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from wda import analysis, io, visualization


def track_file(track):
    tracks = io.load_tracks(track)
    fname = os.path.splitext(os.path.split(track)[-1])[0]
    subfolder = track.split(os.sep)[-2]

    results = []
    print('Number of tracks in file: {}'.format(len(tracks)))
    print()
    print(track)
    for track_idx, track in enumerate(tracks):
        print('Track {}'.format(track_idx + 1))
        #pp = PdfPages('{}-track{}-figures.pdf'.format(fname, track_idx))
        cos_theta_smooth, detected_waggles, detected_waggles_median = \
            analysis.detect_waggles(track)

        #visualization.plot_features(cos_theta_smooth, detected_waggles, detected_waggles_median)
        #pp.savefig(bbox_inches='tight')

        #visualization.plot_track(track.iloc[:-1], Y=detected_waggles_median.astype(np.int32))
        #pp.savefig(bbox_inches='tight')

        waggles = analysis.extract_waggles(track, detected_waggles_median)
        print('Number of detected waggle runs: {}'.format(len(waggles)))
        if len(waggles) == 0:
            continue
        #visualization.plot_waggles(waggles)
        #pp.savefig(bbox_inches='tight')

        #visualization.plot_angle_distribution(waggles)
        #pp.savefig(bbox_inches='tight')
        #pp.close()

        #open('{}-track{}-waggles.json'.format(fname, track_idx), 'w').write(dumps(waggles))

        median_len = np.median([len(w['points']) for w in waggles])
        waggle_lens = [len(w['points']) for w in waggles]
        print(waggle_lens)

        start_time = track.t.iloc[0] / 30
        end_time = track.t.iloc[-1] / 30

        results.append((fname,
                        subfolder,
                        track_idx,
                        start_time,
                        end_time,
                        analysis.extract_most_likely_angle(waggles),
                        len(waggles),
                        median_len))

    return results

    #np.savetxt('{}-results.csv'.format(fname), np.array(results), delimiter=",",
    #           header='track_id,angle', fmt=['%d', '%s'])


@click.command()
@click.option('--track', type=click.Path(exists=True, file_okay=True,
                                         dir_okay=False, readable=True),
              required=False, help='Path to track csv/json')
@click.option('--path', type=click.Path(exists=True, file_okay=False,
                                        dir_okay=True, readable=True),
              required=False, help='Path to track csv/json directory')
def main(track=None, path=None):
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

    results = list(itertools.chain(*map(track_file, tracks)))
    print(np.array(results))
    np.savetxt('results.csv', np.array(results), delimiter=",", comments='',
               header='fname,subfolder,track_id,start_time,end_time,angle,num_waggles,median_waggle_len',
               fmt=['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s'])
