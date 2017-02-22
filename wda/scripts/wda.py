import numpy as np
import click
import matplotlib.pyplot as plt

from wda import analysis, features, io, visualization

@click.command()
@click.option('--track', type=click.Path(exists=True, file_okay=True,
                                         dir_okay=False, readable=True),
              required=True, help='Path to track csv/json')
def main(track):
    tracks = io.load_tracks(track)

    for track in tracks:
        print(track.head())

        cos_theta_smooth, detected_waggles, detected_waggles_median = \
            analysis.detect_waggles(track)
        visualization.plot_features(cos_theta_smooth, detected_waggles, detected_waggles_median)

        fig = visualization.plot_track(track.iloc[:-1], Y=detected_waggles_median.astype(np.int32))

        waggles = analysis.extract_waggles(track, detected_waggles_median)
        fig = visualization.plot_waggles(waggles)

        visualization.plot_angle_distribution(waggles)

    plt.show()
