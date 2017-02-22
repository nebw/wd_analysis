import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from wda.analysis import determine_threshold


def plot_track(track, Y=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = ListedColormap(sns.color_palette("muted", 2))

    ax.scatter(*track[['x', 'y', 't']].as_matrix().T,
               s=10, alpha=.5, c=Y, cmap=cmap)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')

    fig.suptitle('Dance trajectory')

    return fig


def plot_features(cos_theta_smooth, detected_waggles, detected_waggles_median):
    fig, axes = plt.subplots(3, 1, figsize=(8, 18))

    threshold = determine_threshold(cos_theta_smooth)

    axes[0].plot(cos_theta_smooth)
    axes[0].axhline(threshold)
    axes[0].set_title('Smoothed angle derivative from raw trajectory')
    axes[1].plot(detected_waggles)
    axes[1].set_title('Otsu thresholding')
    axes[2].plot(detected_waggles_median)
    axes[2].set_title('Otsu thresholding (median-filtered)')

    fig.suptitle('Waggle detection feature')

    return fig


def rad_to_deg(theta):
    theta_deg = theta / np.pi * 180
    if theta_deg < 0:
        theta_deg = 360 + theta_deg
    return theta_deg


def plot_waggle(waggle, axes):
    theta = waggle['best_theta']
    theta_deg = rad_to_deg(theta)

    dy, dx = np.sin(theta), np.cos(theta)

    axes[0].scatter(*waggle['points'].T, c=plt.cm.viridis.colors[128])
    mx, my = np.mean(axes[0].get_xlim()), np.mean(axes[0].get_ylim())
    axes[0].set_title(r'$\theta = {:.2f}°$'.format(theta_deg))
    axes[0].quiver(mx, my, dx, dy, scale=3., width=0.025, alpha=.75,
                   color=plt.cm.viridis.colors[180])
    axes[0].quiver(mx, my, -dx, -dy, scale=3., width=0.025, alpha=.75,
                   color=plt.cm.viridis.colors[180])
    axes[0].axis('equal')

    axes[1].plot(waggle['theta_bins'], waggle['smoothed_hist'])
    axes[1].set_title('Angle distribution')


def plot_waggles(waggles):
    fig, axes = plt.subplots(len(waggles), 2, figsize=(8, 4 * len(waggles)))

    for idx, waggle in enumerate(waggles):
        plot_waggle(waggle, axes[idx])

    fig.suptitle('Individual waggle run decodings')

    return fig


def plot_angle_distribution(waggles):
    fig = plt.figure(figsize=(6, 6))

    theta_bins = waggles[0]['theta_bins']
    histograms = [w['smoothed_hist'] for w in waggles]

    N = len(theta_bins)
    bottom = 0
    max_height = 4

    median_thetas = np.median(histograms, axis=0)
    median_thetas /= np.max(median_thetas)

    theta = theta_bins
    radii = max_height * median_thetas
    width = (2*np.pi) / N + .1

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, width=width, bottom=bottom)

    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.viridis_r(r / max_height))
        bar.set_alpha(0.5)

    ax.set_rmax(max_height)

    most_likely_deg = rad_to_deg(theta_bins[np.argmax(median_thetas)])

    ax.set_title('Most likely angle: {:.1f}°'.format(most_likely_deg))

    ax.axvline(theta_bins[np.argmax(median_thetas)])
    ticks = set(ax.get_xticks())
    ticks.add((theta_bins[np.argmax(median_thetas)] + 2 * np.pi) % (2 * np.pi))
    ax.set_xticks(list(ticks))

    fig.suptitle('Dance decoding')

    return fig
