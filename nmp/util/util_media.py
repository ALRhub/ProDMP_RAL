"""
    Utilities for generating media stuff
"""

from typing import List
from typing import Literal
from typing import Union

import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt

import nmp.util as util


def savefig(figs: Union[plt.Figure, List[plt.Figure]], media_name,
            fmt=Literal['pdf', 'png', 'jpeg'], dpi=200, overwrite=False):
    """

    Args:
        figs: figure object or a list of figures
        media_name: name of the media
        fmt: format of the figures
        dpi: resolution
        overwrite: if overwrite when old exists

    Returns:
        None

    """
    path = util.get_media_dir(media_name)
    util.mkdir(path, overwrite=overwrite)

    figs = util.make_iterable(figs)

    for i, fig in enumerate(figs):
        fig_path = util.join_path(path, str(i) + '.' + fmt)
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")


def save_subfig(fig: plt.Figure, axes: np.ndarray,
                ax_coordinate: [List[int], List[List[int]]],
                media_name: str, fmt=Literal['pdf', 'png', 'jpeg'], dpi=200,
                overwrite=False,
                x_scale=0.2, y_scale=0.2, x_offset=-0.1, y_offset=-0.1):
    """
    Save subplots as individual plots
    Args:
        fig: figure object
        axes: axes in np array
        ax_coordinate: which subplots you want to save
        media_name: name of the save dir
        fmt: format to save as
        dpi: resolution
        overwrite: overwrite the dir if it exists already
        x_scale: scale of the bounding box in x direction
        y_scale: scale of the bounding box in y direction
        x_offset: offset of the bounding box in x direction
        y_offset: offset of the bounding box in y direction

    Returns:
        None
    """
    if isinstance(ax_coordinate[0], int):
        ax_coordinate = [ax_coordinate, ]
    fig.tight_layout()
    path = util.get_media_dir(media_name)
    util.mkdir(path, overwrite=overwrite)

    for coord in ax_coordinate:
        ax = axes[coord[0], coord[1]]
        ext = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        subfig_name = str(coord).replace(", ", "_") + '.' + fmt
        fig_path = util.join_path(path, subfig_name)
        bbox_inches = ext.expanded(1 + x_scale,
                                   1 + y_scale).translated(x_offset, y_offset)
        fig.savefig(fig_path, dpi=dpi, bbox_inches=bbox_inches)


def from_figures_to_video(figure_list: [], video_name: str,
                          interval: int = 2000, overwrite=False) -> str:
    """
    Generate and save a video given a list of figures
    Args:
        figure_list: list of matplotlib figure objects
        video_name: name of video
        interval: interval between two figures in [ms]
        overwrite: if overwrite when old exists
    Returns:
        path to the saved video
    """
    figure, ax = plt.subplots()
    figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    frames = []

    video_path = util.get_media_dir(video_name)
    util.mkdir(video_path, overwrite)
    for i, fig in enumerate(figure_list):
        fig.savefig(util.join_path(video_path, "{}.png".format(i)), dpi=300,
                    bbox_inches="tight")

    for j in range(len(figure_list)):
        image = plt.imread(util.join_path(video_path, "{}.png".format(j)))
        img = plt.imshow(image, animated=True)
        plt.axis('off')
        plt.gca().set_axis_off()

        frames.append([img])

    ani = animation.ArtistAnimation(figure, frames, interval=interval,
                                    blit=True,
                                    repeat=False)
    save_path = util.join_path(video_path, video_name + '.mp4')
    ani.save(save_path, dpi=300)

    return save_path


def fill_between(x: Union[np.ndarray, torch.Tensor],
                 y_mean: Union[np.ndarray, torch.Tensor],
                 y_std: Union[np.ndarray, torch.Tensor],
                 axis=None, std_scale: int = 2, draw_mean: bool = False,
                 alpha=0.2, color='gray'):
    """
    Utilities to draw std plot
    Args:
        x: x value
        y_mean: y mean value
        y_std: standard deviation of y
        axis: figure axis to draw
        std_scale: filling range of [-scale * std, scale * std]
        draw_mean: plot mean curve as well
        alpha: transparency of std plot
        color: color to fill

    Returns:
        None
    """
    x, y_mean, y_std = util.to_nps(x, y_mean, y_std)
    if axis is None:
        axis = plt.gca()
    if draw_mean:
        axis.plot(x, y_mean)
    axis.fill_between(x=x,
                      y1=y_mean - std_scale * y_std,
                      y2=y_mean + std_scale * y_std,
                      alpha=alpha, color=color)
