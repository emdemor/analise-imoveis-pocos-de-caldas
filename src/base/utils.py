import numpy as np
import matplotlib.pyplot as plt
from src.base.viz import ColorMix, green_red_scale, define_colormap


def plot_decision_regions(data, estimator, n=256):

    X = data[["longitude", "latitude"]]
    y = data["y"]

    color_scale = [np.quantile(y, q) for q in np.linspace(0, 1, len(green_red_scale))]

    color_mix_price = ColorMix(
        scale=color_scale,
        hex_colors=green_red_scale,
    )

    xmin, xmax = X["longitude"].min(), X["longitude"].max()
    ymin, ymax = X["latitude"].min(), X["latitude"].max()

    long = np.linspace(xmin, xmax, n)

    lat = np.linspace(ymin, ymax, n)

    long, lat = np.meshgrid(long, lat)

    X_pred = np.array([long.ravel(), lat.ravel()]).T

    pred = estimator.predict(X_pred).reshape(long.shape)

    fig, ax = plt.subplots()
    contourf_ = ax.contourf(
        long,
        lat,
        pred,
        3,
        alpha=0.4,
        cmap=define_colormap(green_red_scale),
    )
    cbar = fig.colorbar(contourf_)

    ax.contour(long, lat, pred, 3, colors="black")

    plt.xlim([xmin, xmax])

    plt.ylim([ymin, ymax])

    plt.xticks(rotation=45)

    ax.scatter(
        data["longitude"],
        data["latitude"],
        c=list(map(color_mix_price.hex_color, data["y"])),
    )
    plt.show()
