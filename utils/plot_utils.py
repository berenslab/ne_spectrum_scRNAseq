import numpy as np
from scipy.interpolate import splprep, splev



def set_bounds(ax, embd, q):
    y_min, y_max = np.quantile(embd[:, 1], q), np.quantile(embd[:, 1], 1 - q)
    x_min, x_max = np.quantile(embd[:, 0], q), np.quantile(embd[:, 0], 1 - q)
    x_mid = (x_min + x_max) / 2

    span = x_max - x_min

    ax.set_xlim(x_mid - span / 2 * 1.1, x_mid + span / 2 * 1.1)

    y_mid = (embd[:, 1].min() + embd[:, 1].max()) / 2  # (y_min+y_max)/2
    ax.set_ylim(y_mid - span * 1.1 / 2, y_mid + span * 1.1 / 2)
    # ax.set_ylim(x_min, y_max)
    return ax


# %%
def get_time_midpoints(embd, y, d):
    time_x, time_y = [], []
    labels = []
    for label in d["label_colors"].keys():
        mask = y == label

        if mask.sum() > 0:
            time_x.append(embd[mask][:, 0].mean())
            time_y.append(embd[mask][:, 1].mean())
            labels.append(label)

    time_x = np.array(time_x)
    time_y = np.array(time_y)
    return np.stack([time_x, time_y], axis=-1), labels


# %%
def get_time_trajectory(embd, y, d, s=1):
    # get centroids of each time step
    time_x, time_y = get_time_midpoints(embd, y, d)[0].T
    # fit x and y separately as there are not monotonic
    tck, u = splprep([time_x, time_y], s=s, k=3, per=False)

    t = np.linspace(0, 1., 100)

    # evaluate both parametizations; returns list
    curve = splev(t, tck)
    return curve


# %%
def draw_time_trajectory(ax, embd, y, d, s=1, alpha=1):
    curve = get_time_trajectory(embd, y, d, s=s)

    x_vals = curve[0][2::3]
    y_vals = curve[1][2::3]

    dx = np.diff(x_vals)
    dy = np.diff(y_vals)

    norms = np.sqrt(dx ** 2 + dy ** 2)
    dx = dx / norms * 0.5
    dy = dy / norms * 0.5

    ax.plot(curve[0], curve[1], color="black", linestyle="--", alpha=alpha)
    ax.quiver(x_vals[:-1][::7], y_vals[:-1][::7], dx[::7], dy[::7], scale_units='xy', angles='xy', scale=0.75,
              color='k', width=0.5, minlength=0)

    return ax


def turn_label_to_days(label, d):
    time = d["color_to_time"][d["label_colors"][label]]
    time = time.strip()
    if "day" in time:
        return int(time.split(" ")[0])
    else:
        months = time.split(" ")[0]
        return int(months) * 30