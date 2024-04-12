from pkg_resources import resource_stream
import numpy as np
from tsne_slider import TSNESpectrum, CNESpectrum

def get_path(path_type):
    #  util for reading in paths from file
    with resource_stream(__name__, "my_paths") as file:
        lines = file.readlines()

    lines = [line.decode('ascii').split(" ") for line in lines]
    path_dict = {line[0]: " ".join(line[1:]).strip("\n") for line in lines}

    if path_type == "data":
        try:
            return path_dict["data_path"]
        except KeyError:
            print("There is no path 'data_path'.")

    elif path_type == "figures":
        try:
            return path_dict["fig_path"]
        except KeyError:
            print("There is no path 'fig_path'.")
    elif path_type == "style":
        try:
            return path_dict["style_path"]
        except KeyError:
            print("There is no path 'style_path'.")


def get_close_spec_params(selected_spec_params, spectrum_via):
    if spectrum_via == "tsne":
        spectrum = TSNESpectrum()
    elif spectrum_via == "cne":
        spectrum = CNESpectrum()
    else:
        raise ValueError("spectrum_via must be either 'tsne' or 'cne'")
    l = spectrum.kwarg_list
    computed_spec_params = np.array([l[i][spectrum.spectrum_param_name] for i in range(len(l))])
    dists = np.abs(computed_spec_params[:, None] - selected_spec_params[None])
    idx = np.argmin(dists, axis=0)
    close_spec_params = computed_spec_params[idx]
    return close_spec_params, idx, computed_spec_params
