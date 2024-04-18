from pkg_resources import resource_stream
import numpy as np
from ne_spectrum import TSNESpectrum, CNESpectrum
import os
import requests
import pandas as pd
import pickle
from .treutlein_preprocess import preprocess as treut_preprocess


def get_path(path_type):
    #  util for reading in paths from file
    with resource_stream(__name__, "my_paths") as file:
        lines = file.readlines()
    lines = [line.decode('ascii').split(" ") for line in lines]
    path_dict = {line[0]: " ".join(line[1:]).strip("\n") for line in lines}

    if path_type == "data":
        try:
            if path_dict["data_path"].startswith("../"):
                dir = "/".join(__file__.split("/")[:-2])
                return dir + "/" + path_dict["data_path"][3:]
            else:
                return path_dict["data_path"]
        except KeyError:
            print("There is no path 'data_path'.")

    elif path_type == "figures":
        try:
            if path_dict["fig_path"].startswith("../"):
                dir = "/".join(__file__.split("/")[:-2])
                return dir + "/" + path_dict["fig_path"][3:]
            else:
                return path_dict["fig_path"]
        except KeyError:
            print("There is no path 'fig_path'.")
    elif path_type == "style":
        try:
            dir = "/".join(__file__.split("/")[:-1])  # style file lies in the same directory as this file
            return dir+"/"+path_dict["style_path"]
        except KeyError:
            print("There is no path 'style_path'.")


def get_computed_spec_params(spectrum_via, min_spec_param=None, max_spec_param=None):

    # use only those spec param bounds that are provided and resort to the defaults in of ne_spectum instead.
    kwargs = {}
    if min_spec_param is not None:
        kwargs["min_spec_param"] = min_spec_param
    if max_spec_param is not None:
        kwargs["max_spec_param"] = max_spec_param

    # create the spectrum object to obtain the intermediate spec params
    if spectrum_via == "tsne":
        spectrum = TSNESpectrum(**kwargs)
    elif spectrum_via == "cne":
        spectrum = CNESpectrum(**kwargs)
    else:
        raise ValueError("spectrum_via must be either 'tsne' or 'cne'")

    l = spectrum.kwarg_list
    computed_spec_params = np.array([l[i][spectrum.spectrum_param_name] for i in range(len(l))])

    return computed_spec_params


def get_close_spec_params(selected_spec_params, spectrum_via, min_spec_param=None, max_spec_param=None):

    computed_spec_params = get_computed_spec_params(spectrum_via=spectrum_via,
                                                    min_spec_param=min_spec_param,
                                                    max_spec_param=max_spec_param)

    dists = np.abs(computed_spec_params[:, None] - selected_spec_params[None])
    idx = np.argmin(dists, axis=0)
    close_spec_params = computed_spec_params[idx]
    return close_spec_params, idx, computed_spec_params



def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully from {url}.")
    else:
        print(f"Failed to download file from {url}. Status code: {response.status_code}")

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def load_human(root_path):
    root_path = os.path.join(root_path, "human-409b2")
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    try:
        x = np.load(os.path.join(root_path, "human-409b2.data.npy"))
        y = np.load(os.path.join(root_path, "human-409b2.labels.npy"))
        d = load_dict(os.path.join(root_path, "human-409b2.pkl"))
    except FileNotFoundError:

        metafile = os.path.join(root_path,
                                #"unzipped_files",
                                "metadata_human_cells.tsv")
        countfile = os.path.join(root_path,
                                 #"unzipped_files",
                                 "human_cell_counts_consensus.mtx")
        urls = []
        if not os.path.exists(metafile):
            urls.append("https://www.ebi.ac.uk/biostudies/files/E-MTAB-7552/metadata_human_cells.tsv")
        if not os.path.exists(countfile):
            urls.append("https://www.ebi.ac.uk/biostudies/files/E-MTAB-7552/human_cell_counts_consensus.mtx")

        if len(urls) > 0:
            print("Downloading data")
        for url in urls:
            filename = os.path.join(root_path, url.split("/")[-1])
            download_file(url, filename)

        print("Preprocessing data")
        line = "409b2"
        X, stage = treut_preprocess(metafile, countfile, line)

        outputfile = "human-409b2"

        np.save(os.path.join(root_path, outputfile + ".data.npy"), X)
        np.save(os.path.join(root_path, outputfile + ".labels.npy"), stage)
        x = X
        y = stage

        print("Done")

        # meta data
        d = {"label_colors": {
            "iPSCs": "navy",
            "EB": "royalblue",
            "Neuroectoderm": "skyblue",
            "Neuroepithelium": "lightgreen",
            "Organoid-1M": "gold",
            "Organoid-2M": "tomato",
            "Organoid-3M": "firebrick",
            "Organoid-4M": "maroon",
        }, "time_colors": {
            "  0 days": "navy",
            "  4 days": "royalblue",
            "10 days": "skyblue",
            "15 days": "lightgreen",
            "  1 month": "gold",
            "  2 months": "tomato",
            "  3 months": "firebrick",
            "  4 months": "maroon",
        }}

        # cluster assignments
        meta = pd.read_csv(metafile, sep="\t")
        mask = (meta["Line"] == "409b2")* meta["in_FullLineage"]

        d["clusters"] = list(meta[mask]["cl_FullLineage"])

        d["color_to_time"] = {v: k for k, v in d["time_colors"].items()}

        save_dict(d, os.path.join(root_path, f"{outputfile}.pkl"))

    d["clusters"] = np.array(d["clusters"])
    return x, y, d


def load_metrics(spectrum_via, root_path, dataset, n_slides, sub_sample=0, min_spec_param=None, max_spec_param=None):
    k = 15

    computed_spec_params = get_computed_spec_params(spectrum_via, min_spec_param, max_spec_param)

    min_spec_param = np.round(computed_spec_params.min(), 2)
    max_spec_param = np.round(computed_spec_params.max(), 2)

    file_name_recalls = f"human-409b2_{spectrum_via}_n_slides_{n_slides}_min_spec_{min_spec_param}_max_spec_{max_spec_param}_recalls_k_{k}.npy"

    recalls = np.load(os.path.join(root_path, dataset, file_name_recalls))

    file_name_spears = f"human-409b2_{spectrum_via}_n_slides_{n_slides}_min_spec_{min_spec_param}_max_spec_{max_spec_param}_spear_sum_sample_{sub_sample}.npy"

    spears = np.load(os.path.join(root_path, dataset, file_name_spears))

    return recalls, spears