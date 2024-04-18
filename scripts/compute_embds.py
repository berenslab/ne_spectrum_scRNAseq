from ne_spectrum import TSNESpectrum, CNESpectrum
from utils.utils import get_path, load_human
import os


def compute_embds(spectrum_via="tsne"):

    if spectrum_via == "tsne":
        min_spec_param = 0.85
        max_spec_param = 30.0
    elif spectrum_via == "cne":
        min_spec_param = -0.1
        max_spec_param = 2.0
    else:
        raise ValueError(f"Unknown spectrum_via: {spectrum_via}")

    # turn spec params to float
    min_spec_param = float(min_spec_param)
    max_spec_param = float(max_spec_param)

    root_path = get_path("data")
    dataset = "human-409b2"
    n_slides = 60
    seeds = [0, 1, 2]

    x, y, d = load_human(root_path=root_path)

    for seed in seeds:
        file_name = (f"{dataset}_{spectrum_via}_seed_{seed}_n_slides_{n_slides}"
                     f"_min_spec_{min_spec_param}_max_spec_{max_spec_param}")

        if spectrum_via == "tsne":
            spectrum = TSNESpectrum(num_slides=n_slides,
                                    min_exaggeration=min_spec_param,
                                    max_exaggeration=max_spec_param,
                                    seed=seed)
        elif spectrum_via == "cne":
            spectrum = CNESpectrum(num_slides=n_slides,
                                   min_spec_param=min_spec_param,
                                   max_spec_param=max_spec_param,
                                   seed=seed)
        else:
            raise ValueError(f"Unknown spectrum_via: {spectrum_via}")

        try:
            spectrum.load_embeddings(os.path.join(root_path, dataset, file_name+".npy"))
            print(f"Loaded embeddings for seed {seed}")
        except FileNotFoundError:
            print(f"Computing embeddings for seed {seed}")
            spectrum.fit(x)
            spectrum.save_embeddings(os.path.join(root_path, dataset, file_name+".npy"))
            print(f"Done with seed {seed}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum_via",
                        type=str,
                        default="tsne",
                        help="Whether to use cne or tsne for the spectrum.")

    args = parser.parse_args()
    compute_embds(spectrum_via=args.spectrum_via)

