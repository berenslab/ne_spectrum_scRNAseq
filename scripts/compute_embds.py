from tsne_slider.spectra import TSNESpectrum, CNESpectrum
from ne_spectrum_scRNAseq.utils.utils import get_path
from vis_utils.loaders import load_dataset
import os


root_path = get_path("data")

dataset = "human-409b2"
spectrum_via = "cne"

min_spec_param = -0.1  # 0.85
max_spec_param = 2.0  # 30
n_slides = 60

seeds = [0, 1, 2]


x, y, _, _, d = load_dataset(root_path=root_path, dataset=dataset, k=15)

for seed in seeds:
    file_name = (f"{dataset}_{spectrum_via}_seed_{seed}_n_slides_{n_slides}"
                 f"_min_spec_{min_spec_param}_max_spec_{max_spec_param}")

    if spectrum_via == "tsne":
        slider = TSNESpectrum(num_slides=n_slides,
                              min_exaggeration=min_spec_param,
                              max_exaggeration=max_spec_param,
                              seed=seed)
    elif spectrum_via == "cne":
        slider = CNESpectrum(num_slides=n_slides,
                             min_spec_param=min_spec_param,
                             max_spec_param=max_spec_param,
                             seed=seed)
    else:
        raise ValueError(f"Unknown spectrum_via: {spectrum_via}")

    slider.fit(x)

    slider.save_embeddings(os.path.join(root_path, dataset, file_name+".npy"))

    print(f"Done with seed {seed}")


