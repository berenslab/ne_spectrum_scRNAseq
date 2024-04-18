import numpy as np
from utils.utils import get_path, load_human
from utils.metrics_utils import acc_kNN, corr_pdist_subsample
import os


def compute_metrics(spectrum_via="tsne"):
    n_slides = 60

    if spectrum_via == "tsne":
        min_spec_param = 0.85
        max_spec_param = 30.0
    elif spectrum_via == "cne":
        min_spec_param = -0.1
        max_spec_param = 2.0
    else:
        raise ValueError(f"Unknown spectrum_via: {spectrum_via}")
    dataset = "human-409b2"
    seeds = [0, 1, 2]
    k = 15

    root_path = get_path("data")

    x, y, d = load_human(root_path=root_path)

    sub_sample = 0

    # load embeddings
    embeddings = [np.load(os.path.join(root_path, dataset, f"{dataset}_{spectrum_via}_seed_{seed}_n_slides_{n_slides}_"
                                                           f"min_spec_{min_spec_param}_max_spec_{max_spec_param}.npy"))
                  for seed in seeds]
    embeddings = np.array(embeddings)

    # k-nn recalls
    file_name_recalls = (f"{dataset}_{spectrum_via}_n_slides_{n_slides}_"
                         f"min_spec_{min_spec_param}_max_spec_{max_spec_param}_recalls_k_{k}.npy")

    try:
        recalls = np.load(os.path.join(root_path, dataset, file_name_recalls))
    except FileNotFoundError:

        recalls = []
        for i, seed in enumerate(seeds):
            recalls_by_seed = []

            for j in range(n_slides):
                knn_recall = acc_kNN(x=x, y=embeddings[i, j], k=k)
                recalls_by_seed.append(knn_recall)
            print(f"Done with seed {seed} for k-nn recall.")
            recalls.append(recalls_by_seed)
        recalls = np.array(recalls)

        np.save(os.path.join(root_path, dataset, file_name_recalls), recalls)
    print(f"Done with knn recalls")

    # spearmann correlations
    file_name_spears= (f"{dataset}_{spectrum_via}_n_slides_{n_slides}_"
                       f"min_spec_{min_spec_param}_max_spec_{max_spec_param}_spear_sum_sample_{sub_sample}.npy")

    try:
        spears = np.load(os.path.join(root_path, dataset, file_name_spears))

    except FileNotFoundError:
        all_spear = []

        for seed in seeds:
            spear_by_seed = []

            for i in range(n_slides):
                _, spear = corr_pdist_subsample(x, embeddings[seed, i], sample_size=sub_sample, seed=seed)
                spear_by_seed.append(spear)

            all_spear.append(np.array(spear_by_seed))
            print(f"Done with seed {seed} for spearman correlation.")

        spears = np.array(all_spear)

        np.save(os.path.join(root_path, dataset, file_name_spears), spears)

    print("Done with spearman correlations")

    return recalls, spears


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spectrum_via",
        default="tsne",
        help="Whether to use cne or tsne for the spectrum.",
    )

    args = parser.parse_args()

    compute_metrics(spectrum_via=args.spectrum_via)
