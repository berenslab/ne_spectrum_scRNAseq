from vis_utils.loaders import load_dataset
import sys
sys.path.append("/gpfs01/berens/user/sdamrich/mirrored_code/miniumap")
import os
import numpy as np
import cne
import pickle

root_path = "/gpfs01/berens/user/sdamrich/data/miniumap"
fig_path = root_path+ "/figures"
dataset = "mnist"  # "mnist", "human-409b2"

x, y, sknn_graph, pca2 = load_dataset(root_path=root_path, dataset=dataset, k=15)

loss_mode = "neg"
rescale = 1.0
batch_size_init = 1024
optimizer = "sgd"
anneal_lr = True
lr_min_factor = 0.0
momentum = 0
parametric = False
clamp_low = 1e-10
seed = 0
data_on_gpu = True
log_embds = True
log_norms = False
log_kl = False
log_losses = False



############################
# get init
############################
file_name_init = os.path.join(root_path,
                                 dataset,
                                 f"cne_{loss_mode}_n_noise_{5}_n_epochs_{250}_init_pca_rescale_{rescale}_bs_{batch_size_init}"
                                 f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                                 )

try:
    with open(file_name_init, "rb") as file:
        embedder_init_cne = pickle.load(file)
except FileNotFoundError:
    init = pca2 / np.std(pca2[:, 0]) * rescale
    print(f"Starting with {file_name_init}")
    logger = cne.callbacks.Logger(log_embds=log_embds,
                                  log_norms=log_norms,
                                  log_kl=log_kl,
                                  log_losses=log_losses,
                                  graph=sknn_graph,
                                  n=len(x) if parametric else None)
    embedder_init = cne.CNE(loss_mode=loss_mode,
                            parametric=parametric,
                            negative_samples=5,
                            n_epochs=250,
                            batch_size=batch_size_init,
                            data_on_gpu=data_on_gpu,
                            print_freq_epoch=50,
                            callback=logger,
                            optimizer=optimizer,
                            momentum=momentum,
                            save_freq=1,
                            anneal_lr=anneal_lr,
                            lr_min_factor=lr_min_factor,
                            clamp_low=clamp_low,
                            seed=seed,
                            loss_aggregation="sum",
                            force_resample=True
                            )
    embedder_init.fit(x, init=init, graph=sknn_graph)
    embedder_init_cne = embedder_init.cne

    with open(file_name_init, "wb") as file:
        pickle.dump(embedder_init_cne, file, pickle.HIGHEST_PROTOCOL)
    print(f"Done with {file_name_init}")
init = embedder_init_cne.callback.embds[-1]



###############################################
# comput embedding
###############################################


n_noise = 5
n_epochs = 2950
batch_size = 2**15
init_str = "EE"
log_losses = True
log_norms = True
anneal_lr = True
clamp_low = 1e-4


for s in np.linspace(-1, 2, 60):

    file_name = os.path.join(root_path,
                             dataset,
                                     f"cne_{loss_mode}_n_noise_{n_noise}_s_{s}_n_epochs_{n_epochs}_init_{init_str}_bs_{batch_size}"
                                     f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                                     )

    print(f"Starting with {file_name}")
    try:
        with open(file_name, "rb") as file:
            embedder_cne = pickle.load(file)
    except FileNotFoundError:
        logger = cne.callbacks.Logger(log_embds=log_embds,
                                      log_norms=log_norms,
                                      log_kl=log_kl,
                                      log_losses=log_losses,
                                      graph=sknn_graph,
                                      n=len(x) if parametric else None)
        embedder = cne.CNE(loss_mode=loss_mode,
                           parametric=parametric,
                           negative_samples=n_noise,
                           n_epochs=n_epochs,
                           batch_size=batch_size,
                           data_on_gpu=data_on_gpu,
                           print_freq_epoch=100,
                           callback=logger,
                           optimizer=optimizer,
                           momentum=momentum,
                           save_freq=1,
                           anneal_lr=anneal_lr,
                           s=s,
                           lr_min_factor=lr_min_factor,
                           clamp_low=clamp_low,
                           seed=seed,
                           loss_aggregation="sum",
                           force_resample=True,
                           early_exaggeration=False
                           )
        embedder.fit(x, init=init, graph=sknn_graph)
        embedder_cne = embedder.cne

        with open(file_name, "wb") as file:
            pickle.dump(embedder_cne, file, pickle.HIGHEST_PROTOCOL)

    print(f"Done with {file_name}")



