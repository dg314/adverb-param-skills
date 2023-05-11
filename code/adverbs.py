from typing import List

from constants import ADVERB_EMBEDDING_SIZE, INPUT_SIZE
from net import Net, SmallNet, train_net, train_val_split, transfer_layer1, freeze_layer1

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def learn_adverb_skill_groundings(skill1_experiment_results: List[List[float]], skill2_experiment_results: List[List[float]], skill1_name: str, skill2_name: str):
    skill1_experiment_results = np.array(skill1_experiment_results)
    skill2_experiment_results = np.array(skill2_experiment_results)

    assert skill1_experiment_results.shape[1] == ADVERB_EMBEDDING_SIZE + INPUT_SIZE * 2, "Invalid skill1 experiment results shape"
    assert skill2_experiment_results.shape[1] == ADVERB_EMBEDDING_SIZE + INPUT_SIZE * 2, "Invalid skill2 experiment results shape"

    X = skill1_experiment_results[:, :(ADVERB_EMBEDDING_SIZE + INPUT_SIZE)]
    Y = skill1_experiment_results[:, (ADVERB_EMBEDDING_SIZE + INPUT_SIZE):]

    (X_train, X_val), (Y_train, Y_val) = train_val_split(X, Y)

    loss_func = nn.MSELoss()

    skill1_net = Net(ADVERB_EMBEDDING_SIZE + INPUT_SIZE, 10, INPUT_SIZE)
    skill1_optimizer = optim.SGD(skill1_net.parameters(), lr=0.03)

    _, skill1_epoch_val_losses = train_net(skill1_net, X_train, Y_train, X_val, Y_val, skill1_optimizer, loss_func)

    skill2_random_layer1_net = Net(ADVERB_EMBEDDING_SIZE + INPUT_SIZE, 10, INPUT_SIZE)
    freeze_layer1(skill2_random_layer1_net)

    skill2_random_layer1_optimizer = optim.SGD(skill2_random_layer1_net.parameters(), lr=0.03)

    _, skill2_random_layer1_epoch_val_losses = train_net(skill2_random_layer1_net, X_train, Y_train, X_val, Y_val, skill2_random_layer1_optimizer, loss_func)

    skill2_single_layer_net = SmallNet(ADVERB_EMBEDDING_SIZE + INPUT_SIZE, INPUT_SIZE)

    skill2_single_layer_optimizer = optim.SGD(skill2_single_layer_net.parameters(), lr=0.03)

    _, skill2_single_layer_epoch_val_losses = train_net(skill2_single_layer_net, X_train, Y_train, X_val, Y_val, skill2_single_layer_optimizer, loss_func)

    skill2_transfer_net = Net(ADVERB_EMBEDDING_SIZE + INPUT_SIZE, 10, INPUT_SIZE)
    transfer_layer1(skill1_net, skill2_transfer_net)
    freeze_layer1(skill2_transfer_net)

    skill2_transfer_optimizer = optim.SGD(skill2_transfer_net.parameters(), lr=0.03)

    _, skill2_transfer_epoch_val_losses = train_net(skill2_transfer_net, X_train, Y_train, X_val, Y_val, skill2_transfer_optimizer, loss_func)

    skill1_snake_case = skill1_name.replace(" ", "_").lower()
    skill2_snake_case = skill2_name.replace(" ", "_").lower()

    plt.figure(figsize=(10, 6))
    plt.plot(skill1_epoch_val_losses, label=f"{skill1_name} Validation Loss")
    plt.plot(skill2_random_layer1_epoch_val_losses, label=f"{skill2_name} Validation Loss w/ Randomized Layer 1")
    plt.plot(skill2_single_layer_epoch_val_losses, label=f"{skill2_name} Validation Loss w/ Single Layer")
    plt.plot(skill2_transfer_epoch_val_losses, label=f"{skill2_name} Validation Loss w/ Transfer")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Adverb Skill Grounding w/ 100 Experiment Trials ({skill1_name} -> {skill2_name})")
    plt.savefig(f"adverb_generalization_{skill1_snake_case}_to_{skill2_snake_case}_100_trials")
    plt.show()

    skill2_no_transfer_final_epoch_val_losses = []
    skill2_transfer_final_epoch_val_losses = []
    shots_list = np.arange(5, len(X_train), 5)

    for shots in shots_list:
        X_train_subset = X_train[:shots]
        Y_train_subset = Y_train[:shots]
        
        skill2_no_transfer_net = Net(ADVERB_EMBEDDING_SIZE + INPUT_SIZE, 10, INPUT_SIZE)
        skill2_no_transfer_optimizer = optim.SGD(skill2_no_transfer_net.parameters(), lr=0.03)

        _, skill2_no_transfer_epoch_val_losses = train_net(skill2_no_transfer_net, X_train_subset, Y_train_subset, X_val, Y_val, skill2_no_transfer_optimizer, loss_func, num_epochs=10)
        skill2_no_transfer_final_epoch_val_losses.append(skill2_no_transfer_epoch_val_losses[-1])

        skill2_transfer_net = Net(ADVERB_EMBEDDING_SIZE + INPUT_SIZE, 10, INPUT_SIZE)
        transfer_layer1(skill1_net, skill2_transfer_net)

        skill2_transfer_optimizer = optim.SGD(skill2_transfer_net.parameters(), lr=0.03)

        _, skill2_transfer_epoch_val_losses = train_net(skill2_transfer_net, X_train_subset, Y_train_subset, X_val, Y_val, skill2_transfer_optimizer, loss_func, num_epochs=10)
        skill2_transfer_final_epoch_val_losses.append(skill2_transfer_epoch_val_losses[-1])

    plt.plot(shots_list, skill2_no_transfer_final_epoch_val_losses, label=f"{skill2_name} Final Validation Loss w/o Transfer")
    plt.plot(shots_list, skill2_transfer_final_epoch_val_losses, label=f"{skill2_name} Final Validation Loss w/ Transfer")
    plt.legend()
    plt.xlabel(f"Shots on {skill2_name} Environment")
    plt.ylabel("Loss")
    plt.title(f"Few-Shot Learning of Adverb Skill Grounding ({skill1_name} -> {skill2_name})")
    plt.savefig(f"adverb_generalization_{skill1_snake_case}_to_{skill2_snake_case}_few_shots")
    plt.show() 
