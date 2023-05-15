from typing import List, Tuple, Union

from constants import ADVERB_EMBEDDING_SIZE, INPUT_SIZE
from preprocess import get_train_val_data

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SmallNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer1(x)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        return self.layer2(out)
    
def np_array_to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))

def tensor_to_np_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().numpy()
    
def get_val_loss(net: Union[Net, SmallNet], X_val: np.ndarray, Y_val: np.ndarray, loss_func: nn.Module) -> np.ndarray:
    predictions = net.forward(np_array_to_tensor(X_val))
    loss = loss_func(predictions, np_array_to_tensor(Y_val))

    return tensor_to_np_array(loss)
    
def train_net(net: Union[Net, SmallNet], X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, optimizer: optim, loss_func: nn.Module, num_epochs: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epoch_train_losses = []
    epoch_val_losses = []

    train_indices = np.arange(X_train.shape[0])

    for _ in range(num_epochs):
        np.random.shuffle(train_indices)

        X_train_epoch = np_array_to_tensor(X_train[train_indices])
        Y_train_epoch = np_array_to_tensor(Y_train[train_indices])

        optimizer.zero_grad()
        predictions = net.forward(X_train_epoch)
        loss = loss_func(predictions, Y_train_epoch)
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(tensor_to_np_array(loss))
        epoch_val_losses.append(get_val_loss(net, X_val, Y_val, loss_func))

    return np.array(epoch_train_losses), np.array(epoch_val_losses)
    
def transfer_layer1(old_net: Net, new_net: Net):
    state_dict = {}

    for name, param in old_net.named_parameters():
        if "layer1" in name:
            state_dict[name] = param

    new_net.load_state_dict(state_dict, strict=False)

def freeze_layer1(net: Net):
    for name, param in net.named_parameters():
        if "layer1" in name:
            param.requires_grad = False

def learn_net_adverb_skill_groundings(skill1_experiment_results: List[List[float]], skill2_experiment_results: List[List[float]], skill1_name: str, skill2_name: str):
    (X_train, X_val), (Y_train, Y_val) = get_train_val_data(skill1_experiment_results, skill2_experiment_results)

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
