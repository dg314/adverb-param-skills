from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    
def train_val_split(X: np.ndarray, Y: np.ndarray, train_proportion: float = 0.7) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    assert train_proportion >= 0 and train_proportion <= 1, "Invalid train proportion."
    assert len(X) == len(Y), "X and Y must have the same number of examples."

    num_examples = len(X)
    test_start_index = int(num_examples * train_proportion)

    X_train, X_val = X[:test_start_index], X[test_start_index:]
    Y_train, Y_val = Y[:test_start_index], Y[test_start_index:]

    return (X_train, X_val), (Y_train, Y_val)
    
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
