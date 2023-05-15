from typing import Tuple, List

from constants import ADVERB_EMBEDDING_SIZE, INPUT_SIZE

import numpy as np

def train_val_split(X: np.ndarray, Y: np.ndarray, train_proportion: float = 0.7) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    assert train_proportion >= 0 and train_proportion <= 1, "Invalid train proportion."
    assert len(X) == len(Y), "X and Y must have the same number of examples."

    num_examples = len(X)
    test_start_index = int(num_examples * train_proportion)

    X_train, X_val = X[:test_start_index], X[test_start_index:]
    Y_train, Y_val = Y[:test_start_index], Y[test_start_index:]

    return (X_train, X_val), (Y_train, Y_val)

def get_train_val_data(skill1_experiment_results: List[List[float]], skill2_experiment_results: List[List[float]]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    skill1_experiment_results = np.array(skill1_experiment_results)
    skill2_experiment_results = np.array(skill2_experiment_results)

    assert skill1_experiment_results.shape[1] == ADVERB_EMBEDDING_SIZE + INPUT_SIZE * 2, "Invalid skill1 experiment results shape"
    assert skill2_experiment_results.shape[1] == ADVERB_EMBEDDING_SIZE + INPUT_SIZE * 2, "Invalid skill2 experiment results shape"

    X = skill1_experiment_results[:, :(ADVERB_EMBEDDING_SIZE + INPUT_SIZE)]
    Y = skill1_experiment_results[:, (ADVERB_EMBEDDING_SIZE + INPUT_SIZE):]

    return train_val_split(X, Y)
