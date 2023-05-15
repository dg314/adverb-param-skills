from typing import List, Callable, Union, Tuple

from gpt import gpt_convert_env_result_to_adverb_input
from launcher_env import LauncherEnv
from word2vec import download_keyed_vectors, create_word_to_magnitude_predictor, create_word_to_slower_faster_predictor, create_word_to_lower_higher_predictor, create_word_to_closer_farther_predictor

import pickle

def get_experiment_pickle_path(experiment_name: str, env_type: str) -> str:
    experiment_name_snake_case = experiment_name.replace(" ", "_").lower()

    return f"{env_type}_{experiment_name_snake_case}_experiment_results.pickle"

def load_experiment_results(experiment_name: str, env_type: str) -> List:
    pickle_path = get_experiment_pickle_path(experiment_name, env_type)

    try:
        with open(pickle_path, "rb") as pickle_file:
            experiment_results = pickle.load(pickle_file)
    except:
        experiment_results = []

    return experiment_results

def save_experiment_results(experiment_name: str, env_type: str, new_human_experiment_results: List):
    pickle_path = get_experiment_pickle_path(experiment_name, env_type)

    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(new_human_experiment_results, pickle_file)

def run_experiment(experiment_name: str, env_type: str, run_trial: Callable[[float, float, float, float, float, float], Union[List[Tuple[str, str]], None, str]], visualize: bool, max_trials: int = 100):
    keyed_vectors = download_keyed_vectors()

    word_to_magnitude_predictor = create_word_to_magnitude_predictor(keyed_vectors)
    word_to_slower_faster_predictor = create_word_to_slower_faster_predictor(keyed_vectors)
    word_to_lower_higher_predictor = create_word_to_lower_higher_predictor(keyed_vectors)
    word_to_closer_farther_predictor = create_word_to_closer_farther_predictor(keyed_vectors)

    experiment_results = load_experiment_results(experiment_name, env_type)

    for _ in range(max_trials):
        env = LauncherEnv(env_type)

        ang_accel_A, release_time_A = env.sample_simulation_input()
        disp_x_A, vel_x_A, max_height_A = env.simulate(ang_accel_A, release_time_A, "A", visualize=visualize)

        ang_accel_B, release_time_B = env.sample_simulation_input()
        disp_x_B, vel_x_B, max_height_B = env.simulate(ang_accel_B, release_time_B, "B", visualize=visualize)

        adverb_input = run_trial(disp_x_A, vel_x_A, max_height_A, disp_x_B, vel_x_B, max_height_B)

        if adverb_input == "quit":
            break

        if isinstance(adverb_input, str):
            adverb_input = [pair.strip().split(" ") for pair in adverb_input.split(",")]

        if adverb_input is None or not all([len(pair) == 2 for pair in adverb_input]):
            print(f"\033[91mTrial aborted due to incorrectly formatted output.\n\nadverb_input={adverb_input}\n\n\033[0m")
            continue
        else:
            print(f"\033[92mTrial completed with correctly formatted output.\033[0m")
        
        slower_faster_embeddings = []
        lower_higher_embeddings = []
        closer_farther_embeddings = []

        for pair in adverb_input:
            magnitude = word_to_magnitude_predictor(pair[0])
            slower_faster_embedding = word_to_slower_faster_predictor(pair[1]) * magnitude
            lower_higher_embedding = word_to_lower_higher_predictor(pair[1]) * magnitude
            closer_farther_embedding = word_to_closer_farther_predictor(pair[1]) * magnitude

            slower_faster_embeddings.append(slower_faster_embedding)
            lower_higher_embeddings.append(lower_higher_embedding)
            closer_farther_embeddings.append(closer_farther_embedding)

        max_slower_faster_embedding = max(slower_faster_embeddings)
        min_slower_faster_embedding = min(slower_faster_embeddings)
        slower_faster_embedding = (
            max_slower_faster_embedding
            if abs(max_slower_faster_embedding) > abs(min_slower_faster_embedding)
            else min_slower_faster_embedding
        )

        max_lower_higher_embedding = max(lower_higher_embeddings)
        min_lower_higher_embedding = min(lower_higher_embeddings)
        lower_higher_embedding = (
            max_lower_higher_embedding
            if abs(max_lower_higher_embedding) > abs(min_lower_higher_embedding)
            else min_lower_higher_embedding
        )

        max_closer_farther_embedding = max(closer_farther_embeddings)
        min_closer_farther_embedding = min(closer_farther_embeddings)
        closer_farther_embedding = (
            max_closer_farther_embedding
            if abs(max_closer_farther_embedding) > abs(min_closer_farther_embedding)
            else min_closer_farther_embedding
        )

        experiment_result = [
            slower_faster_embedding,
            lower_higher_embedding,
            closer_farther_embedding,
            ang_accel_A,
            release_time_A,
            ang_accel_A - ang_accel_B,
            release_time_A - release_time_B
        ]

        experiment_results.append(experiment_result)

    save_experiment_results(experiment_name, env_type, experiment_results)

def run_human_experiment(env_type: str):
    def run_trial() -> str:
        return input("\n\nA is ______ than B. Input qualifier-adverb pairs, each separated by commas. For example, 'slightly higher, much farther'. Enter 'quit' to complete the session and save the human experiment results.\n> ")
    
    run_experiment("human", env_type, run_trial, visualize=True)

def run_gpt_experiment(env_type: str, num_trials: int = 100):
    def run_trial(disp_x_A: float, vel_x_A: float, max_height_A: float, disp_x_B: float, vel_x_B: float, max_height_B: float) -> Union[List[Tuple[str, str]], None]:
        try:
            adverb_input, _ = gpt_convert_env_result_to_adverb_input((disp_x_A, vel_x_A, max_height_A), (disp_x_B, vel_x_B, max_height_B))

            return adverb_input
        except:
            return None

    run_experiment("gpt", env_type, run_trial, visualize=False, max_trials=num_trials)
    
def autogenerate_experiment_results(env_type: str, num_trials: int = 100000):
    min_disp_x = 0
    max_disp_x = 0
    min_vel_x = 0
    max_vel_x = 0
    min_max_height = 0
    max_max_height = 0

    for _ in range(num_trials):
        env = LauncherEnv(env_type)

        ang_accel_A, release_time_A = env.sample_simulation_input()
        disp_x, vel_x, max_height = env.simulate(ang_accel_A, release_time_A, "A", visualize=False)

        min_disp_x = min(min_disp_x, disp_x)
        max_disp_x = max(max_disp_x, disp_x)
        min_vel_x = min(min_vel_x, vel_x)
        max_vel_x = max(max_vel_x, vel_x)
        min_max_height = min(min_max_height, max_height)
        max_max_height = max(max_max_height, max_height)

    experiment_results = load_experiment_results("auto", env_type)

    for _ in range(num_trials):
        env = LauncherEnv(env_type)

        ang_accel_A, release_time_A = env.sample_simulation_input()
        disp_x_A, vel_x_A, max_height_A = env.simulate(ang_accel_A, release_time_A, "A", visualize=False)

        ang_accel_B, release_time_B = env.sample_simulation_input()
        disp_x_B, vel_x_B, max_height_B = env.simulate(ang_accel_B, release_time_B, "B", visualize=False)

        slower_faster_embedding = min(1, max(-1, (disp_x_A - disp_x_B) / (max_disp_x - min_disp_x)))
        lower_higher_embedding = min(1, max(-1, (vel_x_A - vel_x_B) / (max_vel_x - min_vel_x)))
        closer_farther_embedding = min(1, max(-1, (max_height_A - max_height_B) / (max_max_height - min_max_height)))

        experiment_result = [
            slower_faster_embedding,
            lower_higher_embedding,
            closer_farther_embedding,
            ang_accel_A,
            release_time_A,
            ang_accel_A - ang_accel_B,
            release_time_A - release_time_B
        ]

        experiment_results.append(experiment_result)

    save_experiment_results("auto", env_type, experiment_results)