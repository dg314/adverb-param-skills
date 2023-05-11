from typing import List, Tuple

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

def run_human_experiment(env_type: str):
    keyed_vectors = download_keyed_vectors()

    word_to_magnitude_predictor = create_word_to_magnitude_predictor(keyed_vectors)
    word_to_slower_faster_predictor = create_word_to_slower_faster_predictor(keyed_vectors)
    word_to_lower_higher_predictor = create_word_to_lower_higher_predictor(keyed_vectors)
    word_to_closer_farther_predictor = create_word_to_closer_farther_predictor(keyed_vectors)

    human_experiment_results = load_experiment_results("human", env_type)

    while True:
        env = LauncherEnv(env_type)

        ang_accel_A, release_time_A = env.sample_simulation_input()
        env.simulate(ang_accel_A, release_time_A, "A", visualize=True)

        ang_accel_B, release_time_B = env.sample_simulation_input()
        env.simulate(ang_accel_B, release_time_B, "B", visualize=True)

        user_input = input("\n\nA is ______ than B. Input qualifier-adverb pairs, each separated by commas. For example, 'slightly higher, much farther'. Enter 'quit' to complete the session and save the human experiment results.\n> ")

        if user_input == "quit":
            break

        adverb_input = [pair.strip().split(" ") for pair in user_input.split(",")]

        if not all([len(pair) == 2 for pair in adverb_input]):
            print("\nInvalid adverb format.\n\n")
            continue
        
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

        human_experiment_result = [
            slower_faster_embedding,
            lower_higher_embedding,
            closer_farther_embedding,
            ang_accel_A,
            release_time_A,
            ang_accel_B - ang_accel_A,
            release_time_B - release_time_A
        ]

        human_experiment_results.append(human_experiment_result)

    save_experiment_results("human", env_type, human_experiment_results)

def run_gpt_experiment(env_type: str, iterations: int = 100):
    keyed_vectors = download_keyed_vectors()

    word_to_magnitude_predictor = create_word_to_magnitude_predictor(keyed_vectors)
    word_to_slower_faster_predictor = create_word_to_slower_faster_predictor(keyed_vectors)
    word_to_lower_higher_predictor = create_word_to_lower_higher_predictor(keyed_vectors)
    word_to_closer_farther_predictor = create_word_to_closer_farther_predictor(keyed_vectors)

    gpt_experiment_results = load_experiment_results("gpt", env_type)

    for i in range(iterations):
        try:
            env = LauncherEnv(env_type)

            ang_accel_A, release_time_A = env.sample_simulation_input()
            disp_x_A, vel_x_A, max_height_A = env.simulate(ang_accel_A, release_time_A, "A", visualize=False)

            ang_accel_B, release_time_B = env.sample_simulation_input()
            disp_x_B, vel_x_B, max_height_B = env.simulate(ang_accel_B, release_time_B, "B", visualize=False)

            adverb_input, raw_output = gpt_convert_env_result_to_adverb_input((disp_x_A, vel_x_A, max_height_A), (disp_x_B, vel_x_B, max_height_B))

            if adverb_input is None or not all([len(pair) == 2 for pair in adverb_input]):
                print(f"\033[91mGPT experiment iteration {i} completed with incorrectly formatted output.\n\nadverb_input={adverb_input}\n\nraw_output={raw_output}\n\n\033[0m")
                continue
            else:
                print(f"\033[92mGPT experiment iteration {i} completed with correctly formatted output.\033[0m")
            
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

            gpt_experiment_result = [
                slower_faster_embedding,
                lower_higher_embedding,
                closer_farther_embedding,
                ang_accel_A,
                release_time_A,
                ang_accel_B - ang_accel_A,
                release_time_B - release_time_A
            ]

            gpt_experiment_results.append(gpt_experiment_result)

        except:
            print(f"\033[92mAn unknown error has occurred.\033[0m")

    save_experiment_results("gpt", env_type, gpt_experiment_results)
    