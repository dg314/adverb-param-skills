import argparse

from launcher_env import LauncherEnv
from experiment import run_gpt_experiment, run_human_experiment, load_experiment_results
from adverbs import learn_adverb_skill_groundings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalizing Adverbs Across Parameterized Skills.")
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--run_experiment", action="store_true")
    parser.add_argument("--learn_groundings", action="store_true")
    parser.add_argument("--simulate_env", action="store_true")

    parser_namespace = parser.parse_args()
    
    if parser_namespace.run_experiment:
        env_type = input("Enter env_type, either 'overhand' or 'underhand':\n> ")

        if parser_namespace.human:
            run_human_experiment(env_type)
        else:
            run_gpt_experiment(env_type)

    elif parser_namespace.learn_groundings:
        skill1_env_type = input("Enter env_type for the first skill, either 'overhand' or 'underhand':\n> ")

        experiment_name = "human" if parser_namespace.human else "gpt"
        skill1_experiment_results = load_experiment_results(experiment_name, skill1_env_type)

        skill2_env_type = "overhand" if skill1_env_type == "underhand" else "underhand"
        skill2_experiment_results = load_experiment_results(experiment_name, skill2_env_type)

        learn_adverb_skill_groundings(skill1_experiment_results, skill2_experiment_results, skill1_env_type.capitalize(), skill2_env_type.capitalize())

    elif parser_namespace.simulate_env:
        env = LauncherEnv("underhand")

        # min_dist_500_height_300 = float("inf")
        # best_ang_accel, best_release_time = 0, 0

        # for _ in range(10000):
        #     ang_accel, release_time = env.sample_simulation_input()
        #     disp_x, vel_x, max_height = env.simulate(ang_accel, release_time, visualize=False)

        #     dist_500_height_300 = ((disp_x - 500) ** 2 + (max_height - 300) ** 2) ** 0.5

        #     if dist_500_height_300 < min_dist_500_height_300:
        #         min_dist_500_height_300 = dist_500_height_300
        #         best_ang_accel, best_release_time = ang_accel, release_time

        # print(best_ang_accel, best_release_time)

        env = LauncherEnv("underhand")

        ang_accel_A, release_time_A = 2.2566150251143604, 2.4620174686808043
        env.simulate(ang_accel_A, release_time_A, title="A", visualize=True)

        ang_accel_A, release_time_A = 2.152106307223633, 4.939306633783522
        env.simulate(ang_accel_A, release_time_A, title="B", visualize=True)




