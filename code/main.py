import argparse

from launcher_env import LauncherEnv
from experiment import run_gpt_experiment, run_human_experiment, load_experiment_results, autogenerate_experiment_results
from net import learn_net_adverb_skill_groundings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalizing Adverbs Across Parameterized Skills.")
    parser.add_argument("--experiment_name", choices=["human", "gpt", "auto"], nargs=1)
    parser.add_argument("--run_experiment", action="store_true")
    parser.add_argument("--learn_groundings", action="store_true")
    parser.add_argument("--simulate_env", action="store_true")

    parser_namespace = parser.parse_args()

    experiment_name = None if not parser_namespace.experiment_name else parser_namespace.experiment_name[0]
    
    if parser_namespace.run_experiment:
        env_type = input("Enter env_type, either 'overhand' or 'underhand':\n> ")

        if experiment_name == "human":
            run_human_experiment(env_type)
        elif experiment_name == "gpt":
            run_gpt_experiment(env_type)
        else:
            autogenerate_experiment_results(env_type)

    elif parser_namespace.learn_groundings:
        skill1_env_type = input("Enter env_type for the first skill, either 'overhand' or 'underhand':\n> ")
        skill1_experiment_results = load_experiment_results(experiment_name, skill1_env_type)

        skill2_env_type = "overhand" if skill1_env_type == "underhand" else "underhand"
        skill2_experiment_results = load_experiment_results(experiment_name, skill2_env_type)

        learn_net_adverb_skill_groundings(skill1_experiment_results, skill2_experiment_results, skill1_env_type.capitalize(), skill2_env_type.capitalize())

    elif parser_namespace.simulate_env:
        env = LauncherEnv("underhand")

        ang_accel_A, release_time_A = 2.2566, 2.4620
        env.simulate(ang_accel_A, release_time_A, title="A", visualize=True)

        ang_accel_A, release_time_A = 2.1521, 4.9393
        env.simulate(ang_accel_A, release_time_A, title="B", visualize=True)
