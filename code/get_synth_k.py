import math
import random
import numpy as np
from launcher_env import LauncherEnv 
from synth_adverbs import turn_nums_to_adverbs

# k: the number of pairs of skill parameters (where each pair has params for two skills)
# max_a: maximum acceleration
# max_t: maximum time to release ball
def _step1_create_k_params_pairs(k: int=50, max_a: float=math.pi, max_t: float=2.0) -> np.ndarray:
    params = np.empty((k*2, 2)) #shape = (k*2, num params per skill (|a, t| = 2))
    for i in range(k*2):
        params[i, :] = math.random(0,max_a), math.random(0,max_t)

    return params

def _step2_get_policies(params: np.ndarray, env_type_: str="overhand") -> np.ndarray:
    my_env = LauncherEnv()
    
    policies = np.empty((params.shape[0], 4))

    for i, skill in enumerate(params):
        a1, t1 = skill        
        disp_x_1, vel_x_1 = my_env.simulate(env_type=env_type_, ang_accel=a1, release_time=t1)
        policies[i, :2] = skill
        policies[i, 2:] = disp_x_1, vel_x_1

    return policies

# len(policies) = k*2; len(adverbs_output) = k; adverbs_output[i] is for policies[2*i:2(i+1)] 
def _step3_get_adv_difference(policies: np.ndarray) -> list:
    k = (policies.shape[0])//2
    adverbs = [""]*k

    diff_arr = np.empty((k, 2))
    for i in range(k):
        result1 = policies[2*i, 2:]
        result2 = policies[2*i + 1, 2:]
        diff_arr[i] = result2 - result1

    openai_outputs = turn_nums_to_adverbs(diff_arr)
    #TODO: parse output to get adverb
    adverbs = openai_outputs

    return adverbs

def _step4_get_param_skill_diff(policies: np.ndarray) -> np.ndarray:
    k = (policies.shape[0])//2
    diff_arr = np.empty((k, 2))

    for i in range(k):
        params1 = policies[2*i, :2]
        params2 = policies[2*i + 1, :2]
        diff_arr[i, :] = params1 - params2

    return diff_arr

# output is list of tuples (adverb, param, param_diff_equal_to_adverb)
def get_tuples_K(k_: int=50, env_type: str="overhand", human_experiment=False) -> list:
    two_k_params = _step1_create_k_params_pairs(k=k_)
    two_k_policies = _step2_get_policies(params=two_k_params, env_type_=env_type)
    k_adverbs = _step3_get_adv_difference(two_k_policies)
    k_diff = _step4_get_param_skill_diff(two_k_policies)

    tuples_ = [None]*k_
    for i in range(k_):
        tuples_[i] = (k_adverbs[i], two_k_params[2*i], k_diff[i])

    return tuples_


