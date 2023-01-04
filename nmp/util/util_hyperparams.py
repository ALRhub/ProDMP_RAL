"""
    Utilities of hyper-parameters and randomness
"""

import random

import numpy as np
import torch
from addict import Dict


class HyperParametersPool:
    def __init__(self):
        raise RuntimeError("Do not instantiate this class.")

    @staticmethod
    def set_hyperparameters(hp_dict: Dict):
        """
        Set runtime hyper-parameters
        Args:
            hp_dict: dictionary of hyper-parameters

        Returns:
            None
        """
        if hasattr(HyperParametersPool, "_hp_dict"):
            raise RuntimeError("Hyper-parameters already exist")
        else:
            # Initialize hyper-parameters dictionary
            HyperParametersPool._hp_dict = hp_dict

            # Setup random seeds globally
            seed = hp_dict.get("seed", 1234)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @staticmethod
    def hp_dict():
        """
        Get runtime hyper-parameters
        Returns:
            hp_dict: dictionary of hyper-parameters
        """
        if not hasattr(HyperParametersPool, "_hp_dict"):
            return None
        else:
            hp_dict = HyperParametersPool._hp_dict
            return hp_dict


def decide_hyperparameter(obj: any,
                          run_time_value: any,
                          parameter_key: str,
                          parameter_default: any) -> any:
    """
    A helper function to determine function's hyper-parameter
    Args:
        obj: the object asking for hyper-parameter
        run_time_value: runtime value, will be used if it is not None
        parameter_key: the key to search in the hyper-parameters pool
        parameter_default: use this value if neither runtime nor config value

    Returns:
        the parameter following the preference
        - if runtime value is given, use it
        - else if find it in the config pool, use that one
        - else use the default value
    """
    if run_time_value is not None:
        return run_time_value
    elif hasattr(obj, parameter_key):
        return getattr(obj, parameter_key)
    else:
        hp_dict = HyperParametersPool.hp_dict()
        if hp_dict is not None \
                and parameter_key in hp_dict.keys():
            actual_value = hp_dict.get(parameter_key)
            setattr(obj, parameter_key, actual_value)
            return actual_value
        else:
            return parameter_default


def mlp_arch_3_params(avg_neuron: int, num_hidden: int, shape: float) -> [int]:
    """
    3 params way of specifying dense net, mostly for hyperparameter optimization
    Originally from Optuna work

    Args:
        avg_neuron: average number of neurons per layer
        num_hidden: number of layers
        shape: parameters between -1 and 1:
            shape < 0: "contracting" network, i.e, layers  get smaller,
                        for extrem case (shape = -1):
                        first layer 2 * avg_neuron neurons,
                        last layer 1 neuron, rest interpolating
            shape 0: all layers avg_neuron neurons
            shape > 0: "expanding" network, i.e., representation gets larger,
                        for extrem case (shape = 1)
                        first layer 1 neuron,
                        last layer 2 * avg_neuron neurons, rest interpolating

    Returns:
        architecture: list of integers representing the number of neurons of
                      each layer
    """

    assert avg_neuron >= 0
    assert -1.0 <= shape <= 1.0
    assert num_hidden >= 1
    shape = shape * avg_neuron  # we want the user to provide shape \in [-1, +1]
    architecture = []
    for i in range(num_hidden):
        # compute real-valued 'position' x of current layer (x \in (-1, 1))
        x = 2 * i / (num_hidden - 1) - 1 if num_hidden != 1 else 0.0
        # compute number of units in current layer
        d = shape * x + avg_neuron
        d = int(np.floor(d))
        if d == 0:  # occurs if shape == -avg_neuron or shape == avg_neuron
            d = 1
        architecture.append(d)
    return architecture
