"""
@brief:     Classes and method of data processing
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import nmp.util as util


def get_time_normalizer(time_min: float, time_max: float) -> dict:
    """
    Compute time normalizer under a uniform distribution assumption
    Args:
        time_min: min time range
        time_max: max time range

    Returns:
        dictionary of time normalizer
    """
    # Time normalizer, assume Uniform distributed
    assert time_max - time_min > 1e-5, "Time should be variant, not constant."
    time_mean = (time_max + time_min) / 2
    time_std = (time_max - time_min) / (2 * 1.732)
    time_duration = time_max - time_min
    return {"min": torch.Tensor([time_min]),
            "max": torch.Tensor([time_max]),
            "mean": torch.Tensor([time_mean]),
            "std": torch.Tensor([time_std]),
            "duration": torch.Tensor([time_duration]).item()}


def get_value_normalizer(values, time_dependent=False) -> dict:
    """
    Compute value normalizer
    Args:
        values: data array
        time_dependent: if value is time-dependent

    Returns:
        dictionary of value normalizer

    """
    mean_axis = (0, 1) if time_dependent else (0,)

    value_mean = torch.Tensor(np.mean(values, axis=mean_axis))
    value_std = torch.Tensor(np.std(values, axis=mean_axis, ddof=1))
    value_min = torch.Tensor(np.min(values, axis=mean_axis))
    value_max = torch.Tensor(np.max(values, axis=mean_axis))

    assert torch.min(value_std) > 0.0, \
        "Value should be either time variant or task variant" \
        " rather than constant."
    return {"mean": value_mean,
            "std": value_std,
            "min": value_min,
            "max": value_max}


def check_dataset_length(data_dict):
    """
    Check and return the length (num of trajectories) of the dataset
    Args:
        data_dict: dataset dictionary

    Returns:
        length of dataset
    """

    # Check if len of dataset applies to all data
    data_lengths = list()
    for data in data_dict.values():
        data_lengths.append(len(data["value"]))
        if "time" in data.keys():
            data_lengths.append(len(data["time"]))
    assert all(length == data_lengths[0] for length in data_lengths)

    return data_lengths[0]


def split_dataset(partition: dict,
                  shuffle: bool,
                  data_dict: dict) -> [dict]:
    """
    Split a dataset into train, validate and test sets
    Args:
        partition: ratio of split
        shuffle: if shuffle the date order
        data_dict: dictionary storing data

    Returns:
        train, validate, and test dataset
    """
    # Get length of dataset
    len_dataset = check_dataset_length(data_dict)
    len_train = int(len_dataset * partition["train"])
    len_validate = int(len_dataset * partition["validate"])

    # Shuffle indices
    if shuffle:
        data_idx = np.random.permutation(len_dataset)
    else:
        data_idx = range(len_dataset)
    train_val_test_idx_list = [data_idx[:len_train],
                               data_idx[len_train: len_train + len_validate],
                               data_idx[len_train + len_validate:]]

    # Split dataset into three
    train_val_test_dict_list = [dict(), dict(), dict()]
    for name, data in data_dict.items():
        for idx, dic in zip(train_val_test_idx_list, train_val_test_dict_list):
            dic[name] = dict()
            dic[name]["value"] = data["value"][idx]
            if "time" in data.keys():
                dic[name]["time"] = data["time"][idx]

    # Return
    train_dict, val_dict, test_dict = train_val_test_dict_list
    return train_dict, val_dict, test_dict


def get_data_loaders_and_normalizer(data_dict: dict, **kwargs) \
        -> [DataLoader, DataLoader, DataLoader, dict]:
    """
    Given all data, get data loaders and normalizer
    Args:
        data_dict: dictionary storing all data
        **kwargs: keyword arguments

    Returns:
        Train, Validate, and Test data loaders and data normalizer
    """

    partition = kwargs["partition"]
    shuffle_set = kwargs["shuffle_set"]
    batch_size = kwargs["batch_size"]
    shuffle_train_loader = kwargs["shuffle_train_loader"]

    # Split data into three subsets
    train_set, vali_set, test_set = \
        split_dataset(partition, shuffle_set, data_dict)

    # Get PyTorch dataset
    train_set = MPDataset(train_set, True, **kwargs)
    vali_set = MPDataset(vali_set, False, **kwargs)
    test_set = MPDataset(test_set, False, **kwargs)

    # Decide batch size
    if batch_size is None:
        train_batch_size = train_set.len_dataset
        vali_batch_size = vali_set.len_dataset
        test_batch_size = test_set.len_dataset
    else:
        train_batch_size = vali_batch_size = test_batch_size = batch_size

    # Get PyTorch dataloader, here the GPU generator cannot be set globally
    generator = torch.Generator(device=util.current_device())
    seed = kwargs.get("seed", None)
    if seed is not None:
        generator.manual_seed(seed)
    train_loader = DataLoader(train_set, train_batch_size, shuffle_train_loader,
                              generator=generator)
    vali_loader = DataLoader(vali_set, vali_batch_size, False,
                             generator=generator)
    test_loader = DataLoader(test_set, test_batch_size, False,
                             generator=generator)

    # Get data normalizer
    normalizer = train_set.get_normalizers()
    return train_loader, vali_loader, test_loader, normalizer


def select_ctx_pred_pts(**kwargs):
    """
    Generate context and prediction indices
    Args:
        **kwargs: keyword arguments

    Returns:
        context indices and prediction indices

    """
    num_ctx = kwargs.get("num_ctx", None)
    num_ctx_min = kwargs.get("num_ctx_min", None)
    num_ctx_max = kwargs.get("num_ctx_max", None)
    first_index = kwargs.get("first_index", None)
    fixed_interval = kwargs.get("fixed_interval", False)
    num_all = kwargs.get("num_all", None)
    num_select = kwargs.get("num_select", None)
    ctx_before_pred = kwargs.get("ctx_before_pred", False)

    # Determine how many points shall be selected
    if num_select is None:
        assert fixed_interval is False
        assert first_index is None
        num_select = num_all
    else:
        assert num_select <= num_all

    # Determine how many context points shall be selected
    if num_ctx is None:
        num_ctx = torch.randint(low=num_ctx_min, high=num_ctx_max, size=(1,))
    assert num_ctx < num_select

    # Select points
    if fixed_interval:
        # Select using fixed interval
        interval = num_all // num_select
        residual = num_all % num_select

        if first_index is None:
            # Determine the first index
            first_index = \
                torch.randint(low=0, high=interval + residual, size=[]).item()
        else:
            # The first index is specified
            assert 0 <= first_index < interval + residual
        selected_indices = torch.arange(start=first_index, end=num_all,
                                        step=interval, dtype=torch.long)
    else:
        # Select randomly
        permuted_indices = torch.randperm(n=num_all)
        selected_indices = torch.sort(permuted_indices[:num_select])[0]

    # split ctx and pred
    if num_ctx == 0:
        # No context
        ctx_idx = []
        pred_idx = selected_indices

    else:
        # Ctx + Pred
        if ctx_before_pred:
            ctx_idx = selected_indices[:num_ctx]
            pred_idx = selected_indices[num_ctx:]
        else:
            permuted_select_indices = torch.randperm(n=num_select)
            ctx_idx = selected_indices[permuted_select_indices[:num_ctx]]
            pred_idx = selected_indices[permuted_select_indices[num_ctx:]]
    return ctx_idx, pred_idx


class MPDataset(Dataset):
    def __init__(self, data_dict: dict,
                 compute_normalizer=True,
                 transform=None,
                 **kwargs):
        """
        Initialize a PyTorch dataset

        Args:
            data_dict: dictionary storing dataset in numpy
            compute_normalizer: True if normalizer should be computed
            transform: transform data when indexing the dataset
            **kwargs: keyword arguments
        """

        # Get info referring to the data to be generated as torch dataset
        self.dict_data_info = kwargs["data"]

        # Check if dataset contains all data's keys used for training
        config_keys = set(self.dict_data_info.keys())
        data_keys = set(data_dict.keys())
        assert config_keys.issubset(data_keys)

        # Properties exist
        assert all("time_dependent" in info.keys()
                   for info in self.dict_data_info.values())

        # Transformer and normalizer
        self.transform = transform
        self.compute_normalizer = compute_normalizer

        # Length of the dataset, how many trajectories
        self.len_dataset = None

        # Initialize a dictionary to store data
        self.dict_all_data = dict()

        # Initialize a dictionary to store normalizer
        self.dict_normalizer = dict() if self.compute_normalizer else None

        # Finish initialization
        self._initialize(data_dict, **kwargs)

    def _initialize(self, data_dict: dict, **kwargs):
        """
        Storing data and compute normalizer

        Args:
            data_dict: data dictionary
            kwargs: keyword arguments
        Returns:
            None
        """
        # Time normalizer
        if self.compute_normalizer:
            assert {"time_min", "time_max"}.issubset(kwargs.keys())
            self.dict_normalizer["time"] = \
                get_time_normalizer(kwargs["time_min"], kwargs["time_max"])
        # Loop over all data keys
        for name, info in self.dict_data_info.items():

            # Check len of dataset applies to all data
            if self.len_dataset is None:
                self.len_dataset = data_dict[name]["value"].shape[0]
            else:
                assert self.len_dataset == data_dict[name]["value"].shape[0]

            # Check time dependency
            assert info["time_dependent"] == ("time" in data_dict[name].keys())

            # Value normalizer
            if self.compute_normalizer:
                if info.get("normalize", True):
                    self.dict_normalizer[name] = \
                        get_value_normalizer(data_dict[name]["value"],
                                             info["time_dependent"])
                else:
                    self.dict_normalizer[name] = None
            # Save data
            save_type = kwargs["save_type"]
            if save_type == "array":
                self.dict_all_data[name] = data_dict[name]
            elif save_type == "tensor":
                self.dict_all_data[name] = util.to_tensor_dict(data_dict[name])
            else:
                raise NotImplementedError

    def __len__(self):
        """
        Build-in function

        Returns: len of dataset

        """
        return self.len_dataset

    def __getitem__(self, index):
        """
        Indexing dataset
        Args:
            index: index

        Returns:
            Dictionary of data
        """

        # Initialize a dictionary to store data
        dict_indexed_data = dict()

        # Loop over all data names
        for name, info in self.dict_data_info.items():
            if info["time_dependent"]:
                data_times = self.dict_all_data[name]["time"][index]
                data_values = self.dict_all_data[name]["value"][index]
                dict_indexed_data[name] = {"time": data_times,
                                           "value": data_values}
            else:
                data_values = self.dict_all_data[name]["value"][index]
                dict_indexed_data[name] = {"value": data_values}

        # Apply pre-processing
        if self.transform:
            dict_indexed_data = self.transform(dict_indexed_data)

        # Return dictionary of tuples
        return dict_indexed_data

    def get_normalizers(self):
        """
        Get data normalizer
        Returns:
            A dictionary storing normalizer
        """
        if self.compute_normalizer:
            return self.dict_normalizer
        else:
            raise RuntimeError("No normalizer exist!")

    def get_data_info(self):
        """
        Get data info
        Returns:
            A dict storing data info
        """
        return self.dict_data_info


class PreProcess:
    """ A class for pre-processing when iterate dataset"""

    class ToTensorDict(object):
        """Convert ndarray to PyTorch Tensors."""

        def __call__(self, dict_data: dict):
            return util.to_tensor_dict(dict_data)


class NormProcess:
    """ A class for processing batch data during runtime"""

    @staticmethod
    def batch_normalize(normalizer: dict,
                        dict_batch: dict):
        """
        Normalize batch data, multiple options available

        Note here the dict_batch is supposed to be a dict, for example:
        {
            "x": {"time": times, "value": values},
            "y": {"time": times, "value": values},
            "h": {"value": values}
        }

        Args:
            normalizer: A dictionary-like normalizer
            dict_batch: A dictionary-like raw data batch

        Returns:
            Normalized batch data, dictionary-like
        """

        # Initialization
        normalized_batch = dict()

        # Loop over all data keys
        for name, data_batch in dict_batch.items():

            normalized_batch[name] = dict()

            # Normalize time
            if "time" in data_batch.keys():
                time_normalizer = normalizer["time"]

                time_mean = time_normalizer["mean"]
                time_std = time_normalizer["std"]
                normalized_time = \
                    (data_batch["time"] - time_mean) / time_std

                normalized_batch[name]["time"] = normalized_time

            # Normalize value
            if "value" in data_batch.keys():
                value_normalizer = normalizer[name]
                # todo, remove the redundancy
                value_mean = value_normalizer["mean"][None, None, :]
                value_std = value_normalizer["std"][None, None, :]
                normalized_value = \
                    (data_batch["value"] - value_mean) / value_std

                normalized_batch[name]["value"] = normalized_value

        # Return
        return normalized_batch

    @staticmethod
    def batch_denormalize(normalizer: dict,
                          dict_batch: dict):
        """
        Denormalize batch data, multiple options available

        Note here the dict_batch is supposed to be a dict, for example:
        {
            "x": {"time": times, "value": values},
            "y": {"time": times, "value": values},
            "h": {"value": values}
        }

        Args:
            normalizer: A dictionary-like normalizer
            dict_batch: A dictionary-like batch to be denormalized
        Returns:
            Denormalized batch data, dictionary-like
        """
        # Initialization
        denormalized_batch = dict()

        # Loop over all data keys
        for name, data_batch in dict_batch.items():

            denormalized_batch[name] = dict()

            # Denormalize time
            if "time" in data_batch.keys():
                num_add_dim = data_batch["time"].ndim - 1
                str_add_dim = str([None] * num_add_dim)[1:-1]

                time_normalizer = normalizer["time"]
                # todo, remove the redundancy
                time_mean = time_normalizer["mean"][eval(str_add_dim)]
                time_std = time_normalizer["std"][eval(str_add_dim)]
                denormalized_time = data_batch["time"] * time_std + time_mean
                denormalized_batch[name]["time"] = denormalized_time

            # Denormalize value
            if "value" in data_batch.keys():
                num_add_dim = data_batch["value"].ndim - 1
                str_add_dim = str([None] * num_add_dim)[1:-1]

                value_normalizer = normalizer[name]
                value_mean = value_normalizer["mean"][eval(str_add_dim)]

                value_std = value_normalizer["std"][eval(str_add_dim)]
                denormalized_value = \
                    data_batch["value"] * value_std + value_mean
                denormalized_batch[name]["value"] = denormalized_value

        # Return
        return denormalized_batch

    @staticmethod
    def distribution_denormalize(normalizer: dict,
                                 key: str,
                                 mean_batch: torch.Tensor,
                                 diag_batch: torch.Tensor,
                                 off_diag_batch: torch.Tensor = None):
        """
        Denormalize predicted mean and cov Cholesky
        Args:
            normalizer: A dictionary-like normalizer
            key: data key to be denormalized
            mean_batch: predicted normalized mean batch
            diag_batch: predicted normalized diagonal batch
            off_diag_batch: ... off-diagonal batch

        Returns:
            De-normalized mean and Cholesky Decomposition L

        """

        value_mean = normalizer[key]["mean"]
        value_std = normalizer[key]["std"]

        de_norm_matrix = util.build_lower_matrix(value_std, None)

        # Denormalize mean
        de_mean_batch = torch.einsum('...ij,...j->...i', de_norm_matrix,
                                     mean_batch) + value_mean

        # Denormalize cov Cholesky
        L = util.build_lower_matrix(diag_batch, off_diag_batch)
        de_L_batch = torch.einsum('...ij,...jk->...ik', de_norm_matrix, L)

        return de_mean_batch, de_L_batch
