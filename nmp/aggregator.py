"""
@brief:     Aggregator classes in PyTorch
"""
from typing import Optional

import torch


class BayesianAggregator:
    """A Bayesian Aggregator"""

    def __init__(self, **kwargs):
        """
        Bayesian Aggregator constructor
        Args:
            **kwargs: aggregator configuration
        """

        # Aggregator dimension
        self.dim_lat_obs: int = kwargs["dim_lat"]
        self.dim_lat: int = kwargs["dim_lat"]
        self.multiple_steps: bool = kwargs.get("multiple_steps", False)

        # Scalar prior
        self.prior_mean_init = kwargs["prior_mean"]
        self.prior_var_init = kwargs["prior_var"]
        assert self.prior_var_init >= 0  # We only consider diagonal
        # terms, so always be positive

        # Number of trajectories, i.e. equals to batch size
        self.num_traj = None

        # Number of aggregated subsets, each subset may contain more than 1 obs
        self.num_agg = 0

        # Number of aggregated observations
        self.num_agg_obs = 0

        # Aggregation history of latent variables
        self.mean_lat_var_state = None
        self.variance_lat_var_state = None

    def reset(self, num_traj: int):
        """
        Reset aggregator

        Args:
            num_traj: batch size

        Returns:
            None

        """

        # Reset num_traj, i.e. equals to batch size
        self.num_traj = num_traj

        # Reset number of counters
        self.num_agg = 0
        self.num_agg_obs = 0

        # Reset aggregation history of latent variables
        # i.e. mean_lat_var_state and variance_lat_var_state
        #
        # Note its shape[1] = num_agg + 1, which tells how many context sets
        # have been aggregated by the aggregator, e.g. index 0 denotes the prior
        # distribution of latent variable, index -1 denotes the current
        # distribution of latent variable. Note in each aggregation, the latent
        # observation may have different number of samples.
        #
        # Shape of mean_lat_var_state: [num_traj, num_agg + 1, dim_lat]
        # Shape of variance_lat_var_state: [num_traj, num_agg + 1, dim_lat]

        # Get prior tensors from scalar
        prior_mean, prior_var = self.generate_prior(self.prior_mean_init,
                                                    self.prior_var_init)

        # Add one axis (record number of aggregation)
        self.mean_lat_var_state = prior_mean[:, None, :]
        self.variance_lat_var_state = prior_var[:, None, :]

    def generate_prior(self, mean: float, cov: float) \
            -> (torch.Tensor, torch.Tensor):
        """
        Given scalar values of mean and covariance, generate prior tensor
        Args:
            mean: scalar value of mean
            cov: scalar value of covariance

        Returns: tensors of prior's mean and prior's variance
        """
        # Shape of prior_mean, prior_var:
        # [num_traj, dim_lat]

        prior_mean = torch.full(size=(self.num_traj, self.dim_lat),
                                fill_value=mean)
        prior_var = torch.full(size=(self.num_traj, self.dim_lat),
                               fill_value=cov)
        return prior_mean, prior_var

    def aggregate(self, lat_obs: torch.Tensor, var_lat_obs: torch.Tensor):
        """
        Aggregate info of latent observation and compute new latent variable.
        If there's no latent observation, then return prior of latent variable.

        Args:
            lat_obs: latent observations of samples in certain trajectories
            var_lat_obs: covariance (uncertainty) of latent observations

        Returns:
            None
        """

        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]
        #
        # Shape of var_lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Case without latent observation
        if lat_obs.shape[1] == 0 and var_lat_obs.shape[1] == 0:
            # No latent observation, do not update the latent variable state
            pass

        # Case with latent observation
        else:
            # Check input shapes
            assert lat_obs.ndim == var_lat_obs.ndim == 3
            assert lat_obs.shape == var_lat_obs.shape
            assert lat_obs.shape[0] == self.num_traj
            assert lat_obs.shape[2] == self.dim_lat_obs

            # number of observations
            num_obs = lat_obs.shape[1]

            # Get the latest latent variable distribution
            mean_lat_var = self.mean_lat_var_state[:, -1, :]
            variance_lat_var = self.variance_lat_var_state[:, -1, :]

            # Aggregate
            agg_step = 1 if self.multiple_steps else num_obs
            for idx in range(0, num_obs, agg_step):
                # Update uncertainty of latent variable
                variance_lat_var = \
                    1 / (1 / variance_lat_var
                         + torch.sum(1 / var_lat_obs[:, idx:idx + agg_step, :],
                                     dim=1))
                # Update mean of latent variable
                mean_lat_var = mean_lat_var + variance_lat_var * torch.sum(
                    1 / var_lat_obs[:, idx:idx + agg_step, :]
                    * (lat_obs[:, idx:idx + agg_step, :]
                       - mean_lat_var[:, None, :]), dim=1)

                # Append to latent variable state
                self.mean_lat_var_state = torch.cat(
                    (self.mean_lat_var_state, mean_lat_var[:, None, :]), dim=1)
                self.variance_lat_var_state = \
                    torch.cat((self.variance_lat_var_state,
                               variance_lat_var[:, None, :]), dim=1)

                # Update counters
                self.num_agg += 1
                self.num_agg_obs += agg_step

    def get_agg_state(self, index: Optional[int]) \
            -> (torch.Tensor, torch.Tensor):
        """
        Return all latent variable state, or the one at given index.
        E.g. index -1 denotes the last latent variable state; index 0 the prior

        Returns:
            mean_lat_var_state: mean of the latent variable state
            variance_lat_var_state: covariance of the latent variable state
        """

        # Shape of mean_lat_var_state:
        # [num_traj, num_agg, dim_lat]
        #
        # Shape of variance_lat_var_state:
        # [num_traj, num_agg, dim_lat]
        #
        # num_agg = 1 if index is not None

        if index is None:
            # Full case
            return self.mean_lat_var_state, self.variance_lat_var_state

        elif index == -1 or index + 1 == self.mean_lat_var_state.shape[1]:
            # Index case -1
            return self.mean_lat_var_state[:, index:, :], \
                   self.variance_lat_var_state[:, index:, :]
        else:
            # Other index cases
            return self.mean_lat_var_state[:, index: index + 1, :], \
                   self.variance_lat_var_state[:, index:index + 1, :]


class MeanAggregator:
    """A mean aggregator"""

    def __init__(self, **kwargs):
        """
        Mean aggregator constructor
        Args:
            **kwargs: aggregator configuration
        """
        # Aggregator dimension
        self.dim_lat_obs: int = kwargs["dim_lat"]
        self.multiple_steps: bool = kwargs.get("multiple_steps", False)

        # Scalar prior
        self.prior_mean_init = kwargs["prior_mean"]

        # Number of trajectories, i.e. equals to batch size
        self.num_traj = None

        # Number of aggregated subsets, each subset may have more than 1 obs
        self.num_agg = 0

        # Number of aggregated obs
        self.num_agg_obs = 0

        # Aggregation history of latent observation
        self.mean_lat_obs_state = None

    def reset(self, num_traj: int):
        """
        Reset aggregator

        Args:
            num_traj: batch size

        Returns:
            None

        """
        # Reset num_traj, i.e. equals to batch size
        self.num_traj = num_traj

        # Reset counters
        self.num_agg = 0
        self.num_agg_obs = 0

        # Reset aggregation history of latent observation
        # i.e. mean_lat_rep_state
        #
        # Note its shape[1] = num_agg + 1, which tells how many context
        # "sets" have been aggregated by the aggregator, e.g. index 0
        # denotes the prior mean of latent observation, index -1 denotes the
        # current mean of latent observation. Note in each aggregation, the
        # latent observation to be aggregated may have different number of
        # samples.
        #
        # Shape of mean_lat_obs_state: [num_traj, num_agg + 1, dim_lat]

        # Get prior tensors from scalar
        prior_mean = self.generate_prior(self.prior_mean_init)

        # Add one axis (record number of aggregation)
        self.mean_lat_obs_state = prior_mean[:, None, :]

    def generate_prior(self, mean: float) -> torch.Tensor:
        """
        Given scalar value of mean, generate prior tensor
        Args:
            mean: scalar value of mean

        Returns: tensors of prior's mean for mean latent observation
        """
        # Shape of prior_mean:
        # [num_traj, dim_lat]

        prior_mean = torch.full(size=(self.num_traj, self.dim_lat_obs),
                                fill_value=mean)

        return prior_mean

    def aggregate(self, lat_obs: torch.Tensor):
        """
        Aggregate info of latent observation

        Args:
            lat_obs: latent observations of samples in certain trajectories

        Returns:
            None
        """

        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Case without latent observation
        if lat_obs.shape[1] == 0:
            # No latent observation, do not update the latent observations
            pass

        else:
            # Check input shapes
            assert lat_obs.ndim == 3
            assert lat_obs.shape[0] == self.num_traj
            assert lat_obs.shape[2] == self.dim_lat_obs

            # Number of observations
            num_obs = lat_obs.shape[1]

            # Get latest latent obs
            mean_lat_obs = self.mean_lat_obs_state[:, -1, :]

            # Aggregate
            agg_step = 1 if self.multiple_steps else num_obs
            for step in range(0, num_obs, agg_step):
                # Compute new mean
                mean_lat_obs = \
                    (mean_lat_obs * self.num_agg_obs +
                     torch.sum(lat_obs[:, step:step + agg_step, :], dim=1)) \
                    / (self.num_agg_obs + agg_step)

                # Append
                self.mean_lat_obs_state = torch.cat(
                    (self.mean_lat_obs_state, mean_lat_obs[:, None, :]), dim=1)

                # Update counters
                self.num_agg += 1
                self.num_agg_obs += agg_step

    def get_agg_state(self, index: Optional[int]) -> torch.Tensor:
        """
        Return all latent observation state, or the one at given index.
        E.g. index -1 denotes the last latent obs state; index 0 the prior

        Returns:
            mean_lat_obs_state: mean of the latent observation state
        """

        # Shape of mean_lat_obs_state:
        # [num_traj, num_agg, dim_lat]
        #
        # num_agg = 1 if index is not None

        if index is None:
            # Full case
            return self.mean_lat_obs_state

        elif index == -1 or index + 1 == self.mean_lat_obs_state.shape()[1]:
            # Index case -1
            return self.mean_lat_obs_state[:, index:, :]
        else:
            # Other index cases
            return self.mean_lat_obs_state[:, index: index + 1, :]

    # End of class MeanAggregator


class AggregatorFactory:

    @staticmethod
    def get_aggregator(aggregator_type: str, **kwargs):
        return eval(aggregator_type + "(**kwargs)")
