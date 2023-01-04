"""
@brief:     Decoder classes in PyTorch
"""

from abc import ABC
from abc import abstractmethod
# Import Python libs
from typing import Optional

import torch

import nmp.util as util
from nmp.nn_base import MLP
from nmp.util import mlp_arch_3_params


class Decoder(ABC):
    """Decoder class interface"""

    def __init__(self, **kwargs):
        """
        Constructor

        Args:
            **kwargs: Decoder configuration
        """

        # MLP configuration
        self.dim_add_in: int = kwargs["dim_add_in"]
        self.dim_val: int = kwargs["dim_val"]
        self.dim_lat: int = kwargs["dim_lat"]
        self.std_only: bool = kwargs["std_only"]

        self.mean_hidden: dict = kwargs["mean_hidden"]
        self.variance_hidden: dict = kwargs["variance_hidden"]

        self.act_func: str = kwargs["act_func"]

        # Decoders
        self.mean_val_net = None
        self.cov_val_net = None

        # Create decoders
        self._create_network()

    @property
    def _decoder_type(self) -> str:
        """
        Returns: string of decoder type
        """
        return self.__class__.__name__

    def _create_network(self):
        """
        Create decoder with given configuration

        Returns:
            None
        """

        # compute the output dimension of covariance network
        if self.std_only:
            # Only has diagonal elements
            dim_out_cov = self.dim_val
        else:
            # Diagonal + Non-diagonal elements, form up Cholesky Decomposition
            dim_out_cov = self.dim_val \
                          + (self.dim_val * (self.dim_val - 1)) // 2

        # Two separate value decoders: mean_val_net + cov_val_net
        self.mean_val_net = MLP(name=self._decoder_type + "_mean_val",
                                dim_in=self.dim_add_in + self.dim_lat,
                                dim_out=self.dim_val,
                                hidden_layers=
                                mlp_arch_3_params(**self.mean_hidden),
                                act_func=self.act_func)

        self.cov_val_net = MLP(name=self._decoder_type + "_cov_val",
                               dim_in=self.dim_add_in + self.dim_lat,
                               dim_out=dim_out_cov,
                               hidden_layers=
                               mlp_arch_3_params(**self.variance_hidden),
                               act_func=self.act_func)

    @property
    def network(self):
        """
        Return decoder networks

        Returns:
        """
        return self.mean_val_net, self.cov_val_net

    @property
    def parameters(self) -> []:
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.mean_val_net.parameters()) + \
               list(self.cov_val_net.parameters())

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.mean_val_net.save(log_dir, epoch)
        self.cov_val_net.save(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.mean_val_net.load(log_dir, epoch)
        self.cov_val_net.load(log_dir, epoch)

    def _process_cov_net_output(self, cov_val: torch.Tensor):
        """
        Divide diagonal and off-diagonal elements of cov-net output,
        apply reverse "Log-Cholesky to diagonal elements"
        Args:
            cov_val: output of covariance network

        Returns: diagonal and off-diagonal tensors

        """
        # Decompose diagonal and off-diagonal elements
        diag_cov_val = cov_val[..., :self.dim_val]
        off_diag_cov_val = None if self.std_only \
            else cov_val[..., self.dim_val:]

        # De-parametrize Log-Cholesky for diagonal elements
        diag_cov_val = util.to_softplus_space(diag_cov_val, lower_bound=None)

        # Return
        return diag_cov_val, off_diag_cov_val

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass


class PBDecoder(Decoder):
    """Parameter based decoder"""

    def decode(self,
               add_inputs: Optional[torch.Tensor],
               mean_lat_var: torch.Tensor,
               variance_lat_var: torch.Tensor) \
            -> [torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode and compute target value's distribution

        Here, target value to be predicted is a 4th order tensor with axes:

        traj: this target value is on which trajectory
        aggr: based on how much aggregated context do we make this prediction?
        tar: this target value is on which target time?
        value: vector to be predicted

        Args:
            add_inputs: additional inputs, can be None
            mean_lat_var: mean of latent variable
            variance_lat_var: variance of latent variable

        Returns:
            mean_val: mean of target value

            diag_cov_val: diagonal elements of Cholesky Decomposition of
            covariance of target value

            off_diag_cov_val: None, or off-diagonal elements of Cholesky
            Decomposition of covariance of target value

        """

        # Shape of mean_lat_var:
        # [num_traj, num_agg, dim_lat]
        #
        # Shape of variance_lat_var:
        # [num_traj, num_agg, dim_lat]
        #
        # Shape of add_inputs:
        # [num_traj, num_time_pts, dim_add_in=1]
        #
        # Shape of mean_val:
        # [num_traj, num_agg, num_time_pts, dim_val]
        #
        # Shape of diag_cov_val:
        # [num_traj, num_agg, num_time_pts, dim_val]
        #
        # Shape of off_diag_cov_val:
        # [num_traj, num_agg, num_time_pts, (dim_val * (dim_val - 1) // 2)]

        # Dimension check
        assert mean_lat_var.ndim == variance_lat_var.ndim == 3
        num_agg = mean_lat_var.shape[1]

        # Process add_inputs
        if add_inputs is not None:
            assert add_inputs.ndim == 3
            num_time_pts = add_inputs.shape[1]
            # Add one axis (aggregation-wise batch dimension) to add_inputs
            add_inputs = util.add_expand_dim(add_inputs, [1], [num_agg])
        else:
            num_time_pts = 1

        # Parametrize variance
        variance_lat_var = util.to_log_space(variance_lat_var,
                                             lower_bound=None)

        # Add one axis (time-scale-wise batch dimension) to latent variable
        mean_lat_var = util.add_expand_dim(mean_lat_var, [2], [num_time_pts])
        variance_lat_var = util.add_expand_dim(variance_lat_var, [2],
                                               [num_time_pts])

        # Prepare input to decoder networks
        mean_net_input = mean_lat_var
        cov_net_input = variance_lat_var
        if add_inputs is not None:
            mean_net_input = torch.cat((add_inputs, mean_net_input), dim=-1)
            cov_net_input = torch.cat((add_inputs, cov_net_input), dim=-1)

        # Decode
        mean_val = self.mean_val_net(mean_net_input)
        cov_val = self.cov_val_net(cov_net_input)

        # Process cov net prediction
        diag_cov_val, off_diag_cov_val = self._process_cov_net_output(cov_val)

        # Return
        return mean_val, diag_cov_val, off_diag_cov_val


class CNPDecoder(Decoder):
    """Conditional Neural Processes decoder"""

    def decode(self,
               add_inputs: torch.Tensor,
               mean_lat_obs: torch.Tensor) \
            -> [torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode and compute target value's distribution at target add_inputs

        Here, target value to be predicted is a 4th order tensor with axes:

        traj: this target value is on which trajectory
        aggr: based on how much aggregated context do we make this prediction?
        tar: this target value is on which target time?
        value: vector to be predicted

        Args:
            add_inputs: additional inputs, can be None
            mean_lat_obs: mean of latent observation

        Returns:
            mean_val: mean of target value

            diag_cov_val: diagonal elements of Cholesky Decomposition of
            covariance of target value

            off_diag_cov_val: None, or off-diagonal elements of Cholesky
            Decomposition of covariance of target value

        """

        # Shape of mean_lat_obs:
        # [num_traj, num_agg, dim_lat]
        #
        # Shape of add_inputs:
        # [num_traj, num_time_pts, dim_add_in=1] if add_inputs not None
        #
        # Shape of mean_val:
        # [num_traj, num_agg, num_time_pts, dim_val]
        #
        # Shape of diag_cov_val:
        # [num_traj, num_agg, num_time_pts, dim_val]
        #
        # Shape of off_diag_cov_val:
        # [num_traj, num_agg, num_time_pts, (dim_val * (dim_val - 1) // 2)]

        # Dimension check
        assert mean_lat_obs.ndim == 3
        num_agg = mean_lat_obs.shape[1]

        # Process add_inputs
        if add_inputs is not None:
            assert add_inputs.ndim == 3
            # Get dimensions
            num_time_pts = add_inputs.shape[1]
            # Add one axis (aggregation-wise batch dimension) to add_inputs
            add_inputs = util.add_expand_dim(add_inputs, [1], [num_agg])
        else:
            num_time_pts = 1

        # Add one axis (time-scale-wise batch dimension) to latent observation
        mean_lat_obs = util.add_expand_dim(mean_lat_obs, [2], [num_time_pts])

        # Prepare input to decoder network
        net_input = mean_lat_obs
        if add_inputs is not None:
            net_input = torch.cat((add_inputs, net_input), dim=-1)

        # Decode
        mean_val = self.mean_val_net(net_input)
        cov_val = self.cov_val_net(net_input)

        # Process cov net prediction
        diag_cov_val, off_diag_cov_val = self._process_cov_net_output(cov_val)

        # Return
        return mean_val, diag_cov_val, off_diag_cov_val


class MCDecoder(Decoder):
    """Monte-Carlo decoder"""

    def decode(self,
               add_inputs: torch.Tensor,
               sampled_lat_var: torch.Tensor,
               variance_lat_var: torch.Tensor) \
            -> [torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode and compute target value's distribution

        Here, target value to be predicted is a 5th order tensor with axes:

        traj: this target value is on which trajectory
        aggr: based on how much aggregated context do we make this prediction?
        sample: latent variable samples for Monte-Carlo
        tar: this target value is on which target time?
        value: vector to be predicted

        Args:
            add_inputs: additional inputs, can be None
            sampled_lat_var: sampled latent variable
            variance_lat_var: variance of latent variable

        Returns:
            mean_val: mean of target value

            diag_cov_val: diagonal elements of Cholesky Decomposition of
            covariance of target value

            off_diag_cov_val: None, or off-diagonal elements of Cholesky
            Decomposition of covariance of target value
        """

        # Shape of sampled_lat_var:
        # [num_traj, num_agg, num_smp, dim_lat]
        #
        # Shape of variance_lat_var:
        # [num_traj, num_agg, num_smp, dim_lat]
        #
        # Shape of add_inputs:
        # [num_traj, num_time_pts, dim_add_in=1] if add_inputs not None
        #
        # Shape of mean_val:
        # [num_traj, num_agg, num_smp, num_time_pts, dim_val]
        #
        # Shape of diag_cov_val:
        # [num_traj, num_agg, num_smp, num_time_pts, dim_val]
        #
        # Shape of off_diag_cov_val:
        # [num_traj, num_agg, num_smp, num_time_pts,
        # (dim_val * (dim_val - 1) // 2)]

        # Dimension check
        assert sampled_lat_var.ndim == variance_lat_var.ndim == 4
        num_agg = sampled_lat_var.shape[1]
        num_smp = sampled_lat_var.shape[2]

        # Process add_inputs
        if add_inputs is not None:
            assert add_inputs.ndim == 3
            # Get dimensions
            num_time_pts = add_inputs.shape[1]
            # Add one axis (aggregation-wise batch dimension) to add_inputs
            add_inputs = util.add_expand_dim(add_inputs, [1, 2],
                                             [num_agg, num_smp])

        else:
            num_time_pts = 1

        # Parametrize variance
        variance_lat_var = util.to_log_space(variance_lat_var, lower_bound=None)

        # Add one axis (time-scale-wise batch dimension) to latent observation
        sampled_lat_var = util.add_expand_dim(sampled_lat_var,
                                              [-2], [num_time_pts])
        variance_lat_var = util.add_expand_dim(variance_lat_var,
                                               [-2], [num_time_pts])

        # Prepare input to decoder network
        mean_net_input = sampled_lat_var
        cov_net_input = variance_lat_var
        if add_inputs is not None:
            mean_net_input = torch.cat((add_inputs, sampled_lat_var), dim=-1)
            cov_net_input = torch.cat((add_inputs, variance_lat_var), dim=-1)

        # Decode
        mean_val = self.mean_val_net(mean_net_input)
        cov_val = self.cov_val_net(cov_net_input)

        # Process cov net prediction
        diag_cov_val, off_diag_cov_val = self._process_cov_net_output(cov_val)

        # Return
        return mean_val, diag_cov_val, off_diag_cov_val


class DecoderFactory:

    @staticmethod
    def get_decoder(decoder_type: str, **kwargs):
        return eval(decoder_type + "(**kwargs)")
