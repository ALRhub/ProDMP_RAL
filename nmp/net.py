"""
@brief:     Network class
"""
from typing import Callable
from typing import Tuple
from typing import Union

import torch.optim
from torch import nn

from nmp import CNMPEncoder
from nmp import util
from nmp.aggregator import *
from nmp.decoder import Decoder
from torch.utils.data import DataLoader


class MPNet:
    def __init__(self, encoder_dict: dict,
                 aggregator: Union[MeanAggregator, BayesianAggregator],
                 decoder: Decoder):
        self.encoder_dict = encoder_dict
        self.aggregator = aggregator
        self.decoder = decoder

    def get_net_params(self):
        """
        Get parameters to be optimized
        Returns:
             Tuple of parameters of neural networks

        """
        # Decoder
        parameters = self.decoder.parameters

        # Encoders
        for encoder in self.encoder_dict.values():
            parameters += encoder.parameters

        return (parameters)

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save parameters
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None

        """

        # Encoder
        for encoder in self.encoder_dict.values():
            encoder.save_weights(log_dir, epoch)

        # Decoder
        self.decoder.save_weights(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        """
        Load parameters
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Encoder
        for encoder in self.encoder_dict.values():
            encoder.load_weights(log_dir, epoch)

        # Decoder
        self.decoder.load_weights(log_dir, epoch)

    def predict(self, num_traj: int, enc_inputs: dict,
                dec_input: Optional[torch.Tensor],
                **kwargs):
        """
        Predict using the network

        Args:
            num_traj: batch size of the number of trajectories
            enc_inputs: input of encoder
            dec_input: output of encoder
            **kwargs: keyword arguments

        Returns:
            mean, variance of the predicted values
        """
        # Reset aggregator
        self.aggregator.reset(num_traj=num_traj)

        # Loop over all encoders
        for encoder_name, encoder in self.encoder_dict.items():
            # Get data assigned to it
            encoder_input = enc_inputs[encoder_name]

            # Encode, make result in tuple and store
            lat_obs = util.make_iterable(encoder.encode(encoder_input))

            # Aggregate
            self.aggregator.aggregate(*lat_obs)

        # Get latent variable
        index = None if self.aggregator.multiple_steps else -1
        lat_var = util.make_iterable(self.aggregator.get_agg_state(index=index))

        # Sample latent variables if necessary, todo remove this
        num_mc_smp = kwargs.get("num_mc_smp", 0)
        if num_mc_smp != 0:
            lat_var = self.sample_latent_variable(num_mc_smp, *lat_var)

        # Decode
        mean, diag, off_diag = self.decoder.decode(dec_input, *lat_var)

        # Return
        return mean, diag, off_diag

    @staticmethod
    def sample_latent_variable(num_mc_smp: int, lat_mean, lat_var):
        """
        Sample latent variable for Monte-Carlo approximation

        Args:
            num_mc_smp: num of Monte-Carlo samples when necessary
            lat_mean: mean of latent variable
            lat_var: variance of latent variable

        Returns:
            sampled latent variable, shape:
            [num_traj, num_agg, num_smp, dim_lat]

            variance of latent variable, shape:
            [num_traj, num_agg, num_smp, dim_lat]
        """
        gaussian = torch.distributions.normal.Normal(loc=lat_mean,
                                                     scale=lat_var,
                                                     validate_args=False)

        mc_smp = torch.einsum('kij...->ijk...', gaussian.rsample([num_mc_smp]))
        # lat_var = lat_var[..., None, :]
        lat_var = util.add_expand_dim(lat_var, [-2], [num_mc_smp])
        return mc_smp, lat_var


def avg_batch_loss(data_loader: DataLoader,
                   loss_func: Callable,
                   optimizer: Optional[torch.optim.Optimizer],
                   params: Optional[Tuple[torch.Tensor]],
                   max_norm: Optional[int] = 2):
    loss = 0.0
    num_batch = 0

    gradient_norm_list = []
    for batch in data_loader:
        num_batch += 1

        if optimizer is not None:
            # Training
            batch_loss = loss_func(batch)

            # Optimize
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            if max_norm:
                total_norm = 0
                parameters = [p for p in params if
                              p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norm_list.append(total_norm)

                nn.utils.clip_grad_norm_(params, max_norm=max_norm, norm_type=2)

            optimizer.step()

        else:
            # Validation or Testing
            with torch.no_grad():
                batch_loss = loss_func(batch)

        # Sum up batch loss
        loss += batch_loss.item()

    # Compute average batch loss
    avg_loss = loss / num_batch

    if optimizer is not None:
        print(gradient_norm_list)

    # Return
    return avg_loss


class BehaviorCloningNet:
    def __init__(self, mlp_net: CNMPEncoder):
        # Here CNMP encoder can be used as a Behavior cloning net directly.
        self.mlp_net = mlp_net

    def get_net_params(self):
        """
        Get parameters to be optimized
        Returns:
             Tuple of parameters of neural networks

        """
        # Decoder
        parameters = self.mlp_net.parameters
        return (parameters)

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save parameters
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None

        """
        self.mlp_net.save_weights(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        """
        Load parameters
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.mlp_net.load_weights(log_dir, epoch)

    def predict(self, net_input: torch.Tensor):
        """
        Predict using the network

        Args:
            net_input: input of network

        Returns:
            predicted values
        """

        return self.mlp_net.encode(net_input)


