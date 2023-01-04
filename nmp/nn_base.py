"""
@brief:     Classes of Neural Network Bases
"""
import pickle as pkl
from typing import Callable
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import ModuleList
from torch.nn import functional as F

import nmp.util as util


def get_act_func(key: str) -> Optional[Callable]:
    func_dict = dict()
    func_dict["tanh"] = torch.tanh
    func_dict["relu"] = F.relu
    func_dict["leaky_relu"] = F.leaky_relu
    func_dict["softplus"] = F.softplus
    func_dict["None"] = None
    return func_dict[key]


class MLP(nn.Module):
    def __init__(self,
                 name: str,
                 dim_in: int,
                 dim_out: int,
                 hidden_layers: list,
                 act_func: str):
        """
        Multi-layer Perceptron Constructor

        Args:
            name: name of the MLP
            dim_in: dimension of the input
            dim_out: dimension of the output
            hidden_layers: a list containing hidden layers' dimensions
            act_func: activation function
        """

        super(MLP, self).__init__()

        self.mlp_name = name + "_mlp"

        # Initialize the MLP
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_layers = hidden_layers
        self.act_func_type = act_func
        self.act_func = get_act_func(act_func)

        # Create networks
        # Ugly but useful to distinguish networks in gradient watch
        # e.g. if self.mlp_name is "encoder_mlp"
        # Then below will lead to self.encoder_mlp = self._create_network()
        setattr(self, self.mlp_name, self._create_network())

    def _create_network(self):
        """
        Create MLP Network

        Returns:
        MLP Network
        """

        # Total layers (n+1) = hidden layers (n) + output layer (1)

        # Add first hidden layer
        mlp = ModuleList([nn.Linear(in_features=self.dim_in,
                                    out_features=self.hidden_layers[0])])

        # Add other hidden layers
        for i in range(1, len(self.hidden_layers)):
            mlp.append(nn.Linear(in_features=mlp[-1].out_features,
                                 out_features=self.hidden_layers[i]))

        # Add output layer
        mlp.append(nn.Linear(in_features=mlp[-1].out_features,
                             out_features=self.dim_out))

        return mlp

    def save(self, log_dir: str, epoch: int):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.mlp_name, epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "hidden_layers": self.hidden_layers,
                "act_func_type": self.act_func_type,
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir: str, epoch: int):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.mlp_name, epoch)

        # Check structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            assert self.dim_in == parameters["dim_in"] \
                   and self.dim_out == parameters["dim_out"] \
                   and self.hidden_layers == parameters["hidden_layers"] \
                   and self.act_func_type == parameters["act_func_type"], \
                "NN structure parameters do not match"

        # Load NN weights
        self.load_state_dict(torch.load(w_path))

    def forward(self, data):
        """
        Network forward function

        Args:
            data: input data

        Returns: MLP output

        """

        # Hidden layers (n) + output layer (1)
        mlp = eval("self." + self.mlp_name)
        for i in range(len(self.hidden_layers)):
            data = self.act_func(mlp[i](data))
        data = mlp[-1](data)

        # Return
        return data


class CNNMLP(nn.Module):
    def __init__(self,
                 name: str,
                 image_size: list,
                 kernel_size: int,
                 num_cnn: int,
                 cnn_channels: list,
                 hidden_layers: list,
                 dim_out: int,
                 act_func: str):
        """
        CNN, MLP constructor

        Args:
            name: name of the MLP
            image_size: w and h of input images size
            kernel_size: size of cnn kernel
            num_cnn: number of cnn layers
            cnn_channels: a list containing cnn in and out channels
            hidden_layers: a list containing hidden layers' dimensions
            dim_out: dimension of the output
            act_func: activation function
        """
        super(CNNMLP, self).__init__()

        self.name = name
        self.cnn_mlp_name = name + "_cnn_mlp"

        self.image_size = image_size
        self.kernel_size = kernel_size
        assert num_cnn + 1 == len(cnn_channels)
        self.num_cnn = num_cnn
        self.cnn_channels = cnn_channels
        self.dim_in = self.get_mlp_dim_in()
        self.hidden_layers = hidden_layers
        self.dim_out = dim_out
        self.act_func_type = act_func
        self.act_func = get_act_func(act_func)

        # Initialize the CNN and MLP
        setattr(self, self.cnn_mlp_name, self._create_network())

    def get_mlp_dim_in(self) -> int:
        """
        Compute the input size of mlp layers
        Returns:
            dim_in
        """
        image_out_size = \
            [util.image_output_size(size=s,
                                    num_cnn=self.num_cnn,
                                    cnn_kernel_size=self.kernel_size)
             for s in self.image_size]
        # dim_in = channel * w * h
        dim_in = self.cnn_channels[-1]
        for s in image_out_size:
            dim_in *= s
        return dim_in

    def _create_network(self):
        """
        Create CNNs and MLP

        Returns: cnn_mlp
        """
        cnn_mlp = ModuleList()
        for i in range(self.num_cnn):
            in_channel = self.cnn_channels[i]
            out_channel = self.cnn_channels[i + 1]
            cnn_mlp.append(nn.Conv2d(in_channel, out_channel, self.kernel_size))

        # Initialize the MLP
        cnn_mlp.append(MLP(name=self.name,
                           dim_in=self.dim_in,
                           dim_out=self.dim_out,
                           hidden_layers=self.hidden_layers,
                           act_func=self.act_func_type))
        return cnn_mlp

    def save(self, log_dir: str, epoch: int):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir,
                                                self.cnn_mlp_name,
                                                epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "num_cnn": self.num_cnn,
                "cnn_channels": self.cnn_channels,
                "kernel_size": self.kernel_size,
                "image_size": self.image_size,
                "dim_in": self.dim_in,
                "hidden_layers": self.hidden_layers,
                "dim_out": self.dim_out,
                "act_func_type": self.act_func_type
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir: str, epoch: int):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir,
                                                self.cnn_mlp_name,
                                                epoch)

        # Load structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            assert self.num_cnn == parameters["num_cnn"] \
                   and self.cnn_channels == parameters["cnn_channels"] \
                   and self.kernel_size == parameters["kernel_size"] \
                   and self.image_size == parameters["image_size"] \
                   and self.dim_in == parameters["dim_in"] \
                   and self.hidden_layers == parameters["hidden_layers"] \
                   and self.dim_out == parameters["dim_out"] \
                   and self.act_func_type == parameters["act_func_type"], \
                "NN structure parameters do not match"

        # Load NN weights
        self.load_state_dict(torch.load(w_path))

    def forward(self, data):
        """
        Network forward function

        Args:
            data: input data

        Returns: CNN + MLP output
        """

        # Reshape images batch to [num_traj * num_obs, C, H, W]
        num_traj, num_obs = data.shape[:2]
        data = data.reshape(-1, *data.shape[2:])

        cnns = eval("self." + self.cnn_mlp_name)[:-1]
        mlp = eval("self." + self.cnn_mlp_name)[-1]

        # Forward pass in CNNs
        # todo, check if dropout is critical to training case
        for i in range(len(cnns)-1):
            data = self.act_func(F.max_pool2d(cnns[i](data), 2))
        data = self.act_func(F.max_pool2d(
            F.dropout2d(cnns[-1](data), training=self.training), 2))

        # Flatten
        data = data.view(num_traj, num_obs, self.dim_in)

        # Forward pass in MLPs
        data = mlp(data)

        # Return
        return data


class GruRnn(nn.Module):
    def __init__(self,
                 name: str,
                 dim_in: int,
                 dim_out: int,
                 num_layers: int,
                 seed: int):
        """
        Gated Recurrent Unit of RNN

        Args:
            name: name of the GRU
            dim_in: dimension of the input
            dim_out: dimension of the output
            num_layers: number of hidden layers
            seed: seed for random behaviours
        """

        super(GruRnn, self).__init__()

        self.name = name
        self.gru_name = name + "_gru"

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.seed = seed

        # Create networks
        setattr(self, self.gru_name, self._create_network())

    def _create_network(self):
        """
        Create GRU Network

        Returns:
        GRU Network
        """
        gru = nn.GRU(input_size=self.dim_in,
                     hidden_size=self.dim_out,
                     num_layers=self.num_layers,
                     batch_first=True)

        return gru

    def save(self, log_dir: str, epoch: int):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.gru_name, epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "num_layers": self.num_layers,
                "seed": self.seed,
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir: str, epoch: int):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.gru_name, epoch)

        # Load structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            assert self.dim_in == parameters["dim_in"] \
                   and self.dim_out == parameters["dim_out"] \
                   and self.num_layers == parameters["num_layers"] \
                   and self.seed == parameters["seed"]
            "NN structure parameters do not match"

        # Load NN weights
        self.load_state_dict(torch.load(w_path))

    def forward(self, input_data):
        """
        Network forward function

        Args:
            input_data: input data

        Returns: GRU output

        """
        data = input_data

        gru = eval("self." + self.gru_name)
        data = gru(data)

        # Return
        return data
