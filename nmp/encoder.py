"""
@brief:     Encoder classes in PyTorch
"""
from abc import ABC
from abc import abstractmethod

from nmp.nn_base import CNNMLP
from nmp.nn_base import MLP
from nmp.util import mlp_arch_3_params
from nmp.util import to_softplus_space


class ProNMPEncoder(ABC):

    def __init__(self, **kwargs):
        """
        NMP encoder constructor
        Args:
            **kwargs: Encoder configuration
        """

        # Encoders
        self.lat_mean_net = None
        self.lat_var_net = None
        self._create_network(**kwargs)

    @abstractmethod
    def _create_network(self, *args, **kwargs):
        """
        Create encoder with given configuration

        Returns:
            None
        """
        pass

    @property
    def network(self):
        """
        Return encoder networks

        Returns:
        """
        return self.lat_mean_net, self.lat_var_net

    @property
    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.lat_mean_net.parameters()) + \
               list(self.lat_var_net.parameters())

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.lat_mean_net.save(log_dir, epoch)
        self.lat_var_net.save(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.lat_mean_net.load(log_dir, epoch)
        self.lat_var_net.load(log_dir, epoch)

    @abstractmethod
    def encode(self, *args, **kwargs):
        """
        Encode observations

        Returns:
            lat_obs: latent observations
            var_lat_obs: variance of latent observations
        """
        pass


class ProNMPEncoderMlp(ProNMPEncoder):
    def _create_network(self, **kwargs):
        """
        Create encoder with given configuration

        Returns:
            None
        """
        # MLP configuration
        self.name: str = kwargs["name"]
        self.dim_obs: int = kwargs["dim_obs"]
        self.dim_lat_obs: int = kwargs["dim_lat"]

        self.obs_hidden: dict = kwargs["obs_hidden"]
        self.unc_hidden: dict = kwargs["unc_hidden"]

        self.act_func: str = kwargs["act_func"]

        # Two separate latent observation encoders
        # lat_mean_net + lat_var_net
        self.lat_mean_net = MLP(name="ProNMPEncoder_lat_mean_" + self.name,
                                dim_in=self.dim_obs,
                                dim_out=self.dim_lat_obs,
                                hidden_layers=
                                mlp_arch_3_params(**self.obs_hidden),
                                act_func=self.act_func)

        self.lat_var_net = \
            MLP(name="ProNMPEncoder_lat_var_" + self.name,
                dim_in=self.dim_obs,
                dim_out=self.dim_lat_obs,
                hidden_layers=mlp_arch_3_params(**self.unc_hidden),
                act_func=self.act_func)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            lat_obs: latent observations
            var_lat_obs: variance of latent observations
        """

        # Shape of obs:
        # [num_traj, num_obs, dim_obs],
        #
        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]
        #
        # Shape of var_lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Check input shapes
        assert obs.ndim == 3

        # Encode
        return self.lat_mean_net(obs), \
               to_softplus_space(self.lat_var_net(obs), lower_bound=None)


class ProNMPEncoderCnnMlp(ProNMPEncoder):
    def _create_network(self, **kwargs):
        """
        Create encoder with given configuration

        Returns:
            None
        """
        # configuration
        self.name = kwargs["name"]
        self.image_size = kwargs["image_size"]
        self.kernel_size = kwargs["kernel_size"]
        self.num_cnn = kwargs["num_cnn"]
        self.cnn_channels = kwargs["cnn_channels"]
        self.dim_lat_obs: int = kwargs["dim_lat"]
        self.obs_hidden: dict = kwargs["obs_hidden"]
        self.unc_hidden: dict = kwargs["unc_hidden"]
        self.act_func = kwargs["act_func"]

        # Two separate latent observation encoders
        # lat_mean_net + lat_var_net

        self.lat_mean_net = CNNMLP(name="ProNMPEncoder_lat_mean_" + self.name,
                                   image_size=self.image_size,
                                   kernel_size=self.kernel_size,
                                   num_cnn=self.num_cnn,
                                   cnn_channels=self.cnn_channels,
                                   hidden_layers=
                                   mlp_arch_3_params(**self.obs_hidden),
                                   dim_out=self.dim_lat_obs,
                                   act_func=self.act_func)

        self.lat_var_net = CNNMLP(name="ProNMPEncoder_lat_var_" + self.name,
                                  image_size=self.image_size,
                                  kernel_size=self.kernel_size,
                                  num_cnn=self.num_cnn,
                                  cnn_channels=self.cnn_channels,
                                  hidden_layers=
                                  mlp_arch_3_params(**self.unc_hidden),
                                  dim_out=self.dim_lat_obs,
                                  act_func=self.act_func)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            lat_obs: latent observations
            var_lat_obs: variance of latent observations
        """

        # Shape of obs:
        # [num_traj, num_obs, C, H, W],
        #
        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]
        #
        # Shape of var_lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Check input shapes
        assert obs.ndim == 5

        # Encode
        return self.lat_mean_net(obs), \
               to_softplus_space(self.lat_var_net(obs), lower_bound=None)


class CNMPEncoder(ABC):

    def __init__(self, **kwargs):
        """
        CNMP encoder constructor

        Args:
            **kwargs: Encoder configuration
        """

        # Encoder
        self.lat_obs_net = None
        self._create_network(**kwargs)

    @abstractmethod
    def _create_network(self, *args, **kwargs):
        """
        Create encoder network with given configuration

        Returns:
            None
        """
        pass

    @property
    def network(self):
        """
        Return encoder network

        Returns:
        """
        return self.lat_obs_net

    @property
    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """

        return list(self.lat_obs_net.parameters())

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        self.lat_obs_net.save(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.lat_obs_net.load(log_dir, epoch)

    @abstractmethod
    def encode(self, *args, **kwargs):
        """
        Encode observations

        Returns:
            lat_obs: latent observations
        """
        pass


class CNMPEncoderMlp(CNMPEncoder):

    def _create_network(self, **kwargs):
        """
        Create encoder network with given configuration

        Returns:
            None
        """

        # MLP configuration
        self.name: str = kwargs["name"]
        self.dim_obs: int = kwargs["dim_obs"]
        self.dim_lat_obs: int = kwargs["dim_lat"]
        self.obs_hidden: dict = kwargs["obs_hidden"]
        self.act_func: str = kwargs["act_func"]

        self.lat_obs_net = MLP(name="CNMPEncoder_lat_obs_" + self.name,
                               dim_in=self.dim_obs,
                               dim_out=self.dim_lat_obs,
                               hidden_layers=
                               mlp_arch_3_params(**self.obs_hidden),
                               act_func=self.act_func)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            lat_obs: latent observations
        """

        # Shape of obs:
        # [num_traj, num_obs, dim_obs],
        #
        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Check input shapes
        assert obs.ndim == 3

        # Encode
        return self.lat_obs_net(obs)


class CNMPEncoderCnnMlp(CNMPEncoder):

    def _create_network(self, **kwargs):
        """
        Create encoder with given configuration

        Returns:
            None
        """

        # configuration
        self.name = kwargs["name"]
        self.image_size = kwargs["image_size"]
        self.kernel_size = kwargs["kernel_size"]
        self.num_cnn = kwargs["num_cnn"]
        self.cnn_channels = kwargs["cnn_channels"]
        self.dim_lat_obs: int = kwargs["dim_lat"]
        self.obs_hidden: dict = kwargs["obs_hidden"]
        self.act_func = kwargs["act_func"]

        # lat_obs_net
        self.lat_obs_net = CNNMLP(name="CNMPEncoder_lat_obs_" + self.name,
                                  image_size=self.image_size,
                                  kernel_size=self.kernel_size,
                                  num_cnn=self.num_cnn,
                                  cnn_channels=self.cnn_channels,
                                  hidden_layers=
                                  mlp_arch_3_params(**self.obs_hidden),
                                  dim_out=self.dim_lat_obs,
                                  act_func=self.act_func)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            lat_obs: latent observations
        """

        # Shape of obs:
        # [num_traj, num_obs, C, H, W],
        #
        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]
        #
        # Shape of var_lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Check input shapes
        assert obs.ndim == 5

        # Encode
        return self.lat_obs_net(obs)


class EncoderFactory:

    @staticmethod
    def get_encoder(encoder_type: str, **kwargs):
        return eval(encoder_type + "(**kwargs)")

    @staticmethod
    def get_encoders(**config) -> dict:
        encoder_dict = dict()
        for encoder_name, encoder_info in config.items():
            encoder_info["args"]["name"] = encoder_name
            encoder = EncoderFactory.get_encoder(encoder_info["type"],
                                                 **(encoder_info["args"]))
            encoder_dict[encoder_name] = encoder
        return encoder_dict
