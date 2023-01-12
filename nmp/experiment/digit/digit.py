import random

import numpy as np
import torch

from mp_pytorch.mp import MPFactory
from nmp import get_data_loaders_and_normalizer
from nmp import nll_loss
from nmp import select_ctx_pred_pts
from nmp import util
from nmp.aggregator import AggregatorFactory
from nmp.data_process import NormProcess
from nmp.decoder import DecoderFactory
from nmp.encoder import EncoderFactory
from nmp.net import MPNet
from nmp.others.ellipses_noise import EllipseNoiseTransform


class OneDigit:
    def __init__(self, cfg):
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        # Net
        self.encoder_dict = EncoderFactory.get_encoders(**cfg["encoders"])
        self.aggregator = AggregatorFactory.get_aggregator(cfg["aggregator"]["type"],
                                                           **cfg["aggregator"]["args"])
        self.decoder = DecoderFactory.get_decoder(cfg["decoder"]["type"],
                                                  **cfg["decoder"]["args"])
        self.net = MPNet(self.encoder_dict, self.aggregator, self.decoder)

        # Dataset and Dataloader
        dataset = util.load_npz_dataset(cfg["dataset"]["name"])
        self.train_loader, self.vali_loader, self.test_loader, self.normalizer \
            = get_data_loaders_and_normalizer(dataset, **cfg["dataset"],
                                              seed=cfg["seed"])

        # Data assignment config
        self.assign_config = cfg["assign_config"]

        # Reconstructor
        self.mp = MPFactory.init_mp(device="cuda", **cfg["mp"])

        # Optimizer
        self.optimizer = torch.optim.Adam(params=self.net.get_net_params(),
                                          lr=float(cfg["lr"]),
                                          weight_decay=float(cfg["wd"]))
        self.net_params = self.net.get_net_params()

        # Runtime noise
        self.runtime_noise = cfg.get("runtime_noise", False)

        # Denormalize
        self.denormalize = cfg.get("denormalize", True)

        # Zero start
        self.zero_start = cfg.get("zero_start", False)

    def compute_loss(self, batch):

        if self.runtime_noise:
            batch = self.add_runtime_noise(batch)

        _, pred_index = select_ctx_pred_pts(**self.assign_config)
        pred_pairs = torch.combinations(pred_index, 2)

        num_traj = batch["images"]["value"].shape[0]
        num_agg = batch["images"]["value"].shape[1] + 1
        num_pred_pairs = pred_pairs.shape[0]

        # Get encoder input
        num_total_ctx = batch["images"]["value"].shape[1]
        if num_total_ctx == 1:
            ctx = {"cnn": batch["images"]["value"]}
        elif num_total_ctx > 1:
            # Remove original img in noise case
            # Only use the first 3 noisy images
            ctx = {"cnn": batch["images"]["value"][:, :-1]}
            num_agg -= 1

        # Reconstructor input

        init_time = torch.zeros(num_traj, num_agg, num_pred_pairs)
        init_vel = torch.zeros(num_traj, num_agg, num_pred_pairs, self.mp.num_dof)
        times = util.add_expand_dim(batch["trajs"]["time"][:, pred_pairs],
                                    add_dim_indices=[1],
                                    add_dim_sizes=[num_agg])

        # Ground-truth
        gt = util.add_expand_dim(batch["trajs"]["value"][:, pred_pairs],
                                 add_dim_indices=[1], add_dim_sizes=[num_agg])
        # Switch the time and dof dimension
        gt = torch.einsum('...ji->...ij', gt)
        # Make the time and dof dimensions flat
        gt = gt.reshape(*gt.shape[:-2], -1)

        # Predict
        mean, diag, off_diag = self.net.predict(num_traj=num_traj,
                                                enc_inputs=ctx,
                                                dec_input=None)
        # Denormalize prediction
        if self.denormalize:
            mean, L = NormProcess.distribution_denormalize(self.normalizer,
                                                           "init_x_y_dmp_w_g",
                                                           mean, diag, off_diag)
        else:
            L = util.build_lower_matrix(diag, off_diag)

        # Split initial position and DMP weights
        start_point = mean[..., 0, :self.mp.num_dof]
        mean = mean[..., self.mp.num_dof:].squeeze(-2)
        L = L[..., self.mp.num_dof:, self.mp.num_dof:].squeeze(-3)
        assert mean.ndim == 3

        # Add dim of time group
        mean = util.add_expand_dim(data=mean,
                                   add_dim_indices=[-2],
                                   add_dim_sizes=[num_pred_pairs])
        L = util.add_expand_dim(data=L,
                                add_dim_indices=[-3],
                                add_dim_sizes=[num_pred_pairs])
        start_point = util.add_expand_dim(data=start_point,
                                          add_dim_indices=[-2],
                                          add_dim_sizes=[num_pred_pairs])

        # Reconstruct predicted trajectories
        if self.zero_start:
            init_pos = torch.zeros_like(start_point)
        else:
            init_pos = start_point
        self.mp.update_inputs(times=times, params=mean, params_L=L,
                                 init_time=init_time, init_pos=init_pos,
                                 init_vel=init_vel)
        traj_pos_mean = self.mp.get_traj_pos(flat_shape=True)

        if self.zero_start:
            start_point = util.add_expand_dim(start_point, [-1],
                                              [times.shape[-1]])
            traj_pos_mean += start_point.reshape(*start_point.shape[:-2], -1)

        traj_pos_L = torch.linalg.cholesky(self.mp.get_traj_pos_cov())

        # Loss
        loss = nll_loss(gt, traj_pos_mean, traj_pos_L)
        return loss

    @staticmethod
    def add_runtime_noise(batch):
        transform = EllipseNoiseTransform()
        batch["images"]["value"] = transform(batch["images"]["value"])
        return batch
