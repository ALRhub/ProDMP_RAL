import random

import numpy as np
import torch
from cw2 import cluster_work
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger
from mp_pytorch.mp import MPFactory
from nmp import get_data_loaders_and_normalizer
from nmp import nll_loss
from nmp import util
from nmp.aggregator import AggregatorFactory
from nmp.data_process import NormProcess
from nmp.decoder import DecoderFactory
from nmp.encoder import EncoderFactory
from nmp.net import MPNet
from nmp.net import avg_batch_loss


class RobotPushProMP(experiment.AbstractIterativeExperiment):
    def initialize(self, cw_config: dict,
                   rep: int, logger: cw_logging.LoggerArray) -> None:
        # Device
        util.use_cuda()

        # Random seed
        cfg = cw_config["params"]
        cfg["seed"] = rep
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        # Net
        self.encoder_dict = EncoderFactory.get_encoders(**cfg["encoders"])
        self.aggregator = AggregatorFactory.get_aggregator(
            cfg["aggregator"]["type"],
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

        # Log interval
        self.vali_interval = cfg["vali_log_interval"]
        self.save_model_interval = cfg["save_model_interval"]

        # Logger
        self.save_model_dir = cw_config.get("save_model_dir", None)
        if self.save_model_dir:
            util.remove_file_dir(self.save_model_dir)
            util.mkdir(self.save_model_dir, overwrite=True)

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        max_norm = cw_config["params"].get("max_norm", None)
        train_loss = \
            avg_batch_loss(self.train_loader, self.compute_loss, self.optimizer,
                           self.net_params, max_norm)

        if n % self.vali_interval == 0:
            vali_loss = avg_batch_loss(self.vali_loader, self.compute_loss,
                                       None, None, None)
        else:
            vali_loss = None
        print(n)

        return {"train_loss": train_loss, "vali_loss": vali_loss}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        if self.save_model_dir and ((n + 1) % self.save_model_interval == 0
                                    or (n + 1) == cw_config["iterations"]):
            self.net.save_weights(log_dir=self.save_model_dir,
                                  epoch=n + 1)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None,
                 crash: bool = False):
        torch.cuda.empty_cache()

    def compute_loss(self, batch):
        # Choose the point to start obs
        num_ctx_min = self.assign_config["num_ctx_min"]
        num_ctx_max = self.assign_config["num_ctx_max"]
        num_ctx = torch.randint(num_ctx_min, num_ctx_max + 1, [])
        pred_range_min = self.assign_config["pred_range_min"]
        pred_range_max = self.assign_config["pred_range_max"]
        pred_range = torch.randint(pred_range_min, pred_range_max + 1, [])
        # pred_step = pred_range // 10 + 1
        # num_total = num_ctx + pred_range.item()
        start_obs_idx_max = 301 - num_ctx - 10
        start_obs_idx = torch.randint(0, start_obs_idx_max + 1, [],
                                      dtype=torch.long)
        ctx_index = torch.arange(start_obs_idx, start_obs_idx + num_ctx, step=1,
                                 dtype=torch.long)
        pred_index_max = min(300, ctx_index[-1] + 1 + pred_range)
        pred_index = torch.arange(ctx_index[-1], pred_index_max,
                                  dtype=torch.long)

        pred_pairs = torch.combinations(pred_index, 2).long()

        # Get encoder input
        time_ctx_last = batch["box_robot_state"]["time"][:, ctx_index[-1]]
        ctx_times = (batch["box_robot_state"]["time"]
                     - time_ctx_last[:, None])[:, ctx_index][..., None]
        ctx_values = batch["box_robot_state"]["value"][:, ctx_index]
        norm_ctx_dict = NormProcess.batch_normalize(self.normalizer,
                                                    {"box_robot_state":
                                                         {"time": ctx_times,
                                                          "value": ctx_values}})
        # ctx_times = norm_ctx_dict["box_robot_state"]["time"]
        ctx_values = norm_ctx_dict["box_robot_state"]["value"]

        ctx = {"ctx": torch.cat([ctx_times, ctx_values], dim=-1)}

        # Reconstructor input
        num_traj = batch["box_robot_state"]["value"].shape[0]
        # num_agg = len(ctx_index) + 1
        num_agg = 1
        num_pred_pairs = pred_pairs.shape[0]

        # init_time = batch["des_cart_pos_vel"]["time"][:, ctx_index[-1]]
        init_time = torch.zeros([num_traj])
        init_time = util.add_expand_dim(init_time, [1, -1],
                                        [num_agg, num_pred_pairs])
        init_pos = batch["des_cart_pos_vel"]["value"][:, ctx_index[-1],
                   :self.mp.num_dof]
        init_pos = util.add_expand_dim(init_pos, [1, -2],
                                       [num_agg, num_pred_pairs])

        init_vel = batch["des_cart_pos_vel"]["value"][:, ctx_index[-1],
                   self.mp.num_dof:]
        init_vel = util.add_expand_dim(init_vel, [1, -2],
                                       [num_agg, num_pred_pairs])

        times = util.add_expand_dim(
            (batch["des_cart_pos_vel"]["time"] - time_ctx_last[:, None])[:,
            pred_pairs],
            add_dim_indices=[1], add_dim_sizes=[num_agg])

        # Ground-truth
        gt = util.add_expand_dim(
            batch["des_cart_pos_vel"]["value"][:, pred_pairs, :self.mp.num_dof],
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
        # mean, L = NormProcess.distribution_denormalize(self.normalizer,
        #                                                "idmp",
        #                                                mean, diag, off_diag)
        L = util.build_lower_matrix(diag, off_diag)

        mean = mean.squeeze(-2)
        L = L.squeeze(-3)

        assert mean.ndim == 3

        # Add dim of time group
        mean = util.add_expand_dim(data=mean,
                                   add_dim_indices=[-2],
                                   add_dim_sizes=[num_pred_pairs])
        L = util.add_expand_dim(data=L,
                                add_dim_indices=[-3],
                                add_dim_sizes=[num_pred_pairs])

        # Reconstruct predicted trajectories
        self.mp.update_inputs(times=times, params=mean, params_L=L,
                              init_time=init_time, init_pos=init_pos,
                              init_vel=init_vel)
        traj_pos_mean = self.mp.get_traj_pos(flat_shape=True)
        traj_pos_L = torch.linalg.cholesky(self.mp.get_traj_pos_cov())

        # Loss
        loss = nll_loss(gt, traj_pos_mean, traj_pos_L)
        return loss


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(RobotPushProMP)

    # Optional: Add loggers
    cw.add_logger(WandBLogger())
    cw.run()
