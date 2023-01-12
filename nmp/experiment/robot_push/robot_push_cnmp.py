import random

import numpy as np
import torch
from cw2 import cluster_work
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger
from nmp import get_data_loaders_and_normalizer
from nmp import nll_loss
from nmp import util
from nmp.aggregator import AggregatorFactory
from nmp.data_process import NormProcess
from nmp.decoder import DecoderFactory
from nmp.encoder import EncoderFactory
from nmp.net import MPNet
from nmp.net import avg_batch_loss


class RobotPushCNMP(experiment.AbstractIterativeExperiment):
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
        pred_index = torch.arange(ctx_index[-1] + 1, pred_index_max,
                                  dtype=torch.long)

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

        num_traj = batch["box_robot_state"]["value"].shape[0]

        times = (batch["des_cart_pos_vel"]["time"] - time_ctx_last[:, None])[:,
                pred_index]

        # Ground-truth, dof=2
        gt = batch["des_cart_pos_vel"]["value"][:, pred_index, :2]

        # Predict
        mean, diag, off_diag = self.net.predict(num_traj=num_traj,
                                                enc_inputs=ctx,
                                                dec_input=times[..., None])
        assert off_diag is None

        # Denormalize prediction
        L = util.build_lower_matrix(diag, off_diag)

        mean = mean.squeeze(1)
        L = L.squeeze(1)

        assert mean.ndim == 3

        # Loss
        loss = nll_loss(gt, mean, L)
        return loss


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(RobotPushCNMP)

    # Optional: Add loggers
    cw.add_logger(WandBLogger())
    cw.run()
