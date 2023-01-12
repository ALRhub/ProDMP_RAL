import random

import numpy as np
import torch
from cw2 import cluster_work
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger

from nmp import BehaviorCloningNet
from nmp import get_data_loaders_and_normalizer
from nmp import mse_loss
from nmp import util
from nmp.data_process import NormProcess
from nmp.encoder import EncoderFactory
from nmp.net import avg_batch_loss


class RobotPushBehaviorCloning(experiment.AbstractIterativeExperiment):
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
        self.net = BehaviorCloningNet(
            EncoderFactory.get_encoder(cfg["mlp_net"]["type"],
                                       **cfg["mlp_net"]["args"]))

        # Dataset and Dataloader
        dataset = util.load_npz_dataset(cfg["dataset"]["name"])
        self.train_loader, self.vali_loader, self.test_loader, self.normalizer \
            = get_data_loaders_and_normalizer(dataset, **cfg["dataset"],
                                              seed=cfg["seed"])

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

        # Get net input
        ctx_times = batch["box_robot_state"]["time"][..., None]
        ctx_values = batch["box_robot_state"]["value"]
        norm_ctx_dict = NormProcess.batch_normalize(self.normalizer,
                                                    {"box_robot_state":
                                                         {"time": ctx_times,
                                                          "value": ctx_values}})
        ctx_times = norm_ctx_dict["box_robot_state"]["time"]
        ctx_values = norm_ctx_dict["box_robot_state"]["value"]

        ctx = torch.cat([ctx_times, ctx_values[..., :-2]], dim=-1)
        # ctx = ctx_values[..., :-2]

        # Ground-truth, dof=2
        gt = batch["des_cart_pos_vel"]["value"][..., :2]

        # Predict
        pred_traj = self.net.predict(net_input=ctx)

        # Loss
        loss = mse_loss(gt, pred_traj)

        return loss


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(RobotPushBehaviorCloning)

    # Optional: Add loggers
    cw.add_logger(WandBLogger())
    cw.run()
