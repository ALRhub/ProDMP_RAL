from cw2 import cluster_work
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger

from nmp import util
from nmp.experiment.digit.digit import OneDigit
from nmp.net import avg_batch_loss


class OneDigitCW(experiment.AbstractIterativeExperiment):
    def initialize(self, cw_config: dict,
                   rep: int, logger: cw_logging.LoggerArray) -> None:
        # Device
        util.use_cuda()

        # Random seed
        cfg = cw_config["params"]

        self.exp = OneDigit(cfg)

        # Log interval
        self.vali_interval = cfg["vali_log_interval"]
        self.save_model_interval = cfg["save_model_interval"]

        # Logger
        self.save_model_dir = cw_config.get("save_model_dir", None)
        if self.save_model_dir:
            util.remove_file_dir(self.save_model_dir)
            util.mkdir(self.save_model_dir, overwrite=True)
            # Save configuration
            util.dump_config(dict(cfg), "config", self.save_model_dir)

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        max_norm = cw_config["params"].get("max_norm", None)
        train_loss = avg_batch_loss(self.exp.train_loader,
                                    self.exp.compute_loss, self.exp.optimizer,
                                    self.exp.net_params, max_norm)

        if n % self.vali_interval == 0:
            vali_loss = avg_batch_loss(self.exp.vali_loader,
                                       self.exp.compute_loss, None, None, None)
        else:
            vali_loss = None
        print(n)

        return {"train_loss": train_loss, "vali_loss": vali_loss}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        if self.save_model_dir and ((n + 1) % self.save_model_interval == 0
                                    or (n + 1) == cw_config["iterations"]):
            self.exp.net.save_weights(log_dir=self.save_model_dir,
                                      epoch=n + 1)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None,
                 crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(OneDigitCW)
    cw.add_logger(WandBLogger())
    cw.run()
