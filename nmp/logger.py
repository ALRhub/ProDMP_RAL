"""
@brief:     Logger
"""
import csv
import os

import matplotlib.pyplot as plt
import wandb

import nmp.util as util


class WandbLogger:
    def __init__(self, config):
        """
        Initialize wandb logger
        Args:
            config: config file of current task
        """
        self.project_name = config["logger"]["log_name"]
        entity = config["logger"].get("entity")
        group = config["logger"].get("group")
        self._initialize_log_dir()
        self._run = wandb.init(project=self.project_name, entity=entity,
                               group=group, config=config)

    def _initialize_log_dir(self):
        """
        Clean and initialize local log directory
        Returns:
            True if successfully cleaned
        """
        # Clean old log
        util.remove_file_dir(self.log_dataset_dir)
        util.remove_file_dir(self.log_model_dir)
        util.remove_file_dir(self.log_dir)

        # Make new directory
        os.makedirs(self.log_dir)
        os.makedirs(self.log_dataset_dir)
        os.makedirs(self.log_model_dir)

    @property
    def config(self):
        """
        Log configuration file

        Returns:
            synchronized config from wandb server

        """
        return wandb.config

    @property
    def log_dir(self):
        """
        Get local log saving directory
        Returns:
            log directory
        """
        if not hasattr(self, "_log_dir"):
            self._log_dir = util.make_log_dir_with_time_stamp(self.project_name)

        return self._log_dir

    @property
    def log_dataset_dir(self):
        """
        Get downloaded logged dataset directory
        Returns:
            logged dataset directory
        """
        return os.path.join(self.log_dir, "dataset")

    @property
    def log_model_dir(self):
        """
        Get downloaded logged model directory
        Returns:
            logged model directory
        """
        return os.path.join(self.log_dir, "model")

    def log_dataset(self,
                    dataset_name,
                    pd_df_dict: dict):
        """
        Log raw dataset to Artifact

        Args:
            dataset_name: Name of dataset
            pd_df_dict: dictionary of train, validate and test sets

        Returns:
            None
        """

        # Initialize wandb Artifact
        raw_data = wandb.Artifact(name=dataset_name + "_dataset",
                                  type="dataset",
                                  description="dataset")

        # Save DataFrames in Artifact
        for key, value in pd_df_dict.items():
            for index, pd_df in enumerate(value):
                with raw_data.new_file(key + "_{}.csv".format(index),
                                       mode="w") as file:
                    file.write(pd_df.to_csv(path_or_buf=None,
                                            index=False,
                                            quoting=csv.QUOTE_ALL))

        # Log Artifact
        self._run.log_artifact(raw_data)

    def log_info(self,
                 epoch,
                 key,
                 value):
        self._run.log({"Epoch": epoch,
                       key: value})

    def log_model(self,
                  finished: bool = False):
        """
        Log model into Artifact

        Args:
            finished: True if current training is finished, this will clean
            the old model version without any special aliass

        Returns:
            None
        """
        # Initialize wandb artifact
        model_artifact = wandb.Artifact(name="model", type="model")

        # Get all file names in log dir
        file_names = util.get_file_names_in_directory(self.log_model_dir)

        # Add files into artifact
        for file in file_names:
            path = os.path.join(self.log_model_dir, file)
            model_artifact.add_file(path)

        if finished:
            aliases = ["latest",
                       "finished-{}".format(util.get_formatted_date_time())]
        else:
            aliases = ["latest"]

        # Log and upload
        self._run.log_artifact(model_artifact, aliases=aliases)

        if finished:
            self.delete_useless_model()

    def delete_useless_model(self):
        """
        Delete useless models in WandB server
        Returns:
            None

        """
        api = wandb.Api()

        artifact_type = "model"
        artifact_name = "{}/{}/model".format(self._run.entity,
                                             self._run.project)

        for version in api.artifact_versions(artifact_type, artifact_name):
            # Clean up all versions that don't have an alias such as 'latest'.
            if len(version.aliases) == 0:
                version.delete()

    def load_model(self,
                   model_api: str):
        """
        Load model from Artifact

        model_api: the string for load the model if init_epoch is not zero

        Returns:
            model_dir: Model's directory

        """
        model_api = "self._" + model_api[11:]
        artifact = eval(model_api)
        artifact.download(root=self.log_model_dir)
        file_names = util.get_file_names_in_directory(self.log_model_dir)
        file_names.sort()
        util.print_line_title(title="Download model files from WandB")
        for file in file_names:
            print(file)
        return self.log_model_dir

    def watch_networks(self,
                       networks,
                       log_freq):
        """
        Watch Neural Network weights and gradients
        Args:
            networks: network to being watched
            log_freq: frequency for logging

        Returns:
            None

        """
        for idx, net in enumerate(networks):
            self._run.watch(net,
                            log="all",
                            log_freq=log_freq,
                            idx=idx)

    def log_figure(self,
                   figure_obj: plt.Figure,
                   figure_name: str = "Unnamed Figure"):
        """
        Log figure
        Args:
            figure_obj: Matplotlib Figure object
            figure_name: name of the figure

        Returns:
            None

        """
        self._run.log({figure_name: wandb.Image(figure_obj)})

    def log_video(self,
                  path_to_video: str,
                  video_name: str = "Unnamed Video"):
        """
        Log video
        Args:
            path_to_video: path where the video is stored
            video_name: name of the video

        Returns:
            None
        """
        self._run.log({video_name: wandb.Video(path_to_video)})

    def log_data_dict(self,
                      data_dict: dict):
        """
        Log data in dictionary
        Args:
            data_dict: dictionary to log

        Returns:
            None
        """
        self._run.log(data_dict)


def get_logger_dict():
    return {"wandb": WandbLogger}
