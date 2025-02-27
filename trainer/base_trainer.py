import os

import torch
import wandb
from dotenv import load_dotenv

from torch.utils.data import DataLoader


load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")

class BaseTrainer:

    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        test_dataset,
        optimizer,
        lr_scheduler,
        batch_size,
        epochs,
        output_dir,
        log_steps,
        log_console=True,
        log_wandb=False,
        project_name=None,
        experiment_name=None
    ):
        self.model = model
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.dev_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_console = log_console
        self.log_steps = log_steps
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if log_wandb:
            self.log_wandb = True
            wandb.init(project=project_name, name=experiment_name)
        else:
            self.log_wandb = False

        print("Number of model parameters: {}".format(self.count_parameters()))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    
    def get_log_message(self, epoch, metric, before, after, patient=False):
        if not self.log_console: 
            return None
        if patient:
            log_message = "Epoch {}: {} is {}. Don't improve from {}".format(
                epoch, metric, after, before
            )
        else:
            log_message = "Epoch {}: Improve {} from {} to {}".format(
            epoch, metric, before, after 
        )
        return log_message
    
    def save_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def train(self):
        # Training loop
        pass
