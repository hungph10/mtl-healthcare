import os
from dotenv import load_dotenv

import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import save_json
from utils import pretty_print_json

load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self) -> None:
        # Initialize the parameters here
        pass

    def train():
        # Training loop
        pass

class MultitaskTrainer(Trainer):
    
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        cls_loss_fn,
        reg_loss_fn,
        cls_metric,
        reg_metric,
        optimizer,
        batch_size,
        epochs,
        output_dir,
        log_steps=5,
        log_wandb=False,
        project_name=None,
        experiment_name=None
    ):
        super().__init__()
        
        self.model = model
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        self.cls_loss_fn = cls_loss_fn
        self.reg_loss_fn = reg_loss_fn
        self.cls_metric = cls_metric
        self.reg_metric = reg_metric
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_steps = log_steps
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if log_wandb:
            self.log_wandb = True
            wandb.init(project=project_name, name=experiment_name)
        else:
            self.log_wandb = False

    def train(self):
        # Evaluate before training
        test_log = self.evaluate(
            test_dataloader=self.test_dataloader,
            model=self.model,
            compute_cls_loss=self.cls_loss_fn,
            compute_reg_loss=self.reg_loss_fn,
            cls_metric=self.cls_metric,
            reg_metric=self.reg_metric
        )
        max_f1 = test_log["Test F1"]
        max_mae = test_log["Test MAE"]
        max_loss = test_log["Test Loss"]
        print("Evaluate before training:")
        pretty_print_json(test_log)

        best_cls_log = {}
        best_reg_log = {}
        best_multitask_log = {}
        
        # Training
        self.model.to(device)
        patient = 0
        pbar = tqdm(range(self.epochs), desc="Training")
        for _ in pbar:
            train_log = self._inner_training_loop(
                train_dataloader=self.train_dataloader,
                model=self.model, 
                cls_loss_fn=self.cls_loss_fn,
                reg_loss_fn=self.reg_loss_fn,
                optimizer=self.optimizer,
                cls_metric=self.cls_metric,
                reg_metric=self.reg_metric
            )
            test_log = self.evaluate(
                    test_dataloader=self.test_dataloader,
                    model=self.model,
                    compute_cls_loss=self.cls_loss_fn,
                    compute_reg_loss=self.reg_loss_fn,
                    cls_metric=self.cls_metric,
                    reg_metric=self.reg_metric,
                    pbar=pbar,
                    train_log=train_log
            )

            if self.log_wandb:
                wandb.log(test_log)

            # Save best checkpoint classify
            if test_log["Test F1"] > max_f1:
                print(f"Improve F1 score from {round(max_f1, 2)} to {round(test_log['Test F1'], 2)}")
                max_f1 = test_log["Test F1"]
                best_cls_log = test_log
                checkpoint_path = os.path.join(self.output_dir, "best_cls.pth")
                self.save_checkpoint(checkpoint_path=checkpoint_path)
                
            # Save best checkpoint regression
            if test_log["Test MAE"] < max_mae:
                print(f"Improve MAE score from {round(max_mae, 2)} to {round(test_log['Test MAE'], 2)}")
                max_mae = test_log["Test MAE"]
                best_reg_log = test_log
                checkpoint_path = os.path.join(self.output_dir, "best_reg.pth")
                self.save_checkpoint(checkpoint_path=checkpoint_path)
                
            # Save best checkpoint 
            if test_log["Test Loss"] > max_loss:
                patient += 1
                print(f"Test loss: {test_log['Test Loss']} => Don't improve from {max_loss}")
            else:
                patient = 0
                max_loss = test_log["Test Loss"]
                checkpoint_path = os.path.join(self.output_dir, "best_multitask.pth")
                self.save_checkpoint(checkpoint_path=checkpoint_path)
        #     if patient > 100:
        #         print(f"Early stopping at epoch {epoch + 1}!")
        #         break

        result_training = {
            "best_cls_log": best_cls_log,
            "best_reg_log": best_reg_log,
            "best_multitask_log": best_multitask_log
        }
        log_path = os.path.join(self.output_dir, "result_training.json") 
        save_json(
            data=result_training,
            file_path=log_path
        )


    @staticmethod
    def _inner_training_loop(
            train_dataloader,
            model,
            cls_loss_fn,
            reg_loss_fn,
            optimizer,
            cls_metric,
            reg_metric
        ):
        num_batches = len(train_dataloader)
        total_loss = 0
        
        total_loss_reg = 0
        total_loss_cls = 0
        
        total_mae = 0
        total_acc = 0
        total_f1 = 0
        model.train()
        step = 0
        for x, y_cls, y_reg in train_dataloader:
            reg_output, cls_output = model(x)
            reg_loss = reg_loss_fn(reg_output, y_reg)
            y_cls = y_cls.view(-1)
            cls_loss = cls_loss_fn(cls_output, y_cls)
            
            loss = reg_loss + cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            total_loss_reg += reg_loss.item()
            total_loss_cls += cls_loss.item()
            total_loss += loss.item()
            
            total_mae += reg_metric(reg_output, y_reg).item()
            acc, f1 = cls_metric(cls_output, y_cls)
            total_acc += acc
            total_f1 += f1
            step += 1
        avg_loss = total_loss / num_batches
        avg_loss_cls = total_loss_cls / num_batches
        avg_loss_reg = total_loss_reg / num_batches
        
        avg_mae = total_mae / num_batches
        avg_acc = total_acc / num_batches
        avg_f1 = total_acc / num_batches
        
        log_result = {
            "Train Loss": avg_loss,
            "Train Loss Regression": avg_loss_reg,
            "Train Loss Classify": avg_loss_cls,
            "Train MAE": avg_mae,
            "Train Accuracy": avg_acc,
            "Train F1": avg_f1
        }
        
        return log_result

    @staticmethod
    def evaluate(
        test_dataloader,
        model,
        compute_cls_loss, 
        compute_reg_loss,
        cls_metric,
        reg_metric,
        train_log={},
        pbar=None
    ):
        num_batches = len(test_dataloader)
        total_loss = 0
        total_loss_reg = 0
        total_loss_cls = 0
        
        total_mae = 0
        total_acc = 0
        total_f1 = 0
        
        model.eval()
        with torch.no_grad():
            for x, y_cls, y_reg in test_dataloader:
                y_cls = y_cls.view(-1)
                reg_output, cls_output = model(x)
                reg_loss = compute_reg_loss(reg_output, y_reg)
                cls_loss = compute_cls_loss(cls_output, y_cls)
                
                loss = reg_loss + cls_loss

                total_loss_cls += cls_loss.item()
                total_loss_reg += reg_loss.item()

                total_loss += loss.item()
                total_mae += reg_metric(reg_output, y_reg).item()
                acc, f1 = cls_metric(cls_output, y_cls)
                total_acc += acc
                total_f1 += f1
                
                

        avg_loss = total_loss / num_batches
        
        avg_loss_cls = total_loss_cls / num_batches
        avg_loss_reg = total_loss_reg / num_batches
        avg_mae = total_mae / num_batches
        avg_acc = total_acc / num_batches
        avg_f1 = total_acc / num_batches
        
        log_result = {
            "Test Loss": avg_loss,
            "Test Loss Regression": avg_loss_reg,
            "Test Loss Classify": avg_loss_cls,
            "Test MAE": avg_mae,
            "Test Accuracy": avg_acc,
            "Test F1": avg_f1
        }
        
    #     log_result = {
    #         "Test Loss": avg_loss,
    #         "Test Accuracy": avg_acc,
    #         "Test F1": avg_f1
    #     }
        log_result.update(train_log)
        if pbar:
            pbar.set_postfix(**log_result)
        return log_result

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

class Evaluate:

    def __init__(self) -> None:
        pass

    def eval():
        pass