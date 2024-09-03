import os
import wandb
import torch
from tqdm import tqdm

from utils import save_json
from utils import pretty_print_json
from trainer.multitask_trainer import MultitaskTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultitaskOrthogonalTrainer(MultitaskTrainer):

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
        log_console,
        log_steps,
        log_wandb,
        project_name,
        experiment_name, 
        cls_loss_fn,
        reg_loss_fn,
        cls_metric, 
        reg_metric, 
        weight_regression,
        weight_classify,
        weight_grad
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
            epochs=epochs,
            output_dir=output_dir,
            log_console=log_console,
            log_steps=log_steps,
            log_wandb=log_wandb,
            project_name=project_name,
            experiment_name=experiment_name,
            cls_loss_fn=cls_loss_fn,
            reg_loss_fn=reg_loss_fn,
            cls_metric=cls_metric,
            reg_metric=reg_metric,
            weight_regression=weight_regression,
            weight_classify=weight_classify
        )
        
        self.w_grad = weight_grad

    def _inner_training_loop(
            self,
            train_dataloader,
            model,
            optimizer,
            lr_scheduler,
            cls_loss_fn,
            reg_loss_fn,
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
        lr_current = optimizer.param_groups[0]["lr"]
        for x, y_cls, y_reg in train_dataloader:
            reg_output, cls_output = model(x)
            reg_loss = reg_loss_fn(reg_output, y_reg)
            y_cls = y_cls.view(-1)
            cls_loss = cls_loss_fn(cls_output, y_cls)

            grads_reg = torch.autograd.grad(
                outputs=reg_loss,
                inputs=model.share_parameters(),
                retain_graph=True,
                allow_unused=True
            )
            grads_cls = torch.autograd.grad(
                outputs=cls_loss,
                inputs=model.share_parameters(),
                retain_graph=True,
                allow_unused=True
            )
            grad_loss = 0
            for i in range(len(grads_reg)):
                grad_cls = grads_cls[i]
                grad_reg = grads_reg[i]
                if grad_cls is not None and grad_reg is not None:
                    grad_loss += torch.norm(
                        (torch.mul(grad_cls, grad_reg) - torch.ones_like(grad_reg).to(device)), 2
                    )
            loss = self.w_reg * reg_loss + self.w_cls * cls_loss + self.w_grad * grad_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
                lr_current = lr_scheduler.get_last_lr()[0]

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
            "Train Loss Reg": avg_loss_reg,
            "Train Loss Cls": avg_loss_cls,
            "Train MAE": avg_mae,
            "Train Acc": avg_acc,
            "Train F1": avg_f1,
            "Learning rate": lr_current
        }
        return log_result
