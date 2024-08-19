import random

import numpy as np
import torch

from dataset import (
    get_data_mtl,
    RegressionDataset,
    ClassifyDataset,
    MultitaskDataset
)
from trainer import (
    ClassifyTrainer,
    RegressionTrainer,
    MultitaskTrainer,
    MultitaskOrthogonalTrainer,
    MultitaskOrthogonalTracenormTrainer
)
from net import (
    RegressionLSTM,
    ClassifyLSTM,
    MultitaskLSTM,
    reg_loss_fn,
    reg_metric,
    cls_metric,
    cls_loss_fn
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(args, model):
    return torch.optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate
    )


def get_lr_scheduler(args, optimizer):
    if args.lr_scheduler is None:
        scheduler = None
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max,
            eta_min=1e-7,
            last_epoch=-1
        )
    elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=args.T_mul,
            eta_min=1e-7,
            last_epoch=-1
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {args.lr_scheduler}")
    return scheduler


def get_dataset(npz_path, task):
    data = np.load(npz_path)
    tensor_data = get_data_mtl(data=data)
    if task == "Classify":
        train_dataset = ClassifyDataset(
            features=tensor_data["x_train"],
            cls_target=tensor_data["y_train_cls"]
        )
        dev_dataset = ClassifyDataset(
            features=tensor_data["x_dev"],
            cls_target=tensor_data["y_dev_cls"],
        )
        test_dataset = ClassifyDataset(
            features=tensor_data["x_test"],
            cls_target=tensor_data["y_test_cls"],
        )
    elif task == "Regression":
        train_dataset = RegressionDataset(
            features=tensor_data["x_train"],
            reg_target=tensor_data["y_train_reg"]
        )
        dev_dataset = RegressionDataset(
            features=tensor_data["x_dev"],
            reg_target=tensor_data["y_dev_reg"]
        )
        test_dataset = RegressionDataset(
            features=tensor_data["x_test"],
            reg_target=tensor_data["y_test_reg"]
        )
    elif task in [
        "Multitask", "MultitaskOrthogonal", "MultitaskOrthogonalTracenorm"
    ]:
        train_dataset = MultitaskDataset(
            features=tensor_data["x_train"],
            cls_target=tensor_data["y_train_cls"],
            reg_target=tensor_data["y_train_reg"]
        )
        dev_dataset = MultitaskDataset(
            features=tensor_data["x_dev"],
            cls_target=tensor_data["y_dev_cls"],
            reg_target=tensor_data["y_dev_reg"]
        )
        test_dataset = MultitaskDataset(
            features=tensor_data["x_test"],
            cls_target=tensor_data["y_test_cls"],
            reg_target=tensor_data["y_test_reg"]
        )
    print("- Train data: {} samples".format(len(train_dataset)))
    print("- Dev data: {} samples".format(len(dev_dataset)))
    print("- Test data: {} samples".format(len(test_dataset)))
    return train_dataset, dev_dataset, test_dataset


def get_classify_trainer(args):
    model = ClassifyLSTM(
        input_size=args.input_dim,
        hidden_size_1=args.n_hidden_1,
        hidden_size_2=args.n_hidden_2,
        output_size=args.n_classes,
        dropout=args.p_dropout
    )
    model = model.to(device)
    optimizer = get_optimizer(args=args, model=model)
    lr_scheduler = get_lr_scheduler(args=args, optimizer=optimizer)
    train_dataset, dev_dataset, test_dataset = get_dataset(
        npz_path=args.data_path,
        task=args.task
    )
    trainer = ClassifyTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        test_dataset=test_dataset,
        cls_loss_fn=cls_loss_fn,
        cls_metric=cls_metric,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        log_console=args.log_console,
        log_steps=args.log_steps,
        log_wandb=args.log_wandb,
        project_name=args.project_name,
        experiment_name=args.experiment_name
    )
    return trainer


def get_regression_trainer(args):
    model = RegressionLSTM(
        input_size=args.input_dim,
        hidden_size_1=args.n_hidden_1,
        hidden_size_2=args.n_hidden_2,
        dropout=args.p_dropout
    )
    model = model.to(device)
    optimizer = get_optimizer(args=args, model=model)
    lr_scheduler = get_lr_scheduler(args=args, optimizer=optimizer)
    train_dataset, dev_dataset, test_dataset = get_dataset(
        npz_path=args.data_path,
        task=args.task
    )
    trainer = RegressionTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        test_dataset=test_dataset,
        reg_loss_fn=reg_loss_fn,
        reg_metric=reg_metric,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        log_console=args.log_console,
        log_steps=args.log_steps,
        log_wandb=args.log_wandb,
        project_name=args.project_name,
        experiment_name=args.experiment_name
    )
    return trainer


def get_multitask_trainer(args):
    model = MultitaskLSTM(
        input_size=args.input_dim,
        hidden_size_1=args.n_hidden_1,
        hidden_size_2=args.n_hidden_2,
        output_size=args.n_classes,
        dropout=args.p_dropout
    )
    model = model.to(device)
    optimizer = get_optimizer(args=args, model=model)
    lr_scheduler = get_lr_scheduler(args=args, optimizer=optimizer)
    train_dataset, dev_dataset, test_dataset = get_dataset(
        npz_path=args.data_path,
        task=args.task
    )
    if args.task == "Multitask":
        trainer = MultitaskTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            test_dataset=test_dataset,
            cls_loss_fn=cls_loss_fn,
            reg_loss_fn=reg_loss_fn,
            cls_metric=cls_metric,
            reg_metric=reg_metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
            log_console=args.log_console,
            log_steps=args.log_steps,
            log_wandb=args.log_wandb,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            weight_regression=args.w_regression,
            weight_classify=args.w_classify,
        )
    elif args.task == "MultitaskOrthogonal":
        trainer = MultitaskOrthogonalTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            test_dataset=test_dataset,
            cls_loss_fn=cls_loss_fn,
            reg_loss_fn=reg_loss_fn,
            cls_metric=cls_metric,
            reg_metric=reg_metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
            log_steps=args.log_steps,
            log_wandb=args.log_wandb,
            log_console=args.log_console,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            weight_regression=args.w_regression,
            weight_classify=args.w_classify,
            weight_grad=args.w_grad
        )
    elif args.task == "MultitaskOrthogonalTracenorm":
        trainer = MultitaskOrthogonalTracenormTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            test_dataset=test_dataset,
            cls_loss_fn=cls_loss_fn,
            reg_loss_fn=reg_loss_fn,
            cls_metric=cls_metric,
            reg_metric=reg_metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
            log_console=args.log_console,
            log_steps=args.log_steps,
            log_wandb=args.log_wandb,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            weight_regression=args.w_regression,
            weight_classify=args.w_classify,
            weight_grad=args.w_grad,
            weight_trace_norm=args.w_trace_norm
        )
    return trainer


def get_trainer(args):
    print("Training info:")
    print("- Training task: {}".format(args.task))
    print("- Batch size: {}".format(args.batch_size))
    print("- Number of epochs: {}".format(args.epochs))
    print("- Learning rate: {}".format(args.learning_rate))
    print("- Learning rate scheduler: {}".format(args.lr_scheduler))
    if args.task == "Classify":
        return get_classify_trainer(args)
    elif args.task == "Regression":
        return get_classify_trainer(args)
    if args.task in [
        "Multitask", "MultitaskOrthogonal", "MultitaskOrthogonalTracenorm"
    ]:
        return get_multitask_trainer(args)
    else:
        raise ValueError(f"Unsupported scheduler type: {args.lr_scheduler}, task should be one of \
these tasks: Classify, Regression, Multitask, MultitaskOrthogonal, MultitaskOrthogonalTracenorm")