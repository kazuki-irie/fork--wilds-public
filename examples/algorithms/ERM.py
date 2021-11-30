import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import sys


class ERM(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        print(model)
        num_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"# Trainable params: {num_params}")
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

    def objective(self, results):
        return self.loss.compute(
            results['y_pred'], results['y_true'], return_dict=False)
