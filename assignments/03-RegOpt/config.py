from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomRotation,
    RandomHorizontalFlip,
)


class CONFIG:
    batch_size = 256
    num_epochs = 15
    initial_learning_rate = 0.05
    initial_weight_decay = 0.0001

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "max_epochs": 10,
        "linear_rate": 0.004,
        "gamma": 0.2,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.SGD(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
        momentum=0.9,
    )

    transforms = Compose(
        [
            ToTensor(),
            RandomRotation(10),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
