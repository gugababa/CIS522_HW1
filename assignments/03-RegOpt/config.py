from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 32
    num_epochs = 50
    initial_learning_rate = 8e-3
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "gamma": 0.1,
        "stepsize": 5,
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
            torch.transforms.ColorJitter(75, 80),
            torch.transforms.RandomRotation(10),
            torch.transforms.RandomHorizontalFlip(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensor(),
        ]
    )
