from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math


class CustomLRScheduler(_LRScheduler):

    """
    Implementation of a custom LR scheduler that inherits from PyTorch _LRScheduler
    """

    def __init__(self, optimizer, max_epochs, min_lr, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.T_max = max_epochs
        self.eta_min = min_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        """
        Cosine annealing scheduler derived from PyTorch documentation
        """

        if self.last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                base_lr
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr in self.base_lrs
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (base_lr - self.eta_min)
            + self.eta_min
            for base_lr in self.base_lrs
        ]
