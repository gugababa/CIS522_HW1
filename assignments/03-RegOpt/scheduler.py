from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math


class CustomLRScheduler(_LRScheduler):

    """
    Implementation of a custom LR scheduler that inherits from PyTorch _LRScheduler
    """

    def __init__(self, optimizer, max_epochs, linear_rate, gamma, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.max_linear_epochs = max_epochs
        self.linear_rate = linear_rate
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        """
        Linear increase in learning rate, followed by an exponential decay of learning rate
        """

        if self.last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        else:
            for i in range(self.max_linear_epochs):
                return [base_lr + self.linear_rate for base_lr in self.base_lrs]
            return [base_lr * self.gamma for base_lr in self.base_lrs]
