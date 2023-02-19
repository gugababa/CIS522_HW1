from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class CustomLRScheduler(_LRScheduler):

    """
    Implementation of a custom LR scheduler that inherits from PyTorch _LRScheduler
    """

    def __init__(self, optimizer, gamma, stepsize, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.gamma = gamma
        self.step_size = stepsize
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        """
        Create an exponential scheduler, with the learning rate decaying exponentially with square root of gamma value
        """

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]

        return [i["lr"] * (self.gamma) for i in self.optimizer.param_groups]
