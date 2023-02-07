import torch
from typing import Callable


class MLP(torch.nn.Module):

    """Multi-layer perceptron"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """

        super(MLP, self).__init__()
        # create activation function
        self.actv = activation()

        self.layers = torch.nn.ModuleList()
        # initialize the layers of the MLP
        for i in range(hidden_count):

            num_of_hidden_outputs = hidden_size
            self.layers += [torch.nn.Linear(input_size, num_of_hidden_outputs)]
            input_size = num_of_hidden_outputs
            initializer(self.layers[i].weight)

        # define the output layer
        self.out = torch.nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor()) -> torch.Tensor():
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        # flatten the data to 2D
        x = x.view(x.shape[0], -1)

        for i in self.layers:

            x = self.actv(i(x))

        x = self.out(x)
        return x
