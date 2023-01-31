import numpy as np


class LinearRegression:
    """
    A linear regression model that uses the closed-form solution to fit the model.
    """

    w: np.ndarray
    b: float 

    def __init__(self) -> None:
        self.w = np.zeros(1)
        self.b = np.zeros(1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''Performs fit of the model
        Args:
            X = matrix of features (d) and examples (n)
            Y = vector of targets (n)

        Returns:
            None
        '''
        # add a bias term to X matrix
        X = np.hstack((X, np.ones(X.shape[0]).reshape(-1,1)))
        self.weights = np.zeros(X.shape[1])
        if np.linalg.det(X.T@X) != 0:
            self.weights = np.linalg.inv(X.T@X)@X.T@y
        else:
            print("Matrix is singular. No closed form solution.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
            Predict the targets determined by the weights
            Args:
                X = matrix of features (d) and examples (n)
            Return:
                y = vector of predicted targets

        '''
        self.b = self.weights[-1]
        self.w = self.weights[:-1]
        return X@self.w + self.b

class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def gradient(self, X, y, w):

        return (2/X.shape[0])*X.T@(X@w - y)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000)-> None:

        '''Performs fit of the model
        Args:
            X = matrix of features (d) and examples (n)
            Y = vector of targets (n)
            lr = learning rate
            epochs = number of iterations
        Returns:
            None
        '''
        if (X[:,-1] != np.ones(X.shape[0])).all():
            X = np.hstack((X, np.ones(X.shape[0]).reshape(-1,1)))
        
        self.w = np.zeros(X.shape[1]).reshape(-1,1)
        if y.shape != X.shape[0]:
            y = y.reshape(-1,1)

        for i in range(epochs):
            
            self.w -= lr*self.gradient(X,y,self.w)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X@self.w[:-1] + self.w[-1]
