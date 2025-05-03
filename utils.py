import torch
import pandas as pd
from torch.utils.data import Dataset


def log_beta(alpha, beta):
    """
    Computes the log of the Beta function using the gamma function
    """
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

# Class for handling survival data


class SurvivalDataset(Dataset):
    """
    Custom dataset for survival analysis. This class is designed to hold the features (X),
    survival times (T), and censoring information (C) in a format compatible with PyTorch.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input feature matrix containing the covariates (independent variables) for each observation.

    T : np.ndarray or list
        The survival times for each observation. These represent the time until the event or censoring occurs.

    C : np.ndarray or list
        The censoring indicators for each observation. A value of 1 indicates that the event occurred, 
        while 0 indicates that the observation was censored.

    Attributes
    ----------
    X : torch.Tensor
        The features of the dataset as a tensor of type `float32`.

    T : torch.Tensor
        The survival times as a tensor of type `float32`.

    c : torch.Tensor
        The censoring indicators as a tensor of type `float32`.

    Methods
    -------
    __len__() :
        Returns the number of samples in the dataset.

    __getitem__(idx) :
        Returns the features, survival time, and censoring indicator for a given index.
    """

    def __init__(self, X, T, C):
        super().__init__()

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.X = torch.tensor(X).to(torch.float32)
        self.T = torch.tensor(T).to(torch.float32)
        self.c = torch.tensor(C).to(torch.float32)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The total number of samples in the dataset.
        """
        return len(self.c)

    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the features (X), survival time (T),
            and censoring indicator (C) for the given index.
        """
        return self.X[idx], self.T[idx], self.c[idx]
