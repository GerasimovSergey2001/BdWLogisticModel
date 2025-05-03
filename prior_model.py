import torch
import torch.nn as nn


class BetaLogisticRegression(nn.Module):
    """
    Beta Logistic Regression neural network.

    This model parameterizes a Beta distribution by predicting α and β
    for each input. Typically used in settings where the target variable
    is a probability (e.g., retention modeling).

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden : int, optional
        Size of the hidden layer. Defaults to input_size // 5.
    output_size : int, optional
        Output size, default is 2 for α and β.

    Attributes
    ----------
    linear : nn.Linear
        First linear transformation layer.
    out : nn.Linear
        Final linear layer mapping to output parameters.
    act : nn.SiLU
        Activation function.
    """

    def __init__(self, input_size, hidden=None, output_size=2):
        super().__init__()
        hidden = input_size // 5 if hidden is None else hidden
        self.linear = nn.Linear(input_size, hidden)
        self.out = nn.Linear(hidden, output_size)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return torch.exp(self.out(x))  # ensures α, β > 0


class BdWRegression(nn.Module):
    """
    Beta-discrete-Weibull (BdW) regression model.

    This model predicts the parameters α and β of a Beta distribution
    and the Weibull shape parameter γ. The gamma parameter can be handled
    in three modes:
        - 'constant': a single shared γ for all data.
        - 'partitioned': γ predicted from a subset of input features.
        - 'individual': γ predicted for each observation.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden : int, optional
        Number of hidden units. Defaults to input_size // 5.
    output_size : int, optional
        Output size, default is 2 (α and β). If gamma_type is 'individual',
        output size is incremented by 1.
    gamma_type : str, optional
        One of ['constant', 'partitioned', 'individual']. Determines how γ is modeled.
    gamma_cols_idxs : np.ndarray, optional
        Indexes of input features to be used for gamma prediction
        if gamma_type='partitioned'.

    Attributes
    ----------
    gamma_type : str
        Mode of gamma estimation.
    gamma_cols_idxs : np.ndarray or None
        Indexes used for gamma modeling (partitioned mode only).
    linear : nn.Linear
        First transformation layer.
    out : nn.Linear
        Output layer for α, β (and possibly γ).
    act : nn.SiLU
        Activation function.
    log_gamma : nn.Parameter
        Learnable parameter for constant γ mode.
    gamma_model : nn.Sequential
        Subnetwork for γ prediction in 'partitioned' mode.
    """

    def __init__(self, input_size, hidden=None, output_size=2, gamma_type='constant', gamma_cols_idxs=None):
        super().__init__()
        hidden = input_size // 5 if hidden is None else hidden
        self.gamma_type = gamma_type
        self.gamma_cols_idxs = gamma_cols_idxs

        # If individual gamma is predicted, output one more value
        output_size = output_size + 1 if gamma_type == 'individual' else output_size

        self.linear = nn.Linear(input_size, hidden)
        self.out = nn.Linear(hidden, output_size)
        self.act = nn.SiLU()

        if gamma_type == 'constant':
            self.log_gamma = nn.Parameter(torch.zeros(1))

        elif gamma_type == 'partitioned':
            if gamma_cols_idxs is None:
                raise ValueError(
                    "You should provide indexes of columns by which you want to partition.")

            self.gamma_model = nn.Sequential(
                nn.Linear(gamma_cols_idxs.shape[0],
                          gamma_cols_idxs.shape[0] // 2),
                nn.SiLU(),
                nn.Linear(gamma_cols_idxs.shape[0] // 2, 1)
            )

    def forward(self, x):
        z = self.linear(x)
        z = self.act(z)
        out = torch.exp(self.out(z))  # α, β (and possibly γ), all > 0

        if self.gamma_type == 'constant':
            return out, torch.clip(torch.exp(self.log_gamma + 1e-10), 0, 5)

        elif self.gamma_type == 'partitioned':
            gamma_input = x[:, self.gamma_cols_idxs]
            return out, torch.clip(torch.exp(self.gamma_model(gamma_input)), 0, 5)

        else:  # 'individual'
            return out[:, :2], torch.clip(out[:, -1], 0, 5)
