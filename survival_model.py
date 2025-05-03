import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, brier_score_loss
)

from IPython.display import clear_output
from tqdm import tqdm


from ltv_model.utils import *
from ltv_model.prior_model import *


def BdWLoss(alpha, beta, t, c, gamma=torch.tensor([1.]), eps=1e-6, weights=None):
    """
    Compute the negative log-likelihood loss for the Beta-Discrete-Weibull (BdW) survival model.

    This loss function handles both censored and uncensored survival data. It computes the 
    likelihood based on a Beta-Discrete-Weibull distribution, where the survival function is 
    modeled using the beta function evaluated at transformed time points. The final loss is 
    the negative mean log-likelihood, optionally weighted.

    Parameters
    ----------
    alpha : torch.Tensor
        First shape parameter of the Beta distribution. Shape: (batch_size,).
    beta : torch.Tensor
        Second shape parameter of the Beta distribution. Shape: (batch_size,).
    t : torch.Tensor
        Discrete time to event or censoring. Shape: (batch_size,).
    c : torch.Tensor
        Censoring indicator (0 = uncensored, 1 = right-censored). Shape: (batch_size,).
    gamma : torch.Tensor, optional
        Shape parameter of the Weibull transformation. Default is `torch.tensor([1.])`.
    eps : float, optional
        Small constant for numerical stability. Default is 1e-6.
    weights : Callable[[torch.Tensor], torch.Tensor] or None, optional
        Optional weighting function applied to the per-sample log-likelihoods. If None, no weighting is used.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the negative mean log-likelihood across the batch.

    Notes
    -----
    - For uncensored data (`c == 0`), the log-likelihood is computed using the difference 
      between log-beta functions evaluated at `t` and `t+1`.
    - For censored data (`c == 1`), the log-likelihood uses the log-beta function at `t+1` only.
    - Subtracting `log_beta(alpha, beta)` ensures the likelihood is properly normalized.
    - The function supports optional per-time weights (e.g., inverse-frequency weights).
    """
    gamma = gamma.squeeze()

    a = log_beta(alpha + torch.pow(t, gamma), beta)
    b = log_beta(alpha + torch.pow(t + 1, gamma), beta)

    # Ensure a > b
    max_val = torch.maximum(a, b)
    min_val = torch.minimum(a, b)

    l_uncensored = max_val + torch.log1p(-torch.exp(min_val - max_val) + eps)

    l_censored = log_beta(alpha + torch.pow(t, gamma), beta)

    l = torch.where(c == 0, l_uncensored, l_censored) - log_beta(alpha, beta)

    if weights is None:
        return -torch.mean(l)
    else:
        return -torch.mean(l * weights(t))


class SurvivalModel:
    """
    A survival model class that supports 'weibull' and 'beta_logistic' models for survival analysis.

    This class is designed to handle survival data, where the goal is to predict the time-to-event 
    along with accounting for censoring. It supports two different model types: 'weibull' and 'beta_logistic',
    and it can handle various ways of treating the gamma parameter.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input features (e.g., `(batch_size, num_features)`).
    model_type : str
        Type of the survival model. Options are:
        - 'weibull': A model based on the Weibull distribution.
        - 'beta_logistic': A model based on a Beta distribution and logistic regression.
    gamma_type : str
        Specifies how to treat the gamma parameter. Options are:
        - 'constant': The gamma parameter is a constant across all samples.
        - 'partitioned': The gamma parameter is partitioned across different columns.
        - 'individual': The gamma parameter is individual for each sample.
    device : str
        The device to run the model on. Either 'cuda' (GPU) or 'cpu'.
    gamma_cols_idxs : list of int, optional
        Indices of columns to partition the gamma parameter across when `gamma_type` is 'partitioned'.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network architecture for the survival model, either Weibull or Beta Logistic regression.
    loss_function : callable
        The loss function used for training the model.
    name : str
        A string name representing the model, based on the selected `model_type`.

    Notes
    -----
    - The model can be extended to incorporate other types of survival distributions by modifying 
      the `model_type` parameter.
    - The `gamma_type` determines how the model treats the shape parameter for the survival distribution:
        - For 'constant' gamma, the parameter is shared across all samples.
        - For 'partitioned', a specific set of columns is used to define different gamma parameters.
        - For 'individual', each sample has its own unique gamma parameter.
    """

    def __init__(self, input_shape, model_type='weibull', gamma_type='constant', hidden=None,  device='cuda', gamma_cols_idxs=None):
        self.model_type = model_type
        self.gamma_type = gamma_type
        self.device = device
        self.gamma_cols_idxs = gamma_cols_idxs

        # Choose model type: Weibull or Beta Logistic regression
        self.model = BdWRegression(input_shape, hidden=hidden, gamma_type=gamma_type, gamma_cols_idxs=gamma_cols_idxs).to(device) \
            if model_type == 'weibull' else BetaLogisticRegression(input_shape).to(device)

        self.loss_function = BdWLoss  # The loss function is specific to the BdW model

        # Set the name of the model based on the type
        if model_type == 'weibull':
            self.name = 'bdw'
        else:
            self.name = 'beta_logistic'

    def fit(self, X_train, T_train, c_train, X_test, T_test, c_test, lr=1e-4, weight_decay=1e-4, n_epochs=15, batch_size=64, show_training_plots=True, path=None, save_device='cuda'):
        """
        Trains the survival model using the provided training and testing data.

        This method optimizes the model parameters using stochastic gradient descent with a specified learning rate,
        L2 regularization, and a set number of epochs. It also supports saving the trained model and optionally
        plotting the training and test loss during the training process.

        Parameters
        ----------
        X_train : torch.Tensor
            The input features for the training data (shape: [n_samples, n_features]).
        T_train : torch.Tensor
            The survival times for the training data (shape: [n_samples]).
        c_train : torch.Tensor
            The censoring indicators for the training data (0 for uncensored, 1 for censored) (shape: [n_samples]).
        X_test : torch.Tensor
            The input features for the testing data (shape: [n_samples, n_features]).
        T_test : torch.Tensor
            The survival times for the testing data (shape: [n_samples]).
        c_test : torch.Tensor
            The censoring indicators for the testing data (0 for uncensored, 1 for censored) (shape: [n_samples]).
        lr : float
            The learning rate for the optimization algorithm.
        weight_decay : float
            The L2 regularization coefficient to prevent overfitting.
        n_epochs : int
            The number of epochs to train the model.
        batch_size : int
            The batch size to use during training.
        show_training_plots : bool
            If True, displays training and test loss plots after training.
        path : str, optional
            The path where the trained model will be saved. If None, the model is not saved.
        save_device : str, optional
            The device ('cpu' or 'cuda') to save the trained model.

        Returns
        -------
        self : object
            The trained model instance.

        Notes
        -----
        - This method assumes that the model has been instantiated with the necessary architecture and loss function.
        - If `show_training_plots` is True, the function will plot the training and testing losses after training is complete.
        - The trained model can be saved to the specified `path`. The model will be saved on the device specified by `save_device`.
        """
        self.train_data = SurvivalDataset(X_train, T_train, c_train)
        self.test_data = SurvivalDataset(X_test, T_test, c_test)

        train_loader = DataLoader(
            self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            self.test_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loss_list, test_loss_list = [], []

        for epoch in range(1, n_epochs+1):
            train_loss = self.train(
                train_loader, optimizer, self.loss_function)
            test_loss = self.eval(test_loader, self.loss_function)

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

            if show_training_plots:
                clear_output(wait=True)
                fig, ax = plt.subplots(2, figsize=(10, 5))
                plt.tight_layout()

                ax[0].plot(np.arange(1, len(train_loss_list)+1),
                           train_loss_list)
                ax[1].plot(np.arange(1, len(test_loss_list)+1), test_loss_list)
                plt.title(f'Epoch №{epoch}')
                ax[0].set_xticks(np.arange(1, n_epochs+1))
                ax[1].set_xticks(np.arange(1, n_epochs+1))

                ax[0].set_title('Train')
                ax[1].set_title('Test')

                plt.xlabel('Epoch')
                plt.ylabel('Loss')

                plt.show()

        path = self.name + '_' + self.gamma_type + '.pth' if path is None else path
        torch.save(self.model.to(save_device).state_dict(), path)
        return self

    def load(self, path=None, load_device='cuda'):
        """
        Loads the model weights from a saved file.

        This method loads the model's state dictionary from a specified path and transfers it to the given device.

        Parameters
        ----------
        path : str
            The path to the saved model state file.
        load_device : str, optional
            The device ('cpu' or 'cuda') to load the model onto. Default is 'cpu'.

        Returns
        -------
        self : object
            The model with the loaded weights.

        Notes
        -----
        - The model state is loaded onto the specified device. If the model was saved on a different device,
        it will automatically be transferred to the given `load_device`.
        - Ensure that the model architecture is defined before calling this method, as it requires the same architecture 
        as the one used when saving the model.
        - The model's weights will be loaded, but other attributes like training state (e.g., optimizer states) 
        may need to be handled separately.
        """
        path = self.name + '_' + self.gamma_type + '.pth' if path is None else path
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(load_device)
        return self

    def train(self, train_loader, optimizer, loss_function):
        """
        Trains the model for one epoch.

        This method iterates over the training data in batches, performs forward and backward passes,
        and updates the model parameters using the specified optimizer.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training data, providing batches of inputs and target labels.
        optimizer : torch.optim.Optimizer
            The optimizer used to update model parameters based on the computed gradients.
        loss_function : function
            The loss function to compute the training loss, which guides the optimization process.

        Returns
        -------
        float
            The average training loss for the epoch, computed as the mean loss over all batches.

        Notes
        -----
        - The optimizer steps after computing the gradients for each batch, and the model weights are updated accordingly.
        - The loss function should be appropriate for the type of model (e.g., survival loss function for survival analysis models).
        - This method does not include evaluation of the model on validation/test data; it is focused on training.
        """
        self.model.train()
        loss_list = []
        for x, t, c in tqdm(train_loader):
            x, t, c = x.to(self.device), t.to(self.device), c.to(self.device)
            pred = self.model(x)
            if self.model_type == 'weibull':
                pred, gamma = pred[0], pred[1]
                alpha, beta = pred[:, 0], pred[:, 1]
                loss = loss_function(alpha, beta, t, c, gamma)
            else:
                alpha, beta = pred[:, 0], pred[:, 1]
                loss = loss_function(alpha, beta, t, c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().numpy())

        # Adjust learning rate
        optimizer.param_groups[0]['lr'] *= (1 - 1e-3)
        return np.mean(loss_list)

    def eval(self, test_loader, loss_function):
        """
        Evaluates the model on the test set.

        This method computes the average test loss by iterating over the test data in batches
        and applying the loss function to the model's predictions.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader for the test data, providing batches of inputs and target labels.
        loss_function : function
            The loss function to compute the test loss, used to evaluate the model's performance on the test data.

        Returns
        -------
        float
            The average test loss, computed as the mean loss over all batches in the test set.

        Notes
        -----
        - The model is not updated during evaluation; this is a pure inference phase.
        - The loss function should be appropriate for the task (e.g., survival analysis loss function).
        - This method is intended for evaluating the model's performance on unseen data after training.
        """
        self.model.eval()
        x, t, c = test_loader.dataset[:]
        x, t, c = x.to(self.device), t.to(self.device), c.to(self.device)
        pred = self.model(x)
        if self.model_type == 'weibull':
            pred, gamma = pred[0], pred[1]
            alpha, beta = pred[:, 0], pred[:, 1]
            loss = loss_function(alpha, beta, t, c, gamma)
        else:
            alpha, beta = pred[:, 0], pred[:, 1]
            loss = loss_function(alpha, beta, t, c)
        return loss.detach().cpu().numpy().mean()

    def survival_function(self, X=None, T=None, c=None, method='expectation'):
        """
        Computes the survival function for the given data.

        This method estimates the survival function, which provides the probability of survival at any given time 
        for each observation in the dataset. The computation can be done using different methods, such as 
        'expectation' for expected survival.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The feature matrix containing the input data used for prediction. Each row corresponds to an observation, 
            and each column represents a feature.
        T : np.ndarray
            The survival times for each observation. This is a 1D array where each value represents the observed 
            survival time for a given individual.
        c : np.ndarray
            The censoring indicators. A 1D array where each element is either 0 (uncensored) or 1 (censored).
        method : str, optional, default='expectation'
            The method used to compute the survival function. Common options are:
            - 'expectation' for expected survival based on model predictions.
            - Other methods may be implemented for alternative approaches (e.g., Kaplan-Meier).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed survival function values for each observation. The rows represent 
            observations, and the columns represent survival probabilities at different time points.

        Notes
        -----
        - The 'expectation' method typically calculates the survival function based on the expected time-to-event 
        predicted by the model.
        - If an alternative method is implemented, it should be specified by the `method` argument.
        - The survival function is useful for evaluating the probability that an event of interest has not occurred by 
        a certain time.
        """
        index = None
        if (X is not None) and (T is not None) and (c is not None):
            if isinstance(X, pd.DataFrame):
                index = X.index
            data = SurvivalDataset(X, T, c)
            X, T, c = data[:]
        else:
            X, T, c = self.train_data[:]
        X, T, c = X.to(self.device), T.to(self.device), c.to(self.device)

        survival_dict = {}
        if self.model_type == 'weibull':
            with torch.no_grad():
                pred, gamma = self.model(X)
                alpha, beta = pred[:, 0].cpu(), pred[:, 1].cpu()
                gamma = gamma.cpu().squeeze()
        else:
            with torch.no_grad():
                pred = self.model(X)
                alpha, beta = pred[:, 0].cpu(), pred[:, 1].cpu()
                gamma = torch.tensor([1.])

        for t in range(0, 6):
            t = torch.tensor([float(t)])

            if method == 'expectation':
                S_hat = torch.exp(log_beta(
                    alpha + torch.pow(t + 1, gamma), beta) - log_beta(alpha, beta)).numpy()
            else:
                S_hat = torch.pow(alpha / (alpha + beta),
                                  torch.pow(t + 1, gamma))
            survival_dict[int(t)] = S_hat

        self.survival_hat = pd.DataFrame(survival_dict, index=index).T
        self.gamma = gamma
        return self.survival_hat

    def prediction_results(self, step=1):
        """
        Computes and prints prediction results including accuracy, F1 score, ROC AUC, and others.

        This method evaluates the model's performance based on the predicted outcomes and compares them 
        with the ground truth values. The evaluation includes metrics such as accuracy, F1 score, 
        ROC AUC, and possibly others depending on the implementation.

        Parameters
        ----------
        step : int
            The step for evaluation. Currently, only `step=1` is supported.

        Returns
        -------
        None
            The method does not return any value but prints the evaluation metrics directly to the console.

        Notes
        -----
        - The method assumes that predictions and ground truth values are available at the specified step.
        - It is assumed that the model's output includes probabilities or logits suitable for calculating metrics 
        like ROC AUC.
        - The supported metrics may include:
            - Accuracy: The fraction of correctly predicted instances.
            - F1 score: The harmonic mean of precision and recall.
            - ROC AUC: Area under the Receiver Operating Characteristic curve, which evaluates the model's ability 
            to discriminate between positive and negative classes.
        - Additional metrics may be added depending on future developments or model specifics.
        """
        if step != 1:
            raise NotImplementedError

        gamma_train, gamma_test = None, None
        with torch.no_grad():
            x_train, t_train, c_train = self.train_data[:]
            x_train = x_train.to(self.device)
            x_test, t_test, c_test = self.test_data[:]
            x_test = x_test.to(self.device)

            if self.model_type == 'weibull':
                pred, gamma_train = self.model(x_train)
                alpha_train, beta_train = pred[:, 0].cpu(), pred[:, 1].cpu()
                gamma_train = gamma_train.cpu()

                pred, gamma_test = self.model(x_test)
                alpha_test, beta_test = pred[:, 0].cpu(), pred[:, 1].cpu()
                gamma_test = gamma_test.cpu()

            else:
                pred = self.model(x_train)
                alpha_train, beta_train = pred[:, 0].cpu(), pred[:, 1].cpu()

                pred = self.model(x_test)
                alpha_test, beta_test = pred[:, 0].cpu(), pred[:, 1].cpu()

        self.alpha_train, self.alpha_test = alpha_train, alpha_test
        self.beta_train, self.beta_test = beta_train, beta_test
        self.gamma_train, self.gamma_test = gamma_train, gamma_test

        p = alpha_train / (alpha_train + beta_train)
        y_pred = (p > 0.5) * 1
        rebill1 = (t_train > 0) * 1
        y_pred_proba = p

        # Print classification metrics for train set
        print('Train')
        print("✅ Accuracy:", accuracy_score(rebill1, y_pred))
        print("✅ F1 Score:", f1_score(rebill1, y_pred))
        print("✅ ROC AUC:", roc_auc_score(rebill1, y_pred_proba))
        print("✅ Brier Score:", brier_score_loss(rebill1, y_pred_proba))
        print("\n📊 Classification Report:\n",
              classification_report(rebill1, y_pred))
        print("🧩 Confusion Matrix:\n", confusion_matrix(rebill1, y_pred))

        # Calibration curve for train set
        prob_true, prob_pred = calibration_curve(
            rebill1, y_pred_proba, n_bins=10)
        plt.figure(figsize=(7, 5))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--',
                 color='gray', label='Perfectly Calibrated')
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Frequency")
        plt.title("Calibration Curve (Train)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print classification metrics for test set
        p = alpha_test / (alpha_test + beta_test)
        rebill1 = (t_test > 0) * 1
        y_pred = (p > 0.5) * 1
        y_pred_proba = p

        print('Test')
        print("✅ Accuracy:", accuracy_score(rebill1, y_pred))
        print("✅ F1 Score:", f1_score(rebill1, y_pred))
        print("✅ ROC AUC:", roc_auc_score(rebill1, y_pred_proba))
        print("✅ Brier Score:", brier_score_loss(rebill1, y_pred_proba))
        print("\n📊 Classification Report:\n",
              classification_report(rebill1, y_pred))
        print("🧩 Confusion Matrix:\n", confusion_matrix(rebill1, y_pred))

        # Calibration curve for test set
        prob_true, prob_pred = calibration_curve(
            rebill1, y_pred_proba, n_bins=10)
        plt.figure(figsize=(7, 5))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--',
                 color='gray', label='Perfectly Calibrated')
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Frequency")
        plt.title("Calibration Curve (Test)")
        plt.legend()
        plt.grid(True)
        plt.show()
