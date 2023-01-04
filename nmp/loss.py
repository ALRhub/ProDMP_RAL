"""
@brief:     Custom loss functions in PyTorch
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal


def nll_loss(true_val,
             pred_mean,
             pred_L):
    """
    Log likelihood loss
    Args:
        true_val: true target values
        pred_mean: predicted mean of the Normal distribution
        pred_L: alternatively, use predicted Cholesky Decomposition

    Returns:
        log likelihood

    """
    # Shape of true_val:
    # [*add_dim, dim_val]
    #
    # Shape of pred_mean:
    # [*add_dim, dim_val]
    #
    # Shape of pred_L:
    # [*add_dim, dim_val, dim_val]

    # Construct distribution
    mvn = MultivariateNormal(loc=pred_mean, scale_tril=pred_L,
                             validate_args=False)

    # Compute log likelihood
    ll = mvn.log_prob(true_val).mean()

    # Loss
    ll_loss = -ll
    return ll_loss


def nmll_loss(true_val,
              pred_mean,
              pred_L,
              mc_smp_dim: int = -3):
    """
    Marginal log likelihood loss
    Args:
        true_val: true target values
        pred_mean: predicted mean of the Normal distribution
        pred_L: alternatively, use predicted Cholesky Decomposition
        mc_smp_dim: where is the mc sample dimension

    Returns:

    """
    # Shape of true_val:
    # [*add_dim_1, num_mc_smp, *add_dim_2, dim_val]
    #
    # Shape of pred_mean:
    # [*add_dim_1, num_mc_smp, *add_dim_2, dim_val]
    #
    # Shape of pred_L:
    # [*add_dim_1, num_mc_smp, *add_dim_2, dim_val, dim_val]

    # Check dimensions
    if mc_smp_dim < 0:
        mc_smp_dim = pred_mean.ndim + mc_smp_dim
    shapes = pred_mean.shape
    dimensions = list(range(pred_mean.ndim))
    add_dim_1 = torch.tensor(shapes[:mc_smp_dim])
    num_mc_smp = shapes[mc_smp_dim]
    add_dim_2 = torch.tensor(shapes[mc_smp_dim + 1:-1])

    # Construct distribution
    mvn = MultivariateNormal(loc=pred_mean, scale_tril=pred_L,
                             validate_args=False)

    # Compute log likelihood loss for each trajectory
    ll = mvn.log_prob(true_val)

    # Sum among additional dimensions part 2
    ll = torch.sum(ll, dim=dimensions[mc_smp_dim + 1:-1])

    # MC average
    ll = torch.logsumexp(ll, dim=mc_smp_dim)

    # Average among additional dimensions part 1
    ll = torch.sum(ll, dim=dimensions[:mc_smp_dim])
    assert ll.ndim == 0

    # Get marginal log likelihood
    ll = ll - torch.prod(add_dim_1) * np.log(num_mc_smp)
    mll = ll / (torch.prod(add_dim_1) * torch.prod(add_dim_2))

    # Loss
    mll_loss = -mll
    return mll_loss


def mse_loss(true_val, pred):
    """
    Mean squared error

    Args:
        true_val: Ground truth
        pred: predicted value

    Returns:
        mse
    """
    mse = nn.MSELoss()
    return mse(pred, true_val)
