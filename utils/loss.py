import paddle
import paddle.nn.functional as F
import numpy as np


def evidential_loss(mu, v, alpha, beta, targets, coeff=0.01):
    """Evidential loss function."""
    # 计算NLL损失
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * paddle.log(np.pi / v) \
          - alpha * paddle.log(twoBlambda) \
          + (alpha + 0.5) * paddle.log(v * (targets - mu) ** 2 + twoBlambda) \
          + paddle.lgamma(alpha) \
          - paddle.lgamma(alpha + 0.5)

    # 计算正则化项
    error = paddle.abs(targets - mu)
    reg = error * (2 * v + alpha)

    loss = nll + coeff * reg
    return paddle.mean(loss)


def mse_loss(pred, targets):
    """Mean squared error loss."""
    return F.mse_loss(pred, targets)