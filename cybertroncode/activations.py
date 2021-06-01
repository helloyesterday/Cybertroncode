from mindspore import nn
from mindspore.ops import operations as P

class ShiftedSoftplus(nn.Cell):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    """
    def __init__(self):
        super().__init__()
        # self.softplus = P.Softplus()
        self.log1p = P.Log1p()
        self.exp = P.Exp()
        self.ln2 = 0.6931471805599453

    def __str__(self):
        return "shifted_softplus"

    def construct(self,x):
        # return self.softplus(x) - self.ln2
        return self.log1p(self.exp(x)) - self.ln2

class Swish(nn.Cell):
    r"""Compute swish\SILU\SiL function.

    .. math::
       y_i = x_i / (1 + e^{-beta * x_i})

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def __str__(self):
        return "swish"

    def construct(self,x):
        return x * self.sigmoid(x)
