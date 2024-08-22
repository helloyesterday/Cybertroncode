# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Basic functions
"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import nn
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import Initializer, initializer, Constant

import os
path = os.getenv('MINDSPONGE_HOME')
if path:
    import sys
    sys.path.insert(0, path)
from sponge.function import get_integer, get_arguments
from sponge.function import concat_penulti, keepdims_mean, squeeze_last_dim

from .layer import Dense, Residual
from .cutoff import SmoothCutoff

__all__ = [
    "GraphNorm",
    "Aggregate",
    "SmoothReciprocal",
    "SoftmaxWithMask",
    "PositionalEmbedding",
    "MultiheadAttention",
    "FeedForward",
    "Pondering",
    "ACTWeight",
]


class GraphNorm(Cell):
    r"""Graph normalization

    Args:
        dim_feature (int):          Feature dimension

        axis (int):                 Axis to normalize. Default: -2

        alpha_init (Initializer):   Initializer for alpha. Default: 'one'

        beta_init (Initializer):    Initializer for beta. Default: 'zero'

        gamma_init (Initializer):   Initializer for alpha. Default: 'one'

    """

    def __init__(self,
                 dim_feature: int,
                 axis: int = -2,
                 alpha_init: Initializer = 'one',
                 beta_init: Initializer = 'zero',
                 gamma_init: Initializer = 'one',
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        dim_feature = get_integer(dim_feature)

        self.alpha = Parameter(initializer(alpha_init, dim_feature), name="alpha")
        self.beta = Parameter(initializer(beta_init, dim_feature), name="beta")
        self.gamma = Parameter(initializer(gamma_init, dim_feature), name="gamma")

        self.axis = get_integer(axis)

    def construct(self, nodes: Tensor):
        """Compute graph normalization.

        Args:
            nodes (Tensor):     Tensor with shape (B, A, N, F). Data type is float.

        Returns:
            output (Tensor):    Tensor with shape (B, A, N, F). Data type is float.

        """

        mu = keepdims_mean(nodes, self.axis)

        nodes2 = nodes * nodes
        mu2 = keepdims_mean(nodes2, self.axis)

        a = self.alpha
        sigma2 = mu2 + (a*a - 2*a) * mu * mu
        sigma = F.sqrt(sigma2)

        y = self.gamma * (nodes - a * mu) / sigma + self.beta

        return y


class Aggregate(Cell):
    r"""A network to aggregate Tensor

    Args:

        axis (int):     Axis to aggregate.

        mean (bool):    Whether to average the Tensor. Default: False

    """
    def __init__(self,
                 axis: int,
                 mean: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.average = mean
        self.axis = get_integer(axis)

    def construct(self, inputs: Tensor, mask: Tensor = None):
        """To aggregate the representation of each nodes

        Args:
            inputs (Tensor):    Tensor with shape (B, A, N, F). Data type is float.
            mask (Tensor):      Tensor with shape (B, A, N). Data type is bool.

        Returns:
            output (Tensor):    Tensor with shape (B, A, F). Data type is float.

        """
        # mask input
        if mask is not None:
            inputs = inputs * F.expand_dims(mask, -1)
        # compute sum of input along axis

        y = F.reduce_sum(inputs, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                num = F.reduce_sum(mask, self.axis)
                num = F.maximum(num, other=F.ones_like(num))
            else:
                num = inputs.shape[self.axis]

            y = y / num
        return y


class SmoothReciprocal(Cell):
    r"""A smooth reciprocal function

    Args:

        dmax (float):           Maximum distance

        cutoff_network (Cell):  Cutoff network. Default: None

    """
    def __init__(self,
                 dmax: float,
                 cutoff_network: Cell = None,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        if cutoff_network is None:
            self.cutoff_network = SmoothCutoff(dmax)
        else:
            self.cutoff_network = cutoff_network(dmax)

    def construct(self, rij: Tensor, mask: Tensor):
        """calculate smooth reciprocal of Tensor

        Args:
            rij (Tensor):   Tensor with shape (..., X, ...). Data type is float.
            mask (Tensor):  Tensor with shape (..., X, ...). Data type is bool.

        Returns:
            output (Tensor):    Tensor with shape (..., X, ...). Data type is float.

        """
        phi2rij, _ = self.cutoff_network(rij*2, mask)

        r_near = phi2rij * msnp.reciprocal(F.sqrt(rij * rij + 1.0))
        r_far = msnp.where(rij > 0, (1.0 - phi2rij) * msnp.reciprocal(rij), 0)

        reciprocal = r_near + r_far
        if mask is not None:
            reciprocal = reciprocal * mask

        return reciprocal


class SoftmaxWithMask(Cell):
    r"""Softmax function with mask

    Args:

        axis (int): Axis of Tensor to do softmax. Default: -1

    """
    def __init__(self, axis: int = -1, **kwargs):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.softmax = ops.Softmax(get_integer(axis))

        self.large_neg = Tensor(-5e4, ms.float16)

    def construct(self, x: Tensor, mask: Tensor):
        """Compute softmax of Tensor with mask

        Args:
            x (Tensor):     Tensor with shape (..., X, ...). Data type is float.
            mask (Tensor):  Tensor with shape (..., X, ...). Data type is bool.

        Returns:
            output (Tensor):    Tensor with shape (..., X, ...). Data type is float.

        """

        xm = msnp.where(mask, x, self.large_neg)
        return self.softmax(xm)


class PositionalEmbedding(Cell):
    r"""Positional embedding to generate query, key and value for self-attention

    Args:

        dim (int):                      Last dimension of Tensor.

        use_distances (bool):           Whether to use distance information. Default: True

        use_bonds (bool):               Whether to use bond information. Default: False

        use_public_layer_norm (bool):   Whether to share layer normalization network. Default: True

    """
    def __init__(self,
                 dim: int,
                 use_public_layer_norm: bool = True,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        dim = get_integer(dim)

        if use_public_layer_norm:
            self.norm = nn.LayerNorm((dim,), -1, -1)
            self.x_norm = self.norm
            self.g_norm = self.norm
        else:
            self.x_norm = nn.LayerNorm((dim,), -1, -1)
            self.g_norm = nn.LayerNorm((dim,), -1, -1)

        self.x2q = Dense(dim, dim, has_bias=False)
        self.x2k = Dense(dim, dim, has_bias=False)
        self.x2v = Dense(dim, dim, has_bias=False)

    def construct(self,
                  x_i: Tensor,
                  g_ij: Tensor,
                  t: float = 0,
                  ):
        """Get query, key and query from atom types and positions

        Args:
            x_i (Tensor): Tensor with shape `(B, A, F)`. Data type is float.
            g_ij (Tensor): Tensor with shape `(B, A, A, F)`. Data type is float.

        Symbols:
            B:  Batch size
            A:  Number of atoms
            F:  Dimensions of feature space

        Returns:
            query (Tensor): Tensor with shape (B, A, 1, F). Data type is float.
            key (Tensor):   Tensor with shape (B, A, A, F). Data type is float.
            value (Tensor): Tensor with shape (B, A, A, F). Data type is float.

        """

        # (B, A, F) <- (B, F, A) <- (B, A, A, F)
        # g_i = g_ij.diagonal(0, -2, -3).swapaxes(-1, -2)

        # (B, 1, F) <- (B, F) <- (B, A, A, F)
        g_i = F.expand_dims(g_ij[..., 0, 0, :], -2)

        # (B, A, F) * (B, 1, F)
        xgi = F.mul(x_i, g_i)
        # (B, A, 1, F)
        xgii = F.expand_dims(xgi, -2)

        # (B, A, A, F) = (B, A, 1, F) * (B, A, A, F)
        xgij = F.mul(F.expand_dims(x_i, -3), g_ij)

        # (B, A, A, F)
        xgii = self.norm(xgii + t)
        xgij = self.norm(xgij + t)

        # (B, A, 1, F)
        query = self.x2q(xgii)
        # (B, A, A, F)
        key = self.x2k(xgij)
        # (B, A, A, F)
        value = self.x2v(xgij)

        return query, key, value


class MultiheadAttention(Cell):
    r"""Multi-head attention.

    Args:

        dim_feature (int):  Diension of feature space (F).

        n_heads (int):      Number of heads (h). Default: 8

        dim_tensor (int):   Dimension of input tensor (D). Default: 4

    Symbols:

        X:  Dimension to be aggregated

        F:  Dimension of Feature space

        h:  Number of heads for multi-head attention

        f:  Dimensions per head (F = f * h)

    """

    def __init__(self,
                 dim_feature: int,
                 n_heads: int = 8,
                 dim_tensor: int = 4,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        # D
        dim_tensor = get_integer(dim_tensor)
        if dim_tensor < 2:
            raise ValueError('dim_tensor must be larger than 1')

        # h
        self.n_heads = get_integer(n_heads)
        dim_feature = get_integer(dim_feature)

        # f = F / h
        self.size_per_head = dim_feature // self.n_heads
        # 1.0 / sqrt(f)
        self.scores_mul = 1.0 / msnp.sqrt(float(self.size_per_head))

        # shape = (h, f)
        self.reshape_tail = (self.n_heads, self.size_per_head)

        self.output = Dense(dim_feature, dim_feature, has_bias=False)

        self.softmax = ops.Softmax()
        self.bmm = ops.BatchMatMul()
        self.bmmt = ops.BatchMatMul(transpose_b=True)

        self.softmax_with_mask = SoftmaxWithMask()

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  mask: Tensor = None,
                  cutoff: Tensor = None
                  ):
        """Compute multi-head attention.

        Args:

            query (Tensor):     Tensor with shape (..., Q, F). Data type is float.
            key (Tensor):       Tensor with shape (..., X, F). Data type is float.
            value (Tensor):     Tensor with shape (..., X, F). Data type is float.
            mask (Tensor):      Tensor with shape (..., X). Data type is bool.
            cutoff (Tensor):    Tensor with shape (..., X). Data type is float.

        Returns:
            output (Tensor):    Tensor with shape (..., F). Data type is float.

        """
        #pylint: disable=invalid-name

        if self.n_heads > 1:
            # (..., h, Q, f) <- (..., Q, h, f) <- (..., Q, F)
            query_ = F.reshape(query, query.shape[:-1] + self.reshape_tail).swapaxes(-2, -3)
            # (..., h, X, f) <- (..., X, h, f) <- (..., X, F)
            key_ = F.reshape(key, key.shape[:-1] + self.reshape_tail).swapaxes(-2, -3)
            # (..., h, X, f) <- (..., X, h, f) <- (..., X, F)
            value_ = F.reshape(value, value.shape[:-1] + self.reshape_tail).swapaxes(-2, -3)

            # (..., h, Q, X) = (..., h, Q, f) x (..., h, X, f)^T
            attention_scores = self.bmmt(query_, key_)
            # (..., h, Q, X) / sqrt(f)
            attention_scores = F.mul(attention_scores, self.scores_mul)

            if mask is None:
                attention_probs = self.softmax(attention_scores)
            else:
                attention_probs = self.softmax_with_mask(attention_scores, mask.expand_dims(-2).expand_dims(-2))

                if cutoff is not None:
                    # (..., h, Q, X) <- (..., 1, 1, X) <- (..., X)
                    excut = cutoff.expand_dims(-2).expand_dims(-2)
                    # (..., h, Q, X) = (..., h, X, X) * (..., 1, 1, X)
                    attention_probs = F.mul(attention_probs, excut)

            # (..., h, Q, f) = (..., h, Q, X) x (..., h, X, f)
            context = self.bmm(attention_probs, value_)
            # (..., Q, h, f)
            context = context.swapaxes(-2, -3)
            # (..., Q, F)
            context = F.reshape(context, query.shape)

        else:
            # (..., Q, X) = (..., Q, F) x (..., Q, F)^T / sqrt(F)
            attention_scores = self.bmmt(query, key) * self.scores_mul

            if mask is None:
                # (..., Q, X)
                attention_probs = self.softmax(attention_scores)
            else:
                # (..., 1, X)
                attention_probs = self.softmax_with_mask(attention_scores, F.expand_dims(mask, -2))

                if cutoff is not None:
                    # (..., 1, X) * (..., 1, X)
                    attention_probs = attention_probs * F.expand_dims(cutoff, -2)

            # (..., Q, F) = (..., Q, X) x (..., X, F)
            context = self.bmm(attention_probs, value)

        # (..., Q, F)
        return self.output(context)


class FeedForward(Cell):
    r"""Feed forward network for transformer.

    Args:

        dim (int):          Last dimension of Tensor

        activation (Cell):  Activation function.

        n_hidden (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim: int,
                 activation: Cell,
                 n_hidden: int = 1,
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        dim = get_integer(dim)
        self.norm = nn.LayerNorm((dim,), -1, -1)
        self.residual = Residual(dim, activation=activation, n_hidden=get_integer(n_hidden))

    def construct(self, x: Tensor):
        """Compute feed forward network.

        Args:

            x (Tensor): Tensor with shape (B, A, F). Data type is float.

        Returns:
            y (Tensor): Tensor with shape (B, A, F). Data type is float.

        """

        nx = self.norm(x)
        return self.residual(nx)


class Pondering(Cell):
    r"""Pondering network for adapetive computation time.

    Args:

        n_in (int):         Dimension of input Tensor

        n_hidden (int):     Number of hidden layers. Default: 0

        bias_const (float): Initial value for bias. Default: 1

    """
    def __init__(self,
                 n_in: int,
                 n_hidden: int = 0,
                 bias_const: float = 1.,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        n_in = get_integer(n_in)
        n_hidden = get_integer(n_hidden)

        if n_hidden == 0:
            self.dense = nn.Dense(n_in, 1, has_bias=True, weight_init='xavier_uniform', bias_init=Constant(
                bias_const), activation='sigmoid',)
        elif n_hidden > 0:
            nets = []
            for _ in range(n_hidden):
                nets.append(
                    nn.Dense(n_in, n_in, weight_init='xavier_uniform', activation='relu'))
            nets.append(nn.Dense(n_in, 1, bias_init=Constant(bias_const), activation='sigmoid'))
            self.dense = nn.SequentialCell(nets)
        else:
            raise ValueError("n_hidden cannot be negative!")

    def construct(self, x: Tensor):
        """Calculate pondering network.

        Args:

            x (Tensor): Tensor with shape (B, A, X). Data type is float.

        Returns:
            y (Tensor): Tensor with shape (B, A, 1). Data type is float.

        """
        y = self.dense(x)
        return squeeze_last_dim(y)

class ACTWeight(Cell):
    r"""Adapetive computation time modified from:
        https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/UTransformer.py

    Args:

        n_in (int):         Dimension of input Tensor

        n_hidden (int):     Number of hidden layers. Default: 0

        bias_const (float): Initial value for bias. Default: 1

    """
    def __init__(self, threshold: float = 0.9, **kwargs):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.threshold = threshold

    def construct(self, prob: Tensor, halting_prob: Tensor):
        """Calculate Adapetive computation time.

        Args:

            prob (Tensor):          Tensor with shape (B, A, 1). Data type is float.
            halting_prob (Tensor):  Tensor with shape (B, A, 1). Data type is float.

        Returns:
            w (Tensor):     Tensor with shape (B, A, 1). Data type is float.
            dp (Tensor):    Tensor with shape (B, A, 1). Data type is float.
            dn (Tensor):    Tensor with shape (B, A, 1). Data type is float.

        """

        # Mask for inputs which have not halted last cy
        running = F.cast(halting_prob < 1.0, prob.dtype)

        # Add the halting probability for this step to the halting
        # probabilities for those input which haven't halted yet
        add_prob = prob * running
        new_prob = halting_prob + add_prob
        mask_run = F.cast(new_prob <= self.threshold, prob.dtype)
        mask_halt = F.cast(new_prob > self.threshold, prob.dtype)

        # Mask of inputs which haven't halted, and didn't halt this step
        still_running = mask_run * running
        running_prob = halting_prob + prob * still_running

        # Mask of inputs which halted at this step
        new_halted = mask_halt * running

        # Compute remainders for the inputs which halted at this step
        remainders = new_halted * (1.0 - running_prob)

        # Add the remainders to those inputs which halted at this step
        dp = add_prob + remainders

        # Increment n_updates for all inputs which are still running
        dn = running

        # Compute the weight to be applied to the new state and output
        # 0 when the input has already halted
        # prob when the input hasn't halted yet
        # the remainders when it halted this step
        update_weights = prob * still_running + new_halted * remainders
        w = F.expand_dims(update_weights, -1)

        return w, dp, dn
