import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from cybertroncode.units import units
from cybertroncode.blocks import MLP,Dense
from cybertroncode.base import SoftmaxWithMask
from cybertroncode.base import MultiheadAttention

__all__ = [
    "Aggregator",
    "get_aggregator",
    "TensorSummation",
    "TensorMean",
    "SoftmaxGeneralizedAggregator",
    "PowermeanGeneralizedAggregator",
    "ListAggregator",
    "get_list_aggregator",
    "ListSummation",
    "ListMean",
    "LinearTransformation",
    "MultipleChannelRepresentation",
    ]

_AGGREGATOR_ALIAS = dict()
_LIST_AGGREGATOR_ALIAS = dict()

def _aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _AGGREGATOR_ALIAS:
            _AGGREGATOR_ALIAS[name] = cls

        for alias in aliases:
            if alias not in _AGGREGATOR_ALIAS:
                _AGGREGATOR_ALIAS[alias] = cls

        return cls

    return alias_reg

def _list_aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _LIST_AGGREGATOR_ALIAS:
            _LIST_AGGREGATOR_ALIAS[name] = cls

        for alias in aliases:
            if alias not in _LIST_AGGREGATOR_ALIAS:
                _LIST_AGGREGATOR_ALIAS[alias] = cls

        return cls

    return alias_reg

class Aggregator(nn.Cell):
    def __init__(self,
        dim=None,
        axis=-2,
    ):
        super().__init__()

        self.name = 'aggregator'

        self.dim = dim
        self.axis = axis
        self.reduce_sum = P.ReduceSum()

class ListAggregator(nn.Cell):
    def __init__(self,
        dim=None,
        num_agg=None,
        n_hidden=0,
        activation=None,
    ):
        super().__init__()

        self.dim = dim
        self.num_agg = num_agg
        self.n_hidden = n_hidden
        self.activation = activation

        self.stack = P.Stack(-1)
        self.reduce_sum = P.ReduceSum()

@_aggregator_register('sum')
class TensorSummation(Aggregator):
    def __init__(self,
        dim=None,
        axis=-2,
    ):
        super().__init__(
            dim=None,
            axis=axis,
        )

        self.name = 'sum'

    def __str__(self):
        return "sum"

    def construct(self,nodes,node_mask=None,nodes_number=None):
        if node_mask is not None:
            nodes = nodes * node_mask
        agg = self.reduce_sum(nodes,self.axis)
        return agg

@_aggregator_register('mean')
class TensorMean(Aggregator):
    def __init__(self,
        dim=None,
        axis=-2,
    ):
        super().__init__(
            dim=None,
            axis=axis,
        )
        self.name = 'mean'

        self.reduce_mean = P.ReduceMean()
        self.mol_sum = P.ReduceSum(keep_dims=True)

    def __str__(self):
        return "mean"

    def construct(self,nodes,node_mask=None,nodes_number=None):
        if node_mask is not None:
            nodes = nodes * node_mask
            agg = self.reduce_sum(nodes,self.axis)
            return agg / nodes_number
        else:
            return self.reduce_mean(nodes,self.axis)

# Softmax-based generalized mean-max-sum aggregator
@_aggregator_register('softmax')
class SoftmaxGeneralizedAggregator(Aggregator):
    def __init__(self,
        dim,
        axis=-2,
    ):
        super().__init__(
            dim=dim,
            axis=axis,
        )

        self.name = 'softmax'

        self.beta  = ms.Parameter(initializer('one', 1), name="beta")
        self.rho = ms.Parameter(initializer('zero', 1), name="rho")

        self.softmax = P.Softmax(axis=self.axis)
        self.softmax_with_mask = SoftmaxWithMask(axis=self.axis)
        self.mol_sum = P.ReduceSum(keep_dims=True)

        self.expand_ones = P.Ones()((1,1,self.dim),ms.int32)

    def __str__(self):
        return "softmax"

    def construct(self, nodes, node_mask=None, nodes_number=None):
        if nodes_number is None:
            nodes_number = nodes.shape[self.axis]

        scale = nodes_number / (1 + self.beta * (nodes_number - 1))
        px = nodes * self.rho

        if node_mask is None:
            agg_nodes = self.softmax(px) * nodes
        else:
            mask = (self.expand_ones * node_mask) > 0
            agg_nodes = self.softmax_with_mask(px,mask) * nodes * node_mask
        
        agg_nodes = self.reduce_sum(agg_nodes,self.axis)

        return scale * agg_nodes

# PowerMean-based generalized mean-max-sum aggregator
@_aggregator_register('powermean')
class PowermeanGeneralizedAggregator(Aggregator):
    def __init__(self,
        dim,
        axis=-2,
    ):
        super().__init__(
            dim=dim,
            axis=axis,
        )
        self.name = 'powermean'
        self.beta  = ms.Parameter(initializer('one', 1), name="beta")
        self.rho = ms.Parameter(initializer('one', 1), name="rho")

        self.power = P.Pow()
        self.mol_sum = P.ReduceSum(keep_dims=True)

    def __str__(self):
        return "powermean"

    def construct(self, nodes, node_mask=None, nodes_number=None):
        if nodes_number is None:
            nodes_number = nodes.shape[self.axis]

        scale = nodes_number / (1 + self.beta * (nodes_number - 1))
        xp = self.power(nodes,self.rho)
        if node_mask is not None:
            xp = xp * node_mask
        agg_nodes = self.reduce_sum(xp,self.axis)

        return self.power(scale*agg_nodes,1.0/self.rho)

@_aggregator_register('transformer')
class TransformerAggregator(Aggregator):
    def __init__(self,
        dim,
        axis=-2,
        n_heads=8,
    ):
        super().__init__(
            dim=dim,
            axis=axis,
        )

        self.name = 'transformer'

        self.a2q = Dense(dim,dim,has_bias=False)
        self.a2k = Dense(dim,dim,has_bias=False)
        self.a2v = Dense(dim,dim,has_bias=False)

        self.layer_norm = nn.LayerNorm((dim,),-1,-1)
        
        self.multi_head_attention = MultiheadAttention(dim,n_heads,dim_tensor=3)

        self.squeeze = P.Squeeze(-1)
        self.mean = TensorMean(dim,axis)

    def __str__(self):
        return "transformer"

    def construct(self, nodes, node_mask=None, nodes_number=None):
        r"""Transformer type aggregator.

        Args:
            nodes (Mindspore.Tensor[float] [B, A, F]):

        Returns:
            Mindspore.Tensor [..., F]: multi-head attention output.

        """
        # [B, A, F]
        x =  self.layer_norm(nodes)

        # [B, A, F]
        Q = self.a2q(x)
        K = self.a2k(x)
        V = self.a2v(x)

        if node_mask is not None:
            mask = self.squeeze(node_mask)
        else:
            mask = node_mask

        # [B, A, F]
        x = self.multi_head_attention(Q,K,V,mask)

        # [B, 1, F]
        return self.mean(x,node_mask,nodes_number)

@_list_aggregator_register('sum')
class ListSummation(ListAggregator):
    def __init__(self,
        dim=None,
        num_agg=None,
        n_hidden=0,
        activation=None,
    ):
        super().__init__(
            dim=None,
            num_agg=None,
            n_hidden=0,
            activation=None,
        )

    def __str__(self):
        return "sum"

    def construct(self,xlist,node_mask=None):
        xt = self.stack(xlist)
        y = self.reduce_sum(xt,-1)
        if node_mask is not None:
            y = y * node_mask
        return y

@_list_aggregator_register('mean')
class ListMean(ListAggregator):
    def __init__(self,
        dim=None,
        num_agg=None,
        n_hidden=0,
        activation=None,
    ):
        super().__init__(
            dim=None,
            num_agg=None,
            n_hidden=0,
            activation=None,
        )

        self.reduce_mean = P.ReduceMean()

    def __str__(self):
        return "mean"

    def construct(self,xlist,node_mask=None):
        xt = self.stack(xlist)
        y = self.reduce_mean(xt,-1)
        if node_mask is not None:
            y = y * node_mask
        return y

@_list_aggregator_register('linear')
class LinearTransformation(ListAggregator):
    def __init__(self,
        dim,
        num_agg=None,
        n_hidden=0,
        activation=None,
    ):
        super().__init__(
            dim=dim,
            num_agg=None,
            n_hidden=0,
            activation=None,
        )
        self.scale = ms.Parameter(initializer(Normal(1.0),[self.dim,]), name="scale")
        self.shift = ms.Parameter(initializer(Normal(1.0),[self.dim,]), name="shift")

    def __str__(self):
        return "linear"

    def construct(self,ylist,node_mask=None):
        yt = self.stack(ylist)
        ysum = self.reduce_sum(yt,-1)
        y = self.scale * ysum + self.shift
        if node_mask is not None:
            y = y * node_mask
        return y

# Multiple-Channel Representation Readout
@_list_aggregator_register('mcr')
class MultipleChannelRepresentation(ListAggregator):
    def __init__(self,
        dim,
        num_agg,
        n_hidden=0,
        activation=None,
    ):
        super().__init__(
            dim=dim,
            num_agg=num_agg,
            n_hidden=n_hidden,
            activation=activation,
        )

        sub_dim = self.dim // self.num_agg
        last_dim = self.dim - (sub_dim * (self.num_agg - 1))
        sub_dims = [sub_dim for _ in range(self.num_agg - 1)]
        sub_dims.append(last_dim)

        if self.n_hidden > 0:
            hidden_layers = [dim,] * self.n_hidden
            self.mcr = nn.CellList([
                MLP(self.dim, sub_dims[i], hidden_layers, activation=self.activation)
                for i in range(self.um_agg)
                ])
        else:
            self.mcr = nn.CellList([
                Dense(self.dim, sub_dims[i], activation=self.activation)
                for i in range(self.num_agg)
                ])

        self.concat = P.Concat(-1)

    def __str__(self):
        return "MCR"

    def construct(self,xlist,node_mask=None):
        Xt = ()
        for i in range(self.num_agg):
            Xt = Xt + (self.mcr[i](xlist[i]),)
        y = self.concat(Xt)
        if node_mask is not None:
            y = y * node_mask
        return y

def get_aggregator(
        obj,
        dim,
        axis=-2
    ):
    if obj is None or isinstance(obj,Aggregator):
        return obj
    elif isinstance(obj, str):
        if obj not in _AGGREGATOR_ALIAS.keys():
            raise ValueError("The class corresponding to '{}' was not found.".format(obj))
        return _AGGREGATOR_ALIAS[obj.lower()](dim=dim,axis=axis)
    else:
        raise TypeError("Unsupported Aggregator type '{}'.".format(type(obj)))

def get_list_aggregator(
        obj,
        dim,
        num_agg,
        n_hidden=0,
        activation=None,
    ):
    if obj is None or isinstance(obj,ListAggregator):
        return obj
    elif isinstance(obj, str):
        if obj not in _LIST_AGGREGATOR_ALIAS.keys():
            raise ValueError("The class corresponding to '{}' was not found.".format(obj))
        return _LIST_AGGREGATOR_ALIAS[obj.lower()](
            dim=dim,
            num_agg=num_agg,
            n_hidden=n_hidden,
            activation=activation,
        )
    else:
        raise TypeError("Unsupported ListAggregator type '{}'.".format(type(obj)))