"""Regularizers for RNN units.

Mostly copied from fast.ai v0.7.
"""
import torch
import torch.nn as nn


def dropout_mask(x, sz, dropout):
    """ Applies a dropout mask whose size is determined by passed argument 'sz'.
    Args:
        x (torch.Tensor): A torch Variable object
        sz (tuple(int, int, int)): The expected size of the new tensor
        dropout (float): The dropout fraction to apply

    This method uses the bernoulli distribution to decide which activations to keep.
    Additionally, the sampled activations is rescaled is using the factor 1/(1 - dropout).

    In the example given below, one can see that approximately .8 fraction of the
    returned tensors are zero. Rescaling with the factor 1/(1 - 0.8) returns a tensor
    with 5's in the unit places.

    The official link to the pytorch bernoulli function is here:
        http://pytorch.org/docs/master/torch.html#torch.bernoulli

    Examples:
        >>> a_Var = torch.autograd.Variable(torch.Tensor(2, 3, 4).uniform_(0, 1), requires_grad=False)
        >>> a_Var
            Variable containing:
            (0 ,.,.) =
              0.6890  0.5412  0.4303  0.8918
              0.3871  0.7944  0.0791  0.5979
              0.4575  0.7036  0.6186  0.7217
            (1 ,.,.) =
              0.8354  0.1690  0.1734  0.8099
              0.6002  0.2602  0.7907  0.4446
              0.5877  0.7464  0.4257  0.3386
            [torch.FloatTensor of size 2x3x4]
        >>> a_mask = dropout_mask(a_Var.data, (1,a_Var.size(1),a_Var.size(2)), dropout=0.8)
        >>> a_mask
            (0 ,.,.) =
              0  5  0  0
              0  0  0  5
              5  0  5  0
            [torch.FloatTensor of size 1x3x4]
    """
    return x.new_empty(*sz).bernoulli_(1-dropout)/(1-dropout)


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p:
            return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return m * x


def noop(*args, **kwargs): return


class WeightDrop(nn.Module):
    """A custom torch layer that serves as a wrapper on another torch layer.
    Primarily responsible for updating the weights in the wrapped module based
    on a specified dropout.
    """

    def __init__(self, module, dropout, weights=['weight_hh_l0']):
        """ Default constructor for the WeightDrop module

        Args:
            module (torch.nn.Module): A pytorch layer being wrapped
            dropout (float): a dropout value to apply
            weights (list(str)): the parameters of the wrapped **module**
                which should be fractionally dropped.
        """
        super().__init__()
        self.module, self.weights, self.dropout = module, weights, dropout
        self._setup()

    def _setup(self):
        """ for each string defined in self.weights, the corresponding
        attribute in the wrapped module is referenced, then deleted, and subsequently
        registered as a new parameter with a slightly modified name.

        Args:
            None

         Returns:
             None
        """
        if isinstance(self.module, nn.RNNBase):
            self.module.flatten_parameters = noop
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(
                name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        """ Uses pytorch's built-in dropout function to apply dropout to the parameters of
        the wrapped module.

        Args:
            None
        Returns:
            None
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = nn.functional.dropout(
                raw_w, p=self.dropout, training=self.training)
            if hasattr(self.module, name_w):
                delattr(self.module, name_w)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        """ updates weights and delegates the propagation of the tensor to the wrapped module's
        forward method

        Args:
            *args: supplied arguments

        Returns:
            tensor obtained by running the forward method on the wrapped module.
        """
        self._setweights()
        return self.module.forward(*args)
