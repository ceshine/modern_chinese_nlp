import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from dekisugi.rnn_reg import LockedDropout


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight_g)
        nn.init.kaiming_normal_(self.conv1.weight_v)
        nn.init.kaiming_normal_(self.conv2.weight_g)
        nn.init.kaiming_normal_(self.conv2.weight_v)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            nn.init.constant_(self.downsample.bias, 0)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dropouti=0.1, dilation_sizes=[]):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        self.dropouti = LockedDropout(dropouti)
        if dilation_sizes:
            assert len(dilation_sizes) == len(num_channels)
        for i in range(num_levels):
            if len(dilation_sizes) == 0:
                dilation_size = 2 ** i
            else:
                dilation_size = dilation_sizes[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size, dropout=dropout
                )
            )
        self.layer_groups = nn.ModuleList(layers)

    def forward(self, x):
        """Feed Forward

        Arguments:
            x {toch.FloatTensor} -- shape [timesteps, batch_size, channels]

        Returns:
            toch.FloatTensor -- shape [n_layers, batch_size, channels, timesteps]
            None -- To be compatible with RNN API
        """
        x = self.dropouti(x)
        x = x.permute(1, 2, 0)
        outputs = []
        for layer in self.layer_groups:
            x = layer(x)
            outputs.append(x)
        return torch.stack(outputs, dim=0), None

    def reset(self):
        pass
