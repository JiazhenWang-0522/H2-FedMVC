import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import normalize

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class feature_R_module(nn.Module):
    def __init__(self, feature_dim):
        super(feature_R_module, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 15)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity()
        if feature_dim != 256:
            self.shortcut = nn.Linear(feature_dim, 256)

    def forward(self, z):
        residual = self.shortcut(z)
        out = self.relu(self.fc1(z))
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        out = self.fc3(out)
        return out


class Network(nn.Module):
    def __init__(self, view, num_view, input_size, feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.feature_Rs = []
        self.view_num = view
        self.num_view = num_view
        for v in range(self.view_num):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
            self.feature_Rs.append(feature_R_module(feature_dim).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_Rs = nn.ModuleList(self.feature_Rs)
        self.centroids = Parameter(torch.Tensor(class_num, 15))

        self.feature_H_module = nn.Sequential(
            nn.Linear(self.view_num*feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 15)
        )


    def forward(self, xs):

        xrs = []
        zs = []
        rs = []
        for v in range(self.view_num):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = self.feature_Rs[v](z)
            zs.append(z)
            xrs.append(xr)
            rs.append(r)
        zzs = torch.cat(zs, dim=1)
        h = normalize(self.feature_H_module(zzs), dim=1)
        return xrs, zs, h, rs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs




