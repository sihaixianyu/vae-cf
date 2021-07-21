import torch.nn as nn
import torch.nn.functional as F


class DAE(nn.Module):
    def __init__(self, enc_dims: list, dec_dims: list, device='cpu', dropout=0.2):
        super(DAE, self).__init__()
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.device = device

        self.dropout = dropout

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        assert enc_dims[-1] == dec_dims[0], '编码器和解码器的瓶颈层维度不一致:('

        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

        # 将模型放到cpu或gpu运行
        self.to(self.device)

    def forward(self, x):
        x = F.dropout(F.normalize(x), p=self.dropout, training=self.training)
        for layer in self.encoder:
            x = layer(x)

        y = x
        for layer in self.decoder:
            y = layer(y)

        return y
