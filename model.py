import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, enc_dims: list, dec_dims: list, device='cpu', anneal_num=200000, anneal_cap=0.2, dropout=0.2):
        super(VAE, self).__init__()
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.device = device

        self.anneal_num = anneal_num
        self.anneal_cap = anneal_cap
        self.anneal = 0

        self.update_cnt = 0
        self.dropout = dropout

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        assert enc_dims[-1] == dec_dims[0], '编码器和解码器的瓶颈层维度不一致:('

        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
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

        mu_q = x[:, :self.enc_dims[-1]]
        logvar_q = x[:, self.enc_dims[-1]:]
        std_q = torch.exp(0.5 * logvar_q)

        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        y = sampled_z
        for layer in self.decoder:
            y = layer(y)

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return y, kl_loss
        else:
            return y
