import torch
from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, num_steps, input_size)
        y: Tensor(shape=(batch_size, num_steps)
        """
        x = torch.cat((x, y.unsqueeze(-1)), dim=2)
        _, (h, c) = self.lstm(x)
        return h, c


class Likelihood(nn.Module):

    def __init__(self, in_features, distribution):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=2
        )
        self.distribution = distribution
        self.softplus = nn.Softplus()

    def mean_transform(self, h):
        return NotImplementedError()

    def scale_transform(self, h):
        return NotImplementedError()

    def transform(self, h):
        h = self.fc(h)
        mean_proj, scale_proj = torch.split(h, 1, 2)
        mean = self.mean_transform(mean_proj.squeeze())
        scale = self.scale_transform(scale_proj.squeeze())
        return mean, scale

    def loss(self, h, y):
        raise NotImplementedError()

    def sample(self, h, num_samples):
        raise NotImplementedError()


class GaussianLikelihood(Likelihood):

    def __init__(self, in_features):
        super().__init__(in_features=in_features, distribution=Normal)

    def mean_transform(self, h):
        return h

    def scale_transform(self, h):
        return self.softplus(h)

    def loss(self, h, y):
        mu, sigma = self.transform(h)
        return -self.distribution(mu, sigma).log_prob(y).mean()

    def sample(self, h, num_samples):
        """
        Parameters
        ----------
        h: Tensor(shape=[batch_size, num_steps, hidden_size)

        Returns
        -------
        Tensor(shape=[num_samples, batch_size, num_steps])
        """
        mu, sigma = self.transform(h)
        return self.distribution(mu, sigma).sample(sample_shape=torch.Size([num_samples]))


class NegativeBinomialLikelihood(Likelihood):

    def __init__(self, in_features):
        super().__init__(in_features=in_features, distribution=NegativeBinomial)

    def mean_transform(self, h):
        return self.softplus(h)

    def scale_transform(self, h):
        return self.softplus(h)

    def loss(self, h, y):
        mu, alpha = self.transform(h)
        p = alpha*mu/(1 + alpha*mu)
        r = 1/alpha
        loss = -self.distribution(total_count=r, probs=p).log_prob(y).mean()
        return loss.mean()


class Decoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_decode_steps,
        likelihood=None
    ):
        super().__init__()

        self.num_decode_steps = num_decode_steps

        self.lstm = nn.LSTM(
            input_size=input_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        if likelihood is None:
            likelihood = GaussianLikelihood(in_features=hidden_size)

        self.likelihood = likelihood

    def forward(self, x, y, h0, c0):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, num_decode_steps, input_size)
        y: Tensor(shape=(batch_size, num_decode_steps)
        h0: Tensor(shape=(1, batch_size, hidden_size))
        c0: Tensor(shape=(1, batch_size, hidden_size))

        Returns
        -------
        (mean, scale)

        2*(Tensor(shape=(batch_size, num_decode_steps),)
        """
        x = torch.cat((x, y.unsqueeze(-1)), dim=2)
        h, _ = self.lstm(x, (h0, c0))
        mean, scale = self.likelihood.transform(h)
        return mean, scale

    def predict(self, x, y0, h0, c0, num_samples):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, num_decode_steps, input_size)
        y0: Tensor(shape=(batch_size)
        h0: Tensor(shape=(1, batch_size, hidden_size))
        c0: Tensor(shape=(1, batch_size, hidden_size))

        Returns
        -------
        (mean, scale)

        2*(Tensor(shape=(batch_size, num_decode_steps),)
        """
        ct = c0
        ht = h0
        yt = y0
        for t in range(self.num_decode_steps):
            xt = x[:, t, :]
            xt = torch.cat((xt.unsqueeze(1), yt.reshape(-1, 1, 1)), dim=-1)
            _, (ht, ct) = self.lstm(xt, (ht, ct))
            sample = self.likelihood.sample(ht, num_samples=num_samples)
        return sample


def run():
    N = 8
    T = 365
    D = 4
    H = 128
    num_decode_steps = 7
    num_layers = 2

    xe = torch.randn(N, T, D)
    ye = torch.randn(N, T)

    enc = Encoder(input_size=D, hidden_size=H, num_layers=num_layers)
    h0, c0 = enc(xe, ye)

    xd = torch.randn(N, num_decode_steps, D)
    yd = torch.randn(N, num_decode_steps)
    y0 = torch.randn(N)
    dec = Decoder(input_size=D, hidden_size=H, num_layers=num_layers, num_decode_steps=num_decode_steps)
    pred = dec(xd, y0, h0, c0)
    return locals()
