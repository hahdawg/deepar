import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_cats,
        embedding_dim,
        _lstm=None,
        _embedding=None
    ):
        super().__init__()
        if _lstm is None:
            _lstm = nn.LSTM(
                input_size=input_size + embedding_dim + 1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        self.lstm = _lstm

        if _embedding is None:
            _embedding = nn.Embedding(
                num_embeddings=num_cats,
                embedding_dim=embedding_dim
            )
        self.embedding = _embedding

    def forward(self, x, cat, y):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, num_steps, input_size)
        cat: Tensor(shape=[batch_size, num_steps])
        y: Tensor(shape=(batch_size, num_steps)
        """
        cat = self.embedding(cat)
        x = torch.cat((x, cat, y.unsqueeze(-1)), dim=2)
        _, (h, c) = self.lstm(x)
        return h, c

    def sample(self, h, num_samples, factor):
        """
        Parameters
        ----------
        h: Tensor(shape=[batch_size, num_steps, hidden_size)

        Returns
        -------
        Tensor(shape=[num_samples, batch_size, num_steps])
        """
        raise NotImplementedError()


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

    def shape_transform(self, h):
        return NotImplementedError()

    def factor_transform(self, mean, shape, factor):
        raise NotImplementedError()

    def transform(self, h, factor):
        h = self.fc(h)
        mean_proj, shape_proj = torch.split(h, 1, 2)
        mean = self.mean_transform(mean_proj.squeeze())
        shape = self.shape_transform(shape_proj.squeeze())
        mean, shape = self.factor_transform(mean, shape, factor)
        return mean, shape

    def loss(self, h, y, factor):
        raise NotImplementedError()

    def sample(self, h, num_samples, factor):
        """
        Parameters
        ----------
        h: Tensor(shape=[batch_size, num_steps, hidden_size)

        Returns
        -------
        Tensor(shape=[num_samples, batch_size, num_steps])
        """
        raise NotImplementedError()


class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood.
    """

    def __init__(self, in_features):
        super().__init__(in_features=in_features, distribution=Normal)

    def mean_transform(self, h):
        return h

    def shape_transform(self, h):
        return self.softplus(h)

    def factor_transform(self, mean, shape, factor):
        mean = mean*factor
        shape = shape*factor
        return mean, shape

    def loss(self, h, y, factor):
        mu, sigma = self.transform(h, factor)
        return -self.distribution(mu, sigma).log_prob(y).mean()

    def sample(self, h, num_samples, factor):
        """
        Parameters
        ----------
        h: Tensor(shape=[batch_size, num_steps, hidden_size)

        Returns
        -------
        Tensor(shape=[num_samples, batch_size, num_steps])
        """
        mean, shape = self.transform(h, factor)
        return self.distribution(mean, shape).sample(sample_shape=torch.Size([num_samples]))


class NegativeBinomialLikelihood(Likelihood):
    """
    Negative binomial likelihood.
    """
    def __init__(self, in_features):
        super().__init__(in_features=in_features, distribution=NegativeBinomial)

    def mean_transform(self, h):
        return self.softplus(h)

    def shape_transform(self, h):
        return self.softplus(h)

    def factor_transform(self, mean, shape, factor):
        shape = shape + 1e-6
        mean = mean*factor
        mean_shape = mean*shape
        p = mean_shape/(1 + mean_shape)
        r = 1/shape
        return p, r

    def loss(self, h, y, factor):
        p, r = self.transform(h, factor)
        loss = -self.distribution(total_count=r, probs=p).log_prob(y).mean()
        return loss

    def sample(self, h, num_samples, factor):
        """
        Parameters
        ----------
        h: Tensor(shape=[batch_size, num_steps, hidden_size)

        Returns
        -------
        Tensor(shape=[num_samples, batch_size, num_steps])
        """
        p, r = self.transform(h, factor)
        res = self.distribution(total_count=r, probs=p).sample(sample_shape=torch.Size([num_samples]))
        return res


class Decoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_cats,
        embedding_dim,
        num_decode_steps,
        likelihood=None,
        _lstm=None,
        _embedding=None
    ):
        super().__init__()

        self.num_decode_steps = num_decode_steps

        if _lstm is None:
            _lstm = nn.LSTM(
                input_size=input_size + embedding_dim + 1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        self.lstm = _lstm

        if likelihood is None:
            likelihood = GaussianLikelihood
        self.likelihood = likelihood(in_features=hidden_size)

        if _embedding is None:
            _embedding = nn.Embedding(
                num_embeddings=num_cats,
                embedding_dim=embedding_dim
            )
        self.embedding = _embedding

    def forward(self, x, cat, y, h0, c0):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, num_decode_steps, input_size)
        cat: Tensor(shape=(batch_size, num_decode_steps))
        y: Tensor(shape=(batch_size, num_decode_steps)
        h0: Tensor(shape=(1, batch_size, hidden_size))
        c0: Tensor(shape=(1, batch_size, hidden_size))

        Returns
        -------
        (mean, shape)

        2*(Tensor(shape=(batch_size, num_decode_steps),)
        """
        cat = self.embedding(cat)
        x = torch.cat((x, cat, y.unsqueeze(-1)), dim=2)
        h, _ = self.lstm(x, (h0, c0))
        return h

    def predict(self, x, cat, y0, h0, c0, factor, num_samples):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, num_decode_steps, input_size)
        cat: Tensor(shape=(batch_size, num_decode_steps))
        y0: Tensor(shape=(batch_size)
        h0: Tensor(shape=(1, batch_size, hidden_size))
        c0: Tensor(shape=(1, batch_size, hidden_size))
        factor: Tensor(shape=(batch_size, 1))

        Returns
        -------
        Tensor(shape=(num_samples, batch_size, num_decode_steps))
        """
        ct = c0
        ht = h0
        yt = y0
        preds = []
        for t in range(self.num_decode_steps):
            xt = x[:, t, :]
            cat_t = self.embedding(cat[:, t].unsqueeze(-1))
            xt = torch.cat((xt.unsqueeze(1), cat_t, yt.reshape(-1, 1, 1)), dim=-1)
            _, (ht, ct) = self.lstm(xt, (ht, ct))
            sample = self.likelihood.sample(
                ht[-1, :, :].unsqueeze(1),
                num_samples=num_samples,
                factor=factor
            )
            yt = sample.mean(axis=0).float()
            preds.append(sample)
        preds = torch.stack(preds).permute(1, 2, 0)
        return preds


class DeepAR(nn.Module):
    """
    DeepAR model with separate encoder and decoder.
    """
    def __init__(
        self,
        input_size_encoder,
        input_size_decoder,
        num_cats_encoder,
        num_cats_decoder,
        embedding_dim_encoder,
        embedding_dim_decoder,
        hidden_size,
        num_layers,
        num_decode_steps,
        likelihood,
        _lstm=None,
        _embedding=None
    ):
        super().__init__()

        self.encoder = Encoder(
            input_size=input_size_encoder,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_cats=num_cats_encoder,
            embedding_dim=embedding_dim_encoder,
            _lstm=_lstm,
            _embedding=_embedding
        )

        self.decoder = Decoder(
            input_size=input_size_decoder,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_cats=num_cats_decoder,
            embedding_dim=embedding_dim_decoder,
            num_decode_steps=num_decode_steps,
            likelihood=likelihood,
            _lstm=_lstm,
            _embedding=_embedding
        )

    def forward(self, x_encode, cat_encode, y_encode, x_decode, cat_decode, y_decode, factor, **kwargs):
        """
        Parameters
        ----------
        x_encode: Tensor(shape=(batch_size, num_encode_steps, input_size_encoder)
        cat_encode: Tensor(shape=(batch_size, num_encode_steps))
        y_encode: Tensor(shape=(batch_size, num_encode_steps)
        x_decode: Tensor(shape=(batch_size, num_decode_steps, input_size_decoder)
        cat_decode: Tensor(shape=(batch_size, num_decode_steps))
        y_decode: Tensor(shape=(batch_size, num_decode_steps)
        """
        y_encode /= factor
        y_decode /= factor
        h0, c0 = self.encoder(x=x_encode, cat=cat_encode, y=y_encode)
        h = self.decoder(x=x_decode, cat=cat_decode, y=y_decode, h0=h0, c0=c0)
        return h

    def predict(
        self,
        x_encode,
        cat_encode,
        y_encode,
        x_decode,
        cat_decode,
        num_samples,
        factor,
        **kwargs
    ):
        """
        Parameters
        ----------
        x_encode: Tensor(shape=(batch_size, num_encode_steps, input_size)
        cat_encode: Tensor(shape=(batch_size, num_encode_steps))
        y_encode: Tensor(shape=(batch_size, num_encode_steps)
        x_decode: Tensor(shape=(batch_size, num_decode_steps, input_size_decoder)
        cat_decode: Tensor(shape=(batch_size, num_decode_steps))
        num_samples: int

        Returns
        -------
        Tensor(shape=(num_samples, batch_size, num_decode_steps))
        """
        y_encode /= factor
        h0, c0 = self.encoder(x=x_encode, cat=cat_encode, y=y_encode)
        return self.decoder.predict(
            x=x_decode,
            cat=cat_decode,
            y0=y_encode[:, -1],
            h0=h0,
            c0=c0,
            factor=factor.squeeze(),
            num_samples=num_samples
        )


class DeepARShared(DeepAR):
    """
    DeepAR model with shared encoder and decoder.
    """
    def __init__(
        self,
        input_size,
        num_cats,
        embedding_dim,
        hidden_size,
        num_layers,
        num_decode_steps,
        likelihood
    ):
        lstm = nn.LSTM(
            input_size=input_size + embedding_dim + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim
        )
        super().__init__(
            input_size_encoder=input_size,
            input_size_decoder=input_size,
            num_cats_encoder=num_cats,
            num_cats_decoder=num_cats,
            embedding_dim_encoder=embedding_dim,
            embedding_dim_decoder=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_decode_steps=num_decode_steps,
            likelihood=likelihood,
            _lstm=lstm,
            _embedding=embedding
        )


def quantile_loss(y, y_hat, rho):
    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()
    y_hat = np.percentile(y_hat, q=rho, axis=0)
    rho /= 100.0
    error = y_hat - y
    over = error > 0
    under = ~over
    loss = error
    loss[over] *= rho
    loss[under] *= (rho - 1)
    loss = 2*loss.sum(axis=0)/y.sum(axis=0)
    return loss


def run():
    N = 8
    T = 365
    D = 4
    H = 128
    num_cats = 100
    embedding_dim = 4
    num_decode_steps = 7
    num_layers = 2

    xe = torch.randn(N, T, D)
    cate = torch.randint(low=0, high=num_cats-1, size=(N, T))
    ye = torch.randn(N, T)

    enc = Encoder(input_size=D, hidden_size=H, num_layers=num_layers,
                  num_cats=num_cats, embedding_dim=embedding_dim)
    h0, c0 = enc(xe, cate, ye)

    xd = torch.randn(N, num_decode_steps, D)
    catd = torch.randint(low=0, high=num_cats-1, size=(N, num_decode_steps))
    yd = torch.randn(N, num_decode_steps)
    y = torch.randn(N)
    y0 = torch.randn(N)
    dec = Decoder(input_size=D, hidden_size=H, num_layers=num_layers,
                  num_decode_steps=num_decode_steps, num_cats=num_cats,
                  embedding_dim=embedding_dim)
    pred = dec.predict(xd, catd, y0, h0, c0, num_samples=10)
    return locals()
