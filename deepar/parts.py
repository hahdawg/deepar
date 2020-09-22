from collections import deque
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import torch
import torch.nn as nn

from . import core


def load_raw_data():
    res = pd.read_hdf("~/projects/tsdata/parts.h5")
    return res


def normalize_ts(ts):
    mu = ts.mean(axis=1).reshape(-1, 1)
    std = ts.std(axis=1).reshape(-1, 1)
    res = (ts.astype("float") - mu)/std
    return res


def compute_model_data(raw):
    ts = raw.T
    ts = ts[sorted(ts.columns.tolist())]

    low_demand = ts.sum(axis=1) < 10
    low_demand_start = ts.iloc[:, :11].sum(axis=1) == 0
    low_demand_end = ts.iloc[:, -11:].sum(axis=1) == 0
    remove = low_demand | low_demand_start | low_demand_end

    ts = ts.loc[~remove].copy()
    N, T = ts.shape
    trend = np.repeat(np.arange(T).reshape(1, -1), N, axis=0).astype("float")
    trend = normalize_ts(trend)
    month = np.repeat(ts.columns.month.values.reshape(1, -1) - 1, N, axis=0).astype("float")
    month = normalize_ts(month)
    x = np.stack((trend, month), axis=2)

    encoder = sp.LabelEncoder()
    cat = encoder.fit_transform(ts.index.values)
    cat = np.repeat(cat.reshape(-1, 1), T, axis=1)

    return ts.values, x, cat, ts.index.values, ts.columns.values


def generate_batches(
    model_data,
    batch_size,
    istrain,
    encode_len,
    decode_len,
    device,
    num_epochs=10000000
):
    ts, x, cat, part_id, date = model_data
    N, T = ts.shape
    sample_idx = np.arange(N)
    for _ in range(num_epochs):
        if istrain:
            sample_idx = np.random.permutation(sample_idx)
            ts, x, cat, part_id = [d[sample_idx] for d in (ts, x, cat, part_id)]

        for i in range(0, N, batch_size):
            if istrain and i + batch_size > N:
                break

            ts_batch, x_batch, cat_batch, part_id_batch = \
                [d[i:i + batch_size] for d in (ts, x, cat, part_id)]

            if istrain:
                encode_end_idx = np.random.randint(low=encode_len, high=T - decode_len)
                encode_start_idx = encode_end_idx - encode_len
                decode_start_idx = encode_end_idx + 1
                decode_end_idx = decode_start_idx + decode_len

            else:
                decode_end_idx = T
                decode_start_idx = T - decode_len
                encode_end_idx = decode_start_idx - 1
                encode_start_idx = encode_end_idx - encode_len

            y_encode = ts_batch[:, encode_start_idx:encode_end_idx]
            factor = 1 + y_encode.mean(axis=1).reshape(-1, 1)
            x_encode = x_batch[:, encode_start_idx:encode_end_idx, :]
            cat_encode = cat_batch[:, encode_start_idx:encode_end_idx]

            x_decode = x_batch[:, decode_start_idx:decode_end_idx, :]
            cat_decode = cat_batch[:, decode_start_idx:decode_end_idx]

            date_encode = date[encode_start_idx:encode_end_idx]
            date_decode = date[decode_start_idx:decode_end_idx]

            batch = {
                "y_encode": torch.from_numpy(y_encode).float().to(device),
                "x_encode": torch.from_numpy(x_encode).float().to(device),
                "cat_encode": torch.from_numpy(cat_encode).long().to(device),
                "x_decode": torch.from_numpy(x_decode).float().to(device),
                "cat_decode": torch.from_numpy(cat_decode).long().to(device),
                "date_encode": date_encode,
                "date_decode": date_decode,
                "part_id": part_id_batch,
                "factor": torch.from_numpy(factor).to(device)
            }

            if istrain:
                y_decode = ts_batch[:, (decode_start_idx - 1):(decode_end_idx - 1)]
                target_decode = ts_batch[:, decode_start_idx:decode_end_idx]
                batch["y_decode"] = torch.from_numpy(y_decode).float().to(device)
                batch["target_decode"] = torch.from_numpy(target_decode).float().to(device)

            yield batch


def train():
    device = torch.device("cuda")
    model_data = compute_model_data(load_raw_data())
    bg = generate_batches(
        model_data=model_data,
        batch_size=256,
        istrain=True,
        encode_len=8,
        decode_len=8,
        device=device
    )

    model = core.DeepARShared(
        input_size=2,
        num_cats=1046,
        embedding_dim=1,
        hidden_size=40,
        num_layers=3,
        num_decode_steps=8,
        likelihood=core.NegativeBinomialLikelihood
    )
    model.to(device)

    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())

    checkpoint_freq = 2500
    running_loss = deque(maxlen=checkpoint_freq)
    running_mae = deque(maxlen=checkpoint_freq)
    for i, batch in enumerate(bg):
        optimizer.zero_grad()
        batch = next(bg)
        h = model(**batch)
        batch["factor"] = 1.0
        factor = batch["factor"]
        loss = model.decoder.likelihood.loss(h=h, y=batch["target_decode"], factor=factor)
        running_loss.append(loss.mean().item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        with torch.no_grad():
            y_hat_b, _ = model.decoder.likelihood.transform(model(**batch), factor=factor)
            mae = torch.abs(batch["target_decode"] - y_hat_b).mean().item()
            running_mae.append(mae)

        if i % checkpoint_freq == 0:
            print(f"step: {i}  loss: {np.mean(running_loss):.5f}   mae: {mae:.5f}")
