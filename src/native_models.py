"""Activity-only baselines and sequence VAE. See SOURCES.md for adaptations."""
import ast
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parents[1]


def source_classes(path, names):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    scope = {'torch': torch, 'nn': nn, 'F': F, 'math': math}
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


def per_case_ce(logits, target):
    loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=0, reduction='none')
    mask = target != 0
    return (loss * mask).sum(1) / mask.sum(1).clamp(min=1), loss


def probability_mass_score(logits, target):
    p = logits.softmax(-1)
    observed = p.gather(-1, target.unsqueeze(-1))
    score = (p * (p > observed)).sum(-1)
    return score.masked_fill(target == 0, 0).max(1).values


class VAE(nn.Module):
    def __init__(self, vocab, width, hidden=64, latent=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 32, padding_idx=0)
        self.encoder = nn.GRU(32, hidden, batch_first=True)
        self.mu, self.logvar = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.position = nn.Embedding(width, 16)
        self.init = nn.Linear(latent, hidden)
        self.decoder = nn.GRU(latent + 16, hidden, batch_first=True)
        self.output = nn.Linear(hidden, vocab)

    def forward(self, x):
        length = (x != 0).sum(1).cpu()
        packed = pack_padded_sequence(self.embedding(x), length, batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        mu, logvar = self.mu(h[-1]), self.logvar(h[-1]).clamp(-12, 12)
        z = mu + torch.randn_like(mu) * (logvar * .5).exp() if self.training else mu
        positions = self.position(torch.arange(x.shape[1], device=x.device))[None].expand(len(x), -1, -1)
        inp = torch.cat((z[:, None].expand(-1, x.shape[1], -1), positions), -1)
        decoded, _ = self.decoder(inp, self.init(z).unsqueeze(0))
        return self.output(decoded), mu, logvar

    def objective(self, x, epoch):
        logits, mu, logvar = self(x)
        reconstruction, _ = per_case_ce(logits[:, 1:], x[:, 1:])
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)
        return (reconstruction + .01 * kl).mean()

    def score(self, x):
        logits, mu, _ = self(x)
        mean, loss = per_case_ce(logits[:, 1:], x[:, 1:])
        maximum = loss.masked_fill(x[:, 1:] == 0, 0).max(1).values
        return mean, torch.cat((mu, mean[:, None], maximum[:, None]), 1)


class DAE(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        self.vocab, self.width = vocab, width
        size = (vocab - 1) * width
        hidden = max(64, int(.01 * size))
        self.net = nn.Sequential(nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(.5),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(.5),
                                 nn.Linear(hidden, size), nn.Sigmoid())

    def event_errors(self, x):
        # BPAD has no PAD output category: PAD inputs are all-zero vectors.
        y = F.one_hot(x, self.vocab).float()[..., 1:]
        reconstruction = self.net(y.flatten(1)).reshape_as(y)
        return (reconstruction - y).square().mean(2)

    def errors(self, x):
        error = self.event_errors(x)
        mask = x != 0
        return (error * mask).sum(1) / mask.sum(1).clamp(min=1)

    def objective(self, x, epoch):
        # Reference fit uses full-vector MSE; detect masks PAD separately.
        # BPAD default noise=None. Dropout is retained in the network.
        return self.event_errors(x).mean()

    def score(self, x):
        return self.errors(x), None


class BINet(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        hidden = min(width * 2, 64)
        emb = min(16, max(2, vocab // 10))
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        self.encoder = nn.GRU(emb, hidden, batch_first=True)
        self.norm1 = nn.BatchNorm1d(hidden, eps=.001, momentum=.01)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.norm2 = nn.BatchNorm1d(hidden, eps=.001, momentum=.01)
        self.output = nn.Linear(hidden, vocab - 1)

    def recurrent(self, gru, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = gru(packed)
        return pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])[0]

    def normalize_valid(self, norm, h, mask):
        # Explicit padding-invariant adaptation, not claimed Keras equivalence.
        valid = h[mask]
        if self.training and len(valid) == 1:
            normalized = F.batch_norm(valid, norm.running_mean, norm.running_var,
                                      norm.weight, norm.bias, False, 0., norm.eps)
        else:
            normalized = norm(valid)
        result = torch.zeros_like(h)
        result[mask] = normalized
        return result

    def forward(self, x):
        lengths = (x != 0).sum(1) - 1
        mask = torch.arange(x.shape[1] - 1, device=x.device)[None] < lengths[:, None]
        h = self.recurrent(self.encoder, self.embedding(x[:, :-1]), lengths)
        h = self.normalize_valid(self.norm1, h, mask)
        h = self.recurrent(self.decoder, h, lengths)
        h = self.normalize_valid(self.norm2, h, mask)
        # Keep external token indexing while assigning PAD zero probability.
        return F.pad(self.output(h), (1, 0), value=-1e9)

    def objective(self, x, epoch):
        return per_case_ce(self(x), x[:, 1:])[0].mean()

    def score(self, x):
        return probability_mass_score(self(x), x[:, 1:]), None


class GRASPED(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        path = ROOT / 'vendor/guanwei49__GRASPED/model/GRU_AE.py'
        classes = source_classes(path, {'Encoder', 'Attention', 'Decoder', 'GRU_AE'})
        self.model = classes['GRU_AE']([vocab - 1], 64, 1, 1, 64)

    def forward(self, x):
        # Official decoder squeezes batch=1. Duplicate only that inference batch.
        single = len(x) == 1
        if single:
            x = x.repeat(2, 1)
        result = self.model([x], x == 0)[0][:, 1:]
        return result[:1] if single else result

    def objective(self, x, epoch):
        return per_case_ce(self(x), x[:, 1:])[0].mean()

    def score(self, x):
        return probability_mass_score(self(x), x[:, 1:]), None


class TranAD(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        self.vocab = vocab
        classes = source_classes(ROOT / 'vendor/imperial-qore__TranAD/src/dlutils.py',
                                 {'PositionalEncoding', 'TransformerEncoderLayer', 'TransformerDecoderLayer'})
        self.position = classes['PositionalEncoding'](2 * vocab, .1, 10)
        self.encoder = classes['TransformerEncoderLayer'](2 * vocab, vocab, 16, .1)
        self.decoder1 = classes['TransformerDecoderLayer'](2 * vocab, vocab, 16, .1)
        self.decoder2 = classes['TransformerDecoderLayer'](2 * vocab, vocab, 16, .1)
        self.output = nn.Sequential(nn.Linear(2 * vocab, vocab), nn.Sigmoid())

    def windows(self, x):
        # Windows never cross case boundaries; target is the current last event.
        owners, windows = [], []
        for i, row in enumerate(x):
            length = int((row != 0).sum())
            for t in range(1, length):
                start = max(0, t - 9)
                window = row[start:t + 1]
                if len(window) < 10:
                    window = torch.cat((row[:1].expand(10 - len(window)), window))
                windows.append(window)
                owners.append(i)
        return torch.stack(windows), torch.tensor(owners, device=x.device)

    def forward(self, window):
        source = F.one_hot(window, self.vocab).float().transpose(0, 1)
        target = source[-1:]
        query = target.repeat(1, 1, 2)
        def memory(c):
            return self.encoder(self.position(torch.cat((source, c), 2) * math.sqrt(self.vocab)))
        first = self.output(self.decoder1(query, memory(torch.zeros_like(source))))
        second = self.output(self.decoder2(query, memory((first - source).square())))
        return first, second, target

    def objective(self, x, epoch):
        # Runner uses event-window batches for this model.
        first, second, target = self(x)
        weight = 1. / (epoch + 1)
        return (weight * (first - target).square() + (1 - weight) * (second - target).square()).mean()

    @torch.no_grad()
    def score(self, x):
        windows, owners = self.windows(x)
        scores = torch.zeros(len(x), device=x.device)
        for i in range(0, len(windows), 64):
            _, second, target = self(windows[i:i + 64])
            loss = (second - target).square().mean(2).squeeze(0)
            scores.scatter_reduce_(0, owners[i:i + 64], loss, reduce='amax', include_self=True)
        return scores, None


FACTORIES = {'VAE': VAE, 'DAE': DAE, 'BINet': BINet, 'GRASPED': GRASPED, 'TranAD': TranAD}
