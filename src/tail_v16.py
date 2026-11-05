"""Full categorical tail evidence from the same frozen sequence VAE.

The clean and target-hidden forward passes also produce every legacy
``masked_vae.extract_views`` field with its original arithmetic.  The only new
fields are ``clean_tail_event``, ``clean_tail_stats``, ``masked_tail_event`` and
``masked_tail_stats``.  Event arrays exclude BOS, include EOS and set PAD to
zero.  Each stats array contains the raw [mean, top-three mean, maximum]
negative-log tail mass; no log1p transform is applied here.

This is contextual reconstruction evidence, not a calibrated anomaly p-value.
No UNK adjustment, new model parameters, labels or fitted quantities are used.
"""
import numpy as np
import torch


def categorical_tail(logp, target):
    """Return -log mass of categories no more likely than the observed token.

    Every category tied with the observation is included.  Float64 log-sum-exp
    accumulates that probability mass without underflow even for extreme,
    finite logits.  ``logp`` is a normalized categorical log-probability tensor
    of shape (..., vocabulary); ``target`` has the matching leading shape.
    PAD is token zero.  BOS exclusion is the extractor's responsibility.
    """
    assert logp.shape[:-1] == target.shape
    assert target.dtype == torch.long
    lp = logp.double()
    observed = lp.gather(-1, target[..., None])
    included = lp <= observed
    log_mass = torch.logsumexp(lp.masked_fill(~included, -torch.inf), dim=-1)
    # A float32 log_softmax can sum to slightly above one after promotion.
    # Clip only that numerical excess; do not clip tiny genuine tail masses.
    return (-log_mass.clamp(max=0.)).masked_fill(target == 0, 0.)


def _legacy_statistics(errors, entropy, valid):
    """Keep the original nine-column reconstruction statistics exactly."""
    rows = []
    for a, b, keep in zip(errors, entropy, valid):
        a = a[keep].astype(np.float64)
        b = b[keep].astype(np.float64)
        top = np.sort(a)[::-1]
        rows.append([a.mean(), a.std(), top[:3].mean(), top[:5].mean(), a.max(),
                     np.quantile(a, .9), b.mean(), np.maximum(a-b, 0.).mean(),
                     np.maximum(a-b, 0.).max()])
    return np.asarray(rows)


def _tail_statistics(values, valid):
    rows = []
    for values_row, keep in zip(values, valid):
        a = values_row[keep].astype(np.float64)
        assert len(a) > 0, 'Every reconstruction target must contain EOS'
        rows.append([a.mean(), np.sort(a)[-min(3, len(a)):].mean(), a.max()])
    return np.asarray(rows, dtype=np.float64)


def _view_pack(logp, target):
    """Derive original arithmetic and new tail evidence from one logp tensor."""
    valid = target != 0
    error = -logp.gather(-1, target[..., None]).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    error = error.masked_fill(~valid, 0)
    entropy = entropy.masked_fill(~valid, 0)
    e, h, m = error.cpu().numpy(), entropy.cpu().numpy(), valid.cpu().numpy()
    tail = categorical_tail(logp, target).cpu().numpy()
    result = dict(event=e, entropy=h, stats=_legacy_statistics(e, h, m),
                  native=(error.sum(1)/valid.sum(1).clamp(min=1)).cpu().numpy())
    return result, tail, _tail_statistics(tail, m), m


@torch.no_grad()
def extract_tail_views(model, tokens, device):
    """Extract unchanged legacy fields plus both categorical tail views.

    Exactly one clean forward and the original four phase-hidden forwards run
    per batch.  There is no additional encoder/decoder or feature-dependent
    refitting.  The supplied model must already be in evaluation mode.
    """
    from masked_vae import masked_logits

    assert not model.training, 'Tail extraction must disable stochastic training'
    tokens = np.asarray(tokens, dtype=np.int64)
    assert tokens.ndim == 2 and len(tokens) > 0 and tokens.shape[1] >= 2
    assert np.all(tokens[:, 0] == 1), 'Expected initial BOS token'
    keys = ('latent', 'event', 'entropy', 'length', 'stats', 'native',
            'masked_event', 'masked_entropy', 'masked_stats', 'masked_native',
            'clean_tail_event', 'clean_tail_stats',
            'masked_tail_event', 'masked_tail_stats')
    chunks = {key: [] for key in keys}
    for start in range(0, len(tokens), 128):
        x = torch.as_tensor(tokens[start:start+128], device=device)
        target = x[:, 1:]
        logits, mu, _ = model(x)
        clean_lp = logits[:, 1:].log_softmax(-1)
        clean, clean_tail, clean_stats, valid = _view_pack(clean_lp, target)
        hidden_lp = masked_logits(model, x)[:, 1:].log_softmax(-1)
        hidden, hidden_tail, hidden_stats, hidden_valid = _view_pack(hidden_lp, target)
        assert np.array_equal(valid, hidden_valid)
        chunks['latent'].append(mu.cpu().numpy())
        chunks['length'].append(valid.sum(1))
        for key, value in clean.items():
            chunks[key].append(value)
        for key, value in hidden.items():
            chunks['masked_' + key].append(value)
        chunks['clean_tail_event'].append(clean_tail)
        chunks['clean_tail_stats'].append(clean_stats)
        chunks['masked_tail_event'].append(hidden_tail)
        chunks['masked_tail_stats'].append(hidden_stats)
    result = {key: np.concatenate(value) for key, value in chunks.items()}
    result['tokens'] = tokens[:, 1:].copy()
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def selfcheck(device='cpu'):
    """Generated tensors only: ties, extremes, masking and order invariance."""
    uniform = torch.zeros(2, 3, 11, device=device).log_softmax(-1)
    target = torch.tensor([[4, 2, 0], [7, 3, 2]], device=device)
    assert torch.allclose(categorical_tail(uniform, target),
                          torch.zeros_like(target, dtype=torch.float64), atol=1e-6)
    tied = torch.tensor([[[.4, .2, .2, .1, .1]]], dtype=torch.float64, device=device).log()
    assert np.isclose(float(categorical_tail(tied, torch.tensor([[1]], device=device))), -np.log(.6))
    assert np.isclose(float(categorical_tail(tied, torch.tensor([[3]], device=device))), -np.log(.2))
    extreme = torch.tensor([[[0., -1000., -2000., -3000.]]], device=device).log_softmax(-1)
    assert np.isclose(float(categorical_tail(extreme, torch.tensor([[3]], device=device))), 3000.)
    generator = torch.Generator(device=device).manual_seed(16016)
    logits = torch.randn(5, 7, 13, generator=generator, device=device)
    target = torch.randint(0, 13, (5, 7), generator=generator, device=device)
    lp = logits.log_softmax(-1)
    values = categorical_tail(lp, target)
    assert bool(torch.all(values[target == 0] == 0))
    # PAD keeps its reserved identity; all actual output categories permute.
    vocabulary_order = torch.cat((torch.zeros(1, dtype=torch.long, device=device),
                                  1 + torch.randperm(12, generator=generator, device=device)))
    inverse = torch.argsort(vocabulary_order)
    permuted = categorical_tail(lp[..., vocabulary_order], inverse[target])
    assert torch.allclose(values, permuted, atol=1e-12, rtol=1e-12)
    row_order = torch.tensor([3, 0, 4, 1, 2], device=device)
    assert torch.equal(values[row_order], categorical_tail(lp[row_order], target[row_order]))
    assert torch.equal(values, torch.cat([categorical_tail(lp[:2], target[:2]),
                                         categorical_tail(lp[2:], target[2:])]))
    return dict(ok=True, generated_tensors_only=True, conservative_ties=True,
                float64_log_space_accumulation=True, finite_extreme_tail=True,
                pad_zero=True, permutation_equivariance=True, batch_equivalence=True)


def extraction_selfcheck(device='cpu'):
    """Generated VAE/sequences: legacy values, weights and batching unchanged."""
    from native_models import VAE
    from context_vae import ContextVAE
    from masked_vae import MaskedContextVAE, extract_views
    from common import fingerprint, seed_all

    seed_all(16016)
    model = MaskedContextVAE(ContextVAE(VAE(12, 52), 12)).to(device).eval().requires_grad_(False)
    tokens = np.array([[1, 4, 5, 6, 2, 0, 0, 0], [1, 6, 5, 2, 0, 0, 0, 0],
                       [1, 4, 7, 8, 9, 6, 2, 0], [1, 2, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    before = fingerprint(model)
    legacy = extract_views(model, tokens, device)
    extended = extract_tail_views(model, tokens, device)
    for key, expected in legacy.items():
        assert np.array_equal(extended[key], expected), ('Legacy extraction changed', key)
    single = extract_tail_views(model, tokens[:1], device)
    for key, values in extended.items():
        assert np.allclose(values[:1], single[key], atol=1e-6, rtol=1e-5), key
    for prefix in ('clean', 'masked'):
        event = extended[prefix + '_tail_event']
        assert event.shape == legacy['event'].shape
        assert np.all(event[extended['tokens'] == 0] == 0.)
        assert extended[prefix + '_tail_stats'].shape == (len(tokens), 3)
    assert fingerprint(model) == before
    return dict(ok=True, generated_model_and_sequences_only=True,
                legacy_keys_bitwise_unchanged=True, model_parameters_unchanged=True,
                eos_included_bos_excluded=True, raw_three_column_statistics=True,
                batch_invariance=True)


if __name__ == '__main__':
    import json
    print(json.dumps(dict(tensors=selfcheck(), extraction=extraction_selfcheck()), indent=2))
