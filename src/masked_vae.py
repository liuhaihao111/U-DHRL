"""One sequence VAE, with clean and target-hidden reconstruction views.

The frozen historical ContextVAE is a training-time teacher only.  Its copied
encoder, decoder and causal residual are jointly trained here.  A separate MASK
input embedding never becomes an output class.  Four position phases hide each
non-padding target, including EOS, exactly once.  Fitting accepts only original
training and validation tokens, not anomaly labels or calibration/test data.
"""
import copy
import io
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import CFG, fingerprint, seed_all, write
from context_vae import ContextVAE, cached_logits
from native_models import VAE, per_case_ce
from representation import extract as clean_extract, load_vae


PHASES = 4


def _expanded_embedding(embedding):
    old = embedding.weight.detach()
    result = nn.Embedding(len(old) + 1, old.shape[1], padding_idx=0,
                          device=old.device, dtype=old.dtype)
    with torch.no_grad():
        result.weight[:-1].copy_(old)
        # Fixed, deterministic initialization.  MASK is not PAD or UNK.
        result.weight[-1].copy_(old[4:].mean(0) if len(old) > 4 else old[1:].mean(0))
    return result


class MaskedContextVAE(nn.Module):
    """The same weights supply both views; no deployed teacher/second encoder."""
    def __init__(self, context_model):
        super().__init__()
        self.vocab = int(context_model.base.output.out_features)
        self.mask_id = self.vocab
        self.base = copy.deepcopy(context_model.base).requires_grad_(True)
        self.base.embedding = _expanded_embedding(context_model.base.embedding)
        self.embedding = _expanded_embedding(context_model.embedding)
        self.dropout = copy.deepcopy(context_model.dropout)
        self.context = copy.deepcopy(context_model.context).requires_grad_(True)
        self.output = copy.deepcopy(context_model.output).requires_grad_(True)
        self.logit_bound = float(CFG.get('context_logit_bound', 8.))
        self.register_buffer('mix', context_model.mix.detach().clone())
        self.register_buffer('masked_enabled', torch.tensor(False, device=self.mix.device))

    def correction(self, x):
        h, _ = self.context(self.dropout(self.embedding(x[:, :-1])))
        delta = self.logit_bound * torch.tanh(self.output(h) / self.logit_bound)
        return torch.cat((torch.zeros_like(delta[:, :1]), delta), 1)

    def forward(self, x):
        # Deliberately no no_grad wrapper and no forced base.eval(): the encoder
        # and decoder really train, unlike the historical frozen residual fit.
        logits, mu, logvar = self.base(x)
        if float(self.mix) != 0.:
            logits = logits + self.mix * self.correction(x)
        return logits, mu, logvar


def phase_mask(x, phase):
    """Content-independent target positions; BOS/PAD are never hidden."""
    positions = torch.arange(x.shape[1], device=x.device)[None]
    phase = torch.as_tensor(phase, device=x.device, dtype=torch.long)
    if phase.ndim:
        assert phase.shape == (len(x),)
        phase = phase[:, None]
    valid = (positions > 0) & (x != 0) & (x != 1)
    return valid & ((positions - 1) % PHASES == phase)


def hide_targets(x, phase, mask_id):
    mask = phase_mask(x, phase)
    hidden = x.clone()
    hidden[mask] = int(mask_id)
    return hidden, mask


def _masked_ce(logits, target, mask):
    ce = F.cross_entropy(logits.transpose(1, 2), target, reduction='none')
    count = mask.sum(1)
    per_case = (ce * mask).sum(1) / count.clamp(min=1)
    return per_case[count > 0].mean()


def _anchor_kl(logits, old_logits, target):
    logp = logits.log_softmax(-1)
    old_logp = old_logits.log_softmax(-1)
    divergence = (old_logp.exp() * (old_logp - logp)).sum(-1)
    valid = target != 0
    return ((divergence * valid).sum(1) / valid.sum(1).clamp(min=1)).mean()


@torch.no_grad()
def masked_logits(model, x):
    """Return logits assembled only from the phase hiding each target."""
    assert not model.training, 'Target-hidden inference must disable dropout and latent sampling'
    output = None
    coverage = torch.zeros_like(x, dtype=torch.long)
    for phase in range(PHASES):
        hidden, selected = hide_targets(x, phase, model.mask_id)
        logits = model(hidden)[0]
        if output is None:
            output = torch.zeros_like(logits)
        output[selected] = logits[selected]
        coverage += selected.long()
    expected = (x != 0) & (x != 1)
    expected[:, 0] = False
    assert torch.equal(coverage, expected.long()), 'Every non-PAD target must be hidden exactly once'
    return output


@torch.no_grad()
def _validation(model, tokens, device):
    model.eval()
    clean_sum, masked_sum = 0., 0.
    for st in range(0, len(tokens), 128):
        x = torch.as_tensor(tokens[st:st + 128], device=device, dtype=torch.long)
        clean = model(x)[0]
        hidden = masked_logits(model, x)
        clean_sum += float(per_case_ce(clean[:, 1:], x[:, 1:])[0].sum())
        masked_sum += float(per_case_ce(hidden[:, 1:], x[:, 1:])[0].sum())
    return dict(clean_nll=clean_sum / len(tokens), masked_nll=masked_sum / len(tokens))


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_masked(context_model, train, val, seed, device, out, epochs=None):
    """Fit without calibration, test tokens, synthetic anomalies or labels.

    Selection uses four-phase original-validation reconstruction NLL, subject
    to a separate clean-view retention guard.  Neither criterion establishes
    anomaly precision.  A rejected fit restores the complete initial state.
    """
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train = np.asarray(train, dtype=np.int64)
    val = np.asarray(val, dtype=np.int64)
    assert train.ndim == val.ndim == 2 and len(train) > 0 and len(val) > 0
    assert train.shape[1] == val.shape[1]
    vocab = int(context_model.base.output.out_features)
    for tokens in (train, val):
        assert np.all((tokens >= 0) & (tokens < vocab)), 'Original input contains invalid or MASK token'
        assert np.all(tokens[:, 0] == 1) and not np.any(tokens[:, 1:] == 1), 'Expected one initial BOS'
        counts = (tokens != 0).sum(1)
        assert np.all(counts >= 2) and np.all((tokens == 2).sum(1) == 1), 'Expected exactly one EOS'
        assert np.all(tokens[np.arange(len(tokens)), counts - 1] == 2), 'EOS must end each unpadded trace'
        assert np.array_equal(tokens != 0, np.arange(tokens.shape[1])[None] < counts[:, None]), 'PAD must remain trailing'
    assert int(CFG.get('masked_phases', PHASES)) == PHASES
    weights = dict(masked_ce=1., clean_ce=float(CFG.get('masked_clean_weight', .5)),
                   latent_kl=float(CFG.get('masked_kl_weight', .01)),
                   historical_clean_kl=float(CFG.get('masked_anchor_weight', .1)))
    assert weights == dict(masked_ce=1., clean_ce=.5, latent_kl=.01, historical_clean_kl=.1), 'Unexpected frozen masked objective weights'
    teacher_hash = fingerprint(context_model)
    # Do not even change train/eval or requires_grad flags of the caller's
    # historical model.  This temporary teacher is discarded after fitting.
    teacher = copy.deepcopy(context_model).to(device).eval().requires_grad_(False)
    old_train = cached_logits(teacher, train, device)
    old_val = cached_logits(teacher, val, device)
    original_clean = float(per_case_ce(old_val[:, 1:], torch.as_tensor(val[:, 1:]))[0].mean())
    seed_all(seed + 1414)
    initial_model = MaskedContextVAE(teacher).to(device).eval()
    initial = _state(initial_model)
    initial_values = _validation(initial_model, val, device)
    assert np.isclose(initial_values['clean_nll'], original_clean, rtol=1e-5, atol=1e-7), 'Warm start changed clean outputs'
    initial_encoder = initial_model.base.encoder.weight_ih_l0.detach().cpu().clone()
    clean_tolerance = float(CFG.get('masked_clean_tolerance', .01))
    min_gain = float(CFG.get('masked_min_relative_gain', .01))
    clean_limit = original_clean * (1. + clean_tolerance)
    rates = list(CFG.get('masked_learning_rates', [1e-4, 3e-5]))
    max_epochs = int(epochs if epochs is not None else CFG.get('masked_epochs', 60))
    patience = int(CFG.get('masked_patience', 6))
    assert max_epochs > 0 and patience > 0
    best = initial_values['masked_nll']
    selected = dict(state=initial, lr=None, epoch=0, **initial_values)
    history, runs = [], []
    total_updates = 0
    total_encoder_gradient_updates = 0
    for lr in rates:
        seed_all(seed + 1414)
        rng = np.random.default_rng(seed + 1414)
        model = MaskedContextVAE(teacher).to(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
        stale = 0
        local_best = initial_values['masked_nll']
        updates = 0
        gradient_updates = 0
        maximum_encoder_gradient = 0.
        for epoch in range(max_epochs):
            model.train()
            order = rng.permutation(len(train))
            objective_sum, cases = 0., 0
            for st in range(0, len(order), int(CFG.get('batch_size', 128))):
                ix = order[st:st + int(CFG.get('batch_size', 128))]
                x = torch.as_tensor(train[ix], device=device, dtype=torch.long)
                # A short trace may have fewer than four nonempty phases.
                valid_count = np.maximum((train[ix] != 0).sum(1) - 1, 1)
                phases = np.array([rng.integers(0, min(PHASES, int(n))) for n in valid_count])
                hidden, selected_mask = hide_targets(x, phases, model.mask_id)
                masked_prediction, mu, logvar = model(hidden)
                clean_prediction, _, _ = model(x)
                reconstruction = _masked_ce(masked_prediction, x, selected_mask)
                clean_loss = per_case_ce(clean_prediction[:, 1:], x[:, 1:])[0].mean()
                kl = -.5 * (1. + logvar - mu.square() - logvar.exp()).sum(1).mean()
                anchor = _anchor_kl(clean_prediction[:, 1:], old_train[ix, 1:].to(device), x[:, 1:])
                loss = reconstruction + weights['clean_ce'] * clean_loss + weights['latent_kl'] * kl + weights['historical_clean_kl'] * anchor
                assert torch.isfinite(loss), 'Nonfinite masked VAE objective'
                optimizer.zero_grad()
                loss.backward()
                encoder_grad = sum(float(p.grad.detach().square().sum()) for p in model.base.encoder.parameters() if p.grad is not None) ** .5
                assert np.isfinite(encoder_grad), 'Nonfinite encoder gradient'
                if encoder_grad > 0.:
                    gradient_updates += 1
                maximum_encoder_gradient = max(maximum_encoder_gradient, encoder_grad)
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                updates += 1
                objective_sum += float(loss.detach()) * len(ix)
                cases += len(ix)
            if (epoch + 1) % 5 and epoch + 1 != max_epochs:
                continue
            values = _validation(model, val, device)
            guard = values['clean_nll'] <= clean_limit + 1e-8
            gain = (initial_values['masked_nll'] - values['masked_nll']) / max(initial_values['masked_nll'], 1e-12)
            row = dict(lr=float(lr), epoch=epoch + 1, training_objective=objective_sum / max(cases, 1),
                       clean_guard_passed=bool(guard), masked_relative_gain=float(gain), **values)
            history.append(row)
            write(out / 'masked_history.json', history)
            if guard and values['masked_nll'] < best:
                best = values['masked_nll']
                selected = dict(state=_state(model), lr=float(lr), epoch=epoch + 1, **values)
            if guard and values['masked_nll'] < local_best - 1e-5:
                local_best = values['masked_nll']
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        changed = not torch.equal(model.base.encoder.weight_ih_l0.detach().cpu(), initial_encoder)
        assert updates > 0 and gradient_updates > 0 and changed, 'Masked encoder did not really train'
        runs.append(dict(lr=float(lr), epochs=epoch + 1, updates=updates,
                         encoder_gradient_updates=gradient_updates,
                         maximum_encoder_gradient_norm=maximum_encoder_gradient,
                         encoder_parameters_changed=changed))
        total_updates += updates
        total_encoder_gradient_updates += gradient_updates
        del model, optimizer
    gain = (initial_values['masked_nll'] - best) / max(initial_values['masked_nll'], 1e-12)
    accepted = bool(selected['epoch'] > 0 and gain >= min_gain)
    if not accepted:
        selected = dict(state=initial, lr=None, epoch=0, **initial_values)
    deployed = MaskedContextVAE(teacher).to(device)
    deployed.load_state_dict(selected['state'])
    deployed.masked_enabled.fill_(accepted)
    deployed.eval().requires_grad_(False)
    if not accepted:
        assert all(torch.equal(v.detach().cpu(), initial[k]) for k, v in deployed.state_dict().items()), 'Rejected fit failed exact state rollback'
    assert fingerprint(context_model) == teacher_hash, 'Caller historical model changed'
    assert fingerprint(teacher) == teacher_hash, 'Training-time historical teacher changed'
    info = dict(masked_enabled=accepted, accepted=accepted, phases=PHASES,
                selection='original validation four-phase masked NLL with clean reconstruction retention guard',
                selected_lr=selected['lr'], selected_epoch=selected['epoch'],
                original_clean_validation_nll=original_clean,
                initial_clean_validation_nll=initial_values['clean_nll'],
                initial_masked_validation_nll=initial_values['masked_nll'],
                best_guarded_masked_validation_nll=best,
                deployed_clean_validation_nll=selected['clean_nll'],
                deployed_masked_validation_nll=selected['masked_nll'],
                relative_masked_nll_gain=float(gain), required_relative_gain=min_gain,
                deployed_relative_masked_nll_gain=float(gain) if accepted else 0.,
                maximum_clean_relative_degradation=clean_tolerance,
                clean_validation_nll_limit=clean_limit,
                objective_weights=weights,
                input_mask_id=deployed.mask_id, output_vocabulary=deployed.vocab,
                mask_in_output_vocabulary=False, eos_is_masked=True, bos_pad_never_masked=True,
                train_cases=len(train), validation_cases=len(val),
                attempted_updates=total_updates,
                encoder_gradient_updates=total_encoder_gradient_updates,
                runs=runs, teacher_unchanged=True, teacher_deployed=False,
                deployed_vae_parameter_sets=1,
                exact_initial_state_restored=not accepted,
                uses_anomaly_labels=False, uses_calibration=False, uses_test=False,
                training_seconds=time.perf_counter() - started,
                interpretation='Reconstruction validation is not a guarantee of anomaly AP, precision or F1 improvement')
    torch.save(dict(state=_state(deployed), **info), out / 'masked_vae.pt')
    write(out / 'masked_selection.json', info)
    return deployed, info


def load_masked(ds, seed, vocab, device, path):
    base = load_vae(ds, seed, vocab, device)
    template = ContextVAE(base, vocab).to(device)
    model = MaskedContextVAE(template).to(device)
    saved = torch.load(path, map_location='cpu')
    assert saved['output_vocabulary'] == vocab
    model.load_state_dict(saved['state'])
    assert bool(model.masked_enabled) == bool(saved['masked_enabled'])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def extract_views(model, tokens, device):
    assert not model.training
    tokens = np.asarray(tokens, dtype=np.int64)
    result = clean_extract(model, tokens, device)
    chunks = {k: [] for k in ('event', 'entropy', 'stats', 'native')}
    for st in range(0, len(tokens), 128):
        x = torch.as_tensor(tokens[st:st + 128], device=device, dtype=torch.long)
        logits = masked_logits(model, x)
        logp = logits[:, 1:].log_softmax(-1)
        target = x[:, 1:]
        valid = target != 0
        error = -logp.gather(-1, target[..., None]).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        error = error.masked_fill(~valid, 0.)
        entropy = entropy.masked_fill(~valid, 0.)
        e, h, m = error.cpu().numpy(), entropy.cpu().numpy(), valid.cpu().numpy()
        rows = []
        for errors, uncertainty, keep in zip(e, h, m):
            a, b = errors[keep].astype(np.float64), uncertainty[keep].astype(np.float64)
            top = np.sort(a)[::-1]
            rows.append([a.mean(), a.std(), top[:3].mean(), top[:5].mean(), a.max(),
                         np.quantile(a, .9), b.mean(), np.maximum(a - b, 0.).mean(),
                         np.maximum(a - b, 0.).max()])
        chunks['event'].append(e)
        chunks['entropy'].append(h)
        chunks['stats'].append(np.asarray(rows))
        chunks['native'].append((error.sum(1) / valid.sum(1).clamp(min=1)).cpu().numpy())
    result.update({'masked_' + k: np.concatenate(v) for k, v in chunks.items()})
    result['tokens'] = tokens[:, 1:].copy()
    return result


def selfcheck():
    """Software checks from generated sequences, never benchmark labels."""
    seed_all(14014)
    x = torch.tensor([[1, 4, 5, 6, 7, 8, 2, 0],
                      [1, 7, 4, 2, 0, 0, 0, 0],
                      [1, 5, 8, 9, 6, 4, 5, 2],
                      [1, 2, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    teacher = ContextVAE(VAE(12, 52), 12).eval().requires_grad_(False)
    teacher.mix.fill_(.5)
    with torch.no_grad():
        teacher.output.weight.normal_(0., .05)
    model = MaskedContextVAE(teacher).eval()
    assert model.base.embedding.num_embeddings == 13 and model.base.output.out_features == 12
    with torch.no_grad():
        assert torch.equal(model(x)[0], teacher(x)[0]), 'Clean warm start must be exact'
    coverage = torch.zeros_like(x)
    for phase in range(PHASES):
        hidden, selected = hide_targets(x, phase, model.mask_id)
        coverage += selected.long()
        assert torch.equal(hidden != 0, x != 0), 'MASK changed padding or sequence lengths'
        assert not selected[:, 0].any()
        assert torch.all(hidden[selected] == model.mask_id)
    assert torch.equal(coverage, ((x != 0) & (x != 1)).long())
    # Changing the target identity cannot alter prediction when that target is
    # hidden: both model inputs are exactly identical, not merely close.
    j = 3
    altered = x.clone()
    altered[0, j] = 10
    one, _ = hide_targets(x[:1], (j - 1) % PHASES, model.mask_id)
    two, _ = hide_targets(altered[:1], (j - 1) % PHASES, model.mask_id)
    assert torch.equal(one, two)
    with torch.no_grad():
        assert torch.equal(model(one)[0][:, j], model(two)[0][:, j])
    before = model.base.encoder.weight_ih_l0.detach().clone()
    model.train()
    assert model.base.training and model.base.encoder.training
    hidden, selected = hide_targets(x, torch.tensor([0, 1, 2, 0]), model.mask_id)
    logits, mu, logvar = model(hidden)
    clean_logits = model(x)[0]
    with torch.no_grad():
        old_logits = teacher(x)[0]
    objective = _masked_ce(logits, x, selected) + .5 * per_case_ce(clean_logits[:, 1:], x[:, 1:])[0].mean()
    objective += .01 * (-.5 * (1. + logvar - mu.square() - logvar.exp()).sum(1).mean())
    objective += .1 * _anchor_kl(clean_logits[:, 1:], old_logits[:, 1:], x[:, 1:])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    objective.backward()
    assert model.base.encoder.weight_ih_l0.grad is not None
    assert float(model.base.encoder.weight_ih_l0.grad.abs().sum()) > 0.
    assert model.base.embedding.weight.grad is not None
    assert float(model.base.embedding.weight.grad[-1].abs().sum()) > 0.
    optimizer.step()
    assert not torch.equal(before, model.base.encoder.weight_ih_l0.detach())
    model.eval()
    model.masked_enabled.fill_(True)
    buffer = io.BytesIO()
    torch.save(_state(model), buffer)
    buffer.seek(0)
    clone = MaskedContextVAE(teacher).eval()
    clone.load_state_dict(torch.load(buffer, map_location='cpu'))
    pack = extract_views(model, x.numpy(), 'cpu')
    copied = extract_views(clone, x.numpy(), 'cpu')
    for key in pack:
        assert np.array_equal(pack[key], copied[key]), key
    single = extract_views(model, x[:1].numpy(), 'cpu')
    for key in pack:
        assert np.allclose(pack[key][:1], single[key], rtol=1e-5, atol=1e-6), key
    assert pack['masked_event'].shape == (len(x), x.shape[1] - 1)
    assert np.all(pack['masked_event'][pack['tokens'] == 0] == 0.)
    return dict(ok=True, generated_tokens_only=True, target_invisibility=True,
                exact_phase_coverage=True, eos_covered=True, padding_lengths_preserved=True,
                clean_warm_start_exact=True, encoder_and_mask_embedding_gradient=True,
                actual_encoder_update=True, one_deployed_vae=True,
                serialization_roundtrip=True, batch_invariance=True)


if __name__ == '__main__':
    import json
    print(json.dumps(selfcheck(), indent=2))
