"""Frozen V26 two-stage inference: only fine routes acquire local evidence."""
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from representation import robust_apply
from evidence_v15 import membership_terms
from masked_vae import masked_logits
from tail_v16 import _legacy_statistics, _view_pack, categorical_tail, _tail_statistics
from policy import mask_q, coarse_fine_advantage
from policy_v16 import coarse_prior


def clean_stage(model, tokens, device):
    x = torch.as_tensor(tokens, dtype=torch.long, device=device)
    logits, mu, _ = model(x)
    lp = logits[:, 1:].log_softmax(-1)
    target = x[:, 1:]
    valid = target != 0
    error = (-lp.gather(-1, target[..., None]).squeeze(-1)).masked_fill(~valid, 0)
    entropy = (-(lp.exp()*lp).sum(-1)).masked_fill(~valid, 0)
    e, h, m = error.cpu().numpy(), entropy.cpu().numpy(), valid.cpu().numpy()
    p = dict(latent=mu.cpu().numpy(), event=e, entropy=h,
             length=m.sum(1), stats=_legacy_statistics(e, h, m),
             native=(error.sum(1)/valid.sum(1).clamp(min=1)).cpu().numpy(),
             tokens=target.cpu().numpy())
    return p, x, lp


def global_stage(evidence, p):
    # Reproduce each original float32 cast, including the frozen V15 prior.
    assert evidence.mode == 'optimized'
    base = evidence.frozen_v15
    reference = base.reference
    compat = reference.compat
    g = np.column_stack((robust_apply(compat.global_raw(p), compat.global_fit),
                         compat.ambiguity(p))).astype(np.float32)
    clean = reference.clean
    raw = np.log1p(np.asarray(p['stats'])[:, [0, 2, 4]])
    coarse = clean.coarse_background.transform(raw, p['length'], clean._modes(p))
    cm = (coarse-clean.coarse_x)/clean.coarse_scale
    g = np.column_stack((g, np.arcsinh(cm))).astype(np.float32)
    z = robust_apply(p['latent'], compat.latent_fit)
    logits = -((z[:, None]-compat.centers[None])**2).sum(2)/(2*compat.sigma**2)
    logits -= logits.max(1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(1, keepdims=True)
    m, effective, _ = membership_terms(weights)
    g = np.column_stack((g, 3*m, 3*effective)).astype(np.float32)
    # Original V14 casts cm before V15 distance; V16 does not replace this scale.
    prior = base.distance(cm.astype(np.float32)).astype(np.float32)
    return g, prior


def fine_stage(model, evidence, p, x, clean_lp, ix, audit=False):
    # Clean features are gathered from the first pass; never recompute clean VAE.
    pack = {k: v[ix] for k, v in p.items()}
    hidden_lp = masked_logits(model, x[ix])[:, 1:].log_softmax(-1)
    hidden, tail, stats, _ = _view_pack(hidden_lp, x[ix, 1:])
    for k, v in hidden.items():
        pack['masked_'+k] = v
    pack['masked_tail_event'], pack['masked_tail_stats'] = tail, stats
    if evidence.tail_source == 'clean':
        clean_tail = categorical_tail(clean_lp[ix], x[ix, 1:]).cpu().numpy()
        pack['clean_tail_event'] = clean_tail
        pack['clean_tail_stats'] = _tail_statistics(clean_tail, pack['tokens'] != 0)
    features = evidence.transform(pack)
    return (features, pack, hidden_lp.cpu().numpy()) if audit else features


def low_deferred(low, global_state, prior, goal, fine_features, device):
    """Original LowQ arithmetic; local arrays contain fine rows only."""
    assert not low.mlp
    global_h = low.global_encoder(global_state)
    local = torch.zeros((len(goal), 48), device=device)
    fine = goal == 1
    if bool(fine.any()):
        seq = torch.as_tensor(fine_features['seq'], device=device)
        length = torch.as_tensor(fine_features['length'], device=device)
        extra = torch.as_tensor(fine_features['extra'], device=device)
        packed = pack_padded_sequence(seq, length.cpu(), batch_first=True, enforce_sorted=False)
        h = low.local_encoder(packed)[1][0][-1]
        local[fine] = torch.cat((low.local_stats(extra), h), 1)
    residual = low.head(torch.cat((global_h, local, F.one_hot(goal, 2).float()), 1))
    return coarse_prior({'coarse_reward_prior': prior}) + residual


@torch.no_grad()
def predict(tokens, model, evidence, agent, device, force_goal=None, audit=False):
    scores, classes, routes, diagnostics = [], [], [], []
    for start in range(0, len(tokens), 128):
        clean, x, lp = clean_stage(model, tokens[start:start+128], device)
        glob, prior = global_stage(evidence, clean)
        global_tensor = torch.as_tensor(glob, device=device)
        goal = (agent.high(global_tensor).argmax(1) if force_goal is None else
                torch.full((len(x),), force_goal, dtype=torch.long, device=device))
        ix = torch.where(goal == 1)[0].cpu().numpy()
        fine_raw = None
        masked_logp = None
        if len(ix):
            acquired = fine_stage(model, evidence, clean, x, lp, ix, audit=audit)
            if audit:
                fine, fine_raw, masked_logp = acquired
            else:
                fine = acquired
        else:
            fine = None
        q = low_deferred(agent.low, global_tensor, torch.as_tensor(prior, device=device), goal, fine, device)
        q = mask_q(q, goal)
        score = coarse_fine_advantage(q, goal).cpu().numpy()
        label = q.argmax(1)
        label[(goal == 0) & (label > 0)] = -1
        assert torch.equal(label != 0, torch.as_tensor(score > 0, device=device))
        scores.append(score)
        classes.append(label.cpu().numpy())
        routes.append(goal.cpu().numpy())
        if audit:
            diagnostics.append(dict(start=start, global_state=glob, prior=prior,
                                    fine_indices=ix, fine_features=fine, fine_raw=fine_raw,
                                    masked_logp=masked_logp))
    result = np.concatenate(scores), np.concatenate(classes), np.concatenate(routes)
    return (result, diagnostics) if audit else result
