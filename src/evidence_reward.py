"""Paper equations 9/11 with robust units and an explicit anomaly safety gate.

masked's VAE, score correction, x/y and latent clusters are frozen. This wrapper
changes rewards and exposes their global membership inputs, never test labels.
"""
import copy
import numpy as np
import torch
from common import CFG
from evidence_masked import MaskedEvidence
from representation import robust_apply
from policy import to_tensor as legacy_to_tensor, teacher_inputs as legacy_teacher_inputs


def membership_terms(weights):
    weights = np.asarray(weights, dtype=np.float64)
    assert weights.ndim == 2 and weights.shape[1] == 3
    assert np.all(weights >= 0) and np.allclose(weights.sum(1), 1.)
    ordered = np.sort(weights, axis=1)
    m = ordered[:, 0]
    top_two_ambiguity = 2 * ordered[:, 1] / np.maximum(ordered[:, 2] + ordered[:, 1], 1e-30)
    effective = m + (1/3 - m) * top_two_ambiguity
    return m, np.clip(effective, 0., 1/3), top_two_ambiguity


def to_tensor(p, index, device):
    result = legacy_to_tensor(p, index, device)
    result['coarse_reward_prior'] = torch.as_tensor(p['coarse_reward_prior'][index], device=device)
    return result


def teacher_inputs(p, index):
    result = legacy_teacher_inputs(p, index)
    for key in ('membership_min', 'membership_effective'):
        result[key] = p[key][index]
    return result


class RewardEvidence:
    def fit(self, reference, train, validation, seed):
        assert type(reference) is MaskedEvidence
        assert not any(k in p for p in (train, validation) for k in ('y', 'labels', 'y_three', 'test'))
        self.reference = reference
        self.mode = 'optimized'
        self.beta = float(CFG['paper_reward_beta'])
        self.lam = float(CFG['paper_reward_lambda'])
        assert self.beta > 0 and self.lam > 0
        p = reference.transform(validation)
        self.reward_scale = max(float(np.quantile(p['fine_margin'], .9) - np.quantile(p['fine_margin'], .1)),
                                float(CFG['paper_reward_scale_floor']))
        for key in ('x', 'y', 'native_x', 'native_y', 'cluster_centers', 'sigma', 'threshold_diagnostics', 'optimization_diagnostics'):
            setattr(self, key, copy.deepcopy(getattr(reference, key)))
        assert self.x == 0 and self.y > self.x
        self.reward_diagnostics = dict(beta=self.beta, lambda_=self.lam,
            scale=self.reward_scale, scale_reference='original unlabeled validation Q90-Q10',
            internal_distance='beta * asinh((delta-boundary)/scale)',
            delta='frozen masked entropy-excess reconstruction margin; x=0; masked y unchanged',
            fine_reward='binary gate plus anomalous-only subtype utility minus its maximum',
            normal_fine_subtype_training=False, external='equation 9 with ambiguity-corrected min membership',
            high_return='same binary utility for both branches; subtype reward trains low-level only',
            x_tie='normal', y_tie='simple', fine_gate='learned normal-vs-anomaly Q advantage',
            test_labels_used=False, exact_manuscript_reproduction=False)
        return self

    def raw(self):
        result = copy.copy(self)
        result.mode = 'paper_raw'
        result.reward_scale = 1.
        return result

    def distance(self, margin):
        value = np.asarray(margin, dtype=np.float64)
        if self.mode == 'optimized':
            value = np.arcsinh(value / self.reward_scale)
        return self.beta * value

    def transform(self, p):
        result = self.reference.transform(p)
        compat = self.reference.compat
        z = robust_apply(p['latent'], compat.latent_fit)
        logits = -((z[:, None] - compat.centers[None])**2).sum(2) / (2 * compat.sigma**2)
        logits -= logits.max(1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(1, keepdims=True)
        m, effective, top_two = membership_terms(weights)
        result.update(membership_min=m.astype(np.float32), membership_effective=effective.astype(np.float32),
            membership_top_two=top_two.astype(np.float32),
            coarse_reward_prior=self.distance(result['coarse_margin']).astype(np.float32))
        # These are existing latent-space reward signals, not new detectors.
        result['global_state'] = np.column_stack((result['global_state'], 3*m, 3*effective)).astype(np.float32)
        # Strict x boundary: equality is normal; equality at y is simple.
        fm = result['fine_margin']
        result['pseudo'] = ((fm > 0).astype(np.int64) + (fm > self.y).astype(np.int64))
        assert all(np.isfinite(v).all() for v in result.values())
        return result

    def gate_rewards(self, p):
        dx = self.distance(p['fine_margin'])
        return np.column_stack((-dx, dx)).astype(np.float32)

    def rewards(self, p, goal):
        margin = np.asarray(p['fine_margin'], dtype=np.float64)
        dx = self.distance(margin)
        dy = self.distance(margin - self.y)
        r = np.column_stack((-dx, dx, dx))
        use = (np.asarray(goal) == 1) & (margin > 0)
        # Equivalent to dx + [ -dy, +dy ] - |dy|, only on anomalies.
        # The winning subtype keeps exactly dx; it cannot boost binary returns.
        r[use, 1] -= 2*np.maximum(dy[use], 0)
        r[use, 2] -= 2*np.maximum(-dy[use], 0)
        return r.astype(np.float32)

    def external(self, p, goal):
        m = p['membership_effective' if self.mode == 'optimized' else 'membership_min']
        # Paper equation 9, including its original 1/6 crossover.
        return (self.lam * (2*np.asarray(goal)-1) * (2*np.asarray(m)-1/3)).astype(np.float32)
