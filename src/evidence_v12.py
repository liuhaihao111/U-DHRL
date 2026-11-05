"""Frozen single-VAE evidence with unlabeled held-out background references.

This module never sees anomaly labels, injected examples or benchmark rates.
Its tail budgets describe the original reference distribution, not a guaranteed
normal false-positive rate. All fitting precedes benchmark evaluation.
"""
import pickle
import numpy as np
from common import CFG
from representation import EvidenceV10, robust_fit, robust_apply, adaptive_thresholds


def _width(a):
    a = np.asarray(a, dtype=np.float64)
    return max(float(np.quantile(a, .9) - np.quantile(a, .1)), .1)


def _mad(a, floor=.05):
    a = np.asarray(a, dtype=np.float64)
    return max(float(np.median(np.abs(a - np.median(a))) * 1.4826), floor)


def local_surprise(p):
    """Reconstruction surprisal beyond the same VAE's predictive entropy."""
    event = np.asarray(p['event'], dtype=np.float64)
    entropy = np.asarray(p['entropy'], dtype=np.float64)
    assert event.shape == entropy.shape
    lengths = np.asarray(p['length'], dtype=np.int64)
    rows = []
    for errors, uncertainty, n in zip(event, entropy, lengths):
        assert 1 <= n <= event.shape[1]
        a = np.maximum(errors[:n] - uncertainty[:n], 0.)
        top = np.sort(a)[-min(3, n):]
        rows.append((a.mean(), top.mean(), a.max()))
    return np.log1p(np.asarray(rows, dtype=np.float64))


class _Background:
    """Low-dimensional robust background regression and shrunken spread.

    Latent-mode identities and length-bin edges are fixed from training data.
    Original held-out scores alone fit regression/reference scales. The bounded
    correction cannot erase an arbitrarily large reconstruction deviation.
    """
    def __init__(self, edges):
        self.edges = np.asarray(edges, dtype=np.float64)

    def _design(self, lengths, modes, fit=False):
        loglength = np.log1p(np.asarray(lengths, dtype=np.float64))
        modes = np.asarray(modes, dtype=np.int64)
        if fit:
            self.length_center = float(np.median(loglength))
            self.length_scale = _mad(loglength, .1)
            self.mode_mass = np.bincount(modes, minlength=3) / len(modes)
        l = np.clip((loglength - self.length_center) / self.length_scale, -4., 4.)
        if fit:
            self.length_mean = float(l.mean())
            self.square_mean = float((l * l).mean())
        return np.column_stack((np.eye(3)[modes] - self.mode_mass,
                                l - self.length_mean, l * l - self.square_mean))

    def _groups(self, lengths, modes):
        bins = np.searchsorted(self.edges, lengths, side='right')
        return np.asarray(modes, dtype=np.int64) * (len(self.edges) + 1) + bins

    def fit(self, raw, lengths, modes):
        self.reference_fit = robust_fit(raw)
        score = robust_apply(raw, self.reference_fit) @ np.array([.25, .5, .25])
        self.center = float(np.median(score))
        self.correction_limit = float(CFG.get('background_residual_mix', .5) * _width(score))
        x = self._design(lengths, modes, fit=True)
        y = score - self.center
        self.beta = np.zeros(x.shape[1], dtype=np.float64)
        ridge = float(CFG.get('background_ridge', 10.))
        # Fixed eight Huber IRLS updates. No checkpoint or labels choose a fit.
        for _ in range(8):
            residual = y - x @ self.beta
            scale = _mad(residual)
            weight = np.minimum(1., 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
            self.beta = np.linalg.solve(x.T @ (weight[:, None] * x) + ridge * np.eye(x.shape[1]),
                                        x.T @ (weight * y))
        residual = y - x @ self.beta
        self.residual_scale = _mad(residual)
        groups = self._groups(lengths, modes)
        count = 3 * (len(self.edges) + 1)
        self.group_scale = np.full(count, self.residual_scale)
        self.group_counts = np.bincount(groups, minlength=count)
        shrink = float(CFG.get('reference_shrinkage', 32.))
        for g, n in enumerate(self.group_counts):
            if n < 3:
                continue
            loc = _mad(residual[groups == g], max(.05, .1 * self.residual_scale))
            w = float(n / (n + shrink))
            self.group_scale[g] = w * loc + (1. - w) * self.residual_scale
        transformed = self.transform(raw, lengths, modes)
        self.diagnostics = dict(reference_cases=len(score),
            mode_counts=np.bincount(modes, minlength=3).tolist(),
            group_counts=self.group_counts.tolist(), ridge=ridge,
            shrinkage_cases=shrink, correction_limit=self.correction_limit,
            coefficients=self.beta.tolist(), residual_scale=self.residual_scale,
            reference_raw_width=_width(score), reference_adjusted_width=_width(transformed),
            max_reference_correction=float(np.max(np.abs(transformed - score))))
        return self

    def transform(self, raw, lengths, modes):
        base = robust_apply(raw, self.reference_fit) @ np.array([.25, .5, .25])
        bias = self._design(lengths, modes) @ self.beta
        scale = self.group_scale[self._groups(lengths, modes)]
        adjusted = self.center + (base - self.center - bias) * self.residual_scale / scale
        return base + np.clip(adjusted - base, -self.correction_limit, self.correction_limit)


class EvidenceV12:
    """Different coarse/local pseudo-rewards, fitted on original held-out cases."""
    def _modes(self, p):
        z = robust_apply(p['latent'], self.compat.latent_fit)
        return ((z[:, None] - self.centers[None]) ** 2).sum(2).argmin(1)

    def _cutoff(self, score, budget, seed):
        x, y, info = adaptive_thresholds(score, seed)
        guard = float(np.quantile(score, 1. - budget, method='higher'))
        final_x = max(float(x), guard)
        final_y = max(float(y), final_x + max(1e-4, .1 * _width(score)))
        return final_x, final_y, dict(info, kmeans_x=float(x), kmeans_y=float(y),
            reference_tail_budget=float(budget), reference_quantile_guard=guard,
            final_x=final_x, final_y=final_y,
            observed_reference_positive_fraction=float(np.mean(score > final_x)),
            guarantee_of_normal_fpr=False, test_distribution_used=False)

    def fit(self, p_train, p_reference, seed):
        assert len(p_train['length']) >= 3 and len(p_reference['length']) >= 3
        # Retain dimensions/scalers of the original coarse policy's state.
        # Only training data identify latent modes and fixed length groups.
        self.compat = EvidenceV10().fit(p_train, seed)
        self.centers = self.compat.centers.copy()
        self.sigma = self.compat.sigma
        modes = self._modes(p_reference)
        raw = np.log1p(np.asarray(p_reference['stats'])[:, [0, 2, 4]])
        self.coarse_background = _Background(self.compat.edges).fit(raw, p_reference['length'], modes)
        self.local_background = _Background(self.compat.edges).fit(local_surprise(p_reference),
                                                                  p_reference['length'], modes)
        coarse = self.coarse_background.transform(raw, p_reference['length'], modes)
        local = self.local_background.transform(local_surprise(p_reference), p_reference['length'], modes)
        coarse_budget = float(CFG.get('coarse_tail_budget', .25))
        fine_budget = float(CFG.get('fine_tail_budget', .20))
        assert 0. < fine_budget <= coarse_budget < 1.
        self.coarse_x, self.coarse_y, cinfo = self._cutoff(coarse, coarse_budget, seed)
        self.local_x, self.local_y, linfo = self._cutoff(local, fine_budget, seed)
        self.coarse_scale, self.local_scale = _width(coarse), _width(local)
        self.local_weight = float(CFG.get('local_margin_weight', .3))
        assert 0. < self.local_weight < 1.
        coarse_margin = (coarse - self.coarse_x) / self.coarse_scale
        local_margin = (local - self.local_x) / self.local_scale
        combined = (1. - self.local_weight) * coarse_margin + self.local_weight * local_margin
        # The component guards alone do not constrain the mixture's tail.
        # Apply a final one-sided guard to the ORIGINAL held-out reference,
        # never to the test batch or an assumed benchmark injection rate.
        self.fine_shift = max(0., float(np.quantile(combined, 1. - fine_budget, method='higher')))
        final_reference_margin = combined - self.fine_shift
        self.coarse_severity_margin = (self.coarse_y - self.coarse_x) / self.coarse_scale
        local_severity = (self.local_y - self.local_x) / self.local_scale
        self.fine_severity_margin = max(1e-4,
            (1. - self.local_weight) * self.coarse_severity_margin +
            self.local_weight * local_severity - self.fine_shift)
        # score is the combined fine margin, so the direct evidence control has
        # exactly the same zero boundary as the fine branch pseudo-rewards.
        self.x, self.y, self.reward_scale = 0., self.fine_severity_margin, 1.
        native = np.log1p(np.asarray(p_reference['native'], dtype=np.float64))
        self.native_x, self.native_y, native_info = self._cutoff(native, coarse_budget, seed)
        self.threshold_diagnostics = dict(reference='original validation plus calibration; unlabeled',
            reference_cases=len(coarse), coarse=cinfo, local=linfo, native=native_info,
            boundary_in_margin_units=0., local_margin_weight=self.local_weight,
            combined_reference_positive_fraction_before_guard=float(np.mean(combined > 0.)),
            combined_reference_positive_fraction=float(np.mean(final_reference_margin > 0.)),
            combined_reference_tail_budget=fine_budget, fine_margin_shift=self.fine_shift,
            original_sequences_not_verified_normal=True)
        self.native_threshold_diagnostics = native_info
        self.cluster_centers = dict(coarse=cinfo['centers'], local=linfo['centers'])
        self.optimization_diagnostics = dict(
            coarse_background=self.coarse_background.diagnostics,
            local_background=self.local_background.diagnostics,
            coarse_scale=self.coarse_scale, local_scale=self.local_scale,
            latent_centers_fitted_on_train_only=True, extra_policy_features_added=True,
            appended_state_features=['coarse_margin', 'fine_margin', 'local_margin', 'disagreement'],
            appended_margin_clip=[-10., 10.],
            labels_used=False, other_model_outputs_used=False)
        return self

    def _scores(self, p):
        modes = self._modes(p)
        raw = np.log1p(np.asarray(p['stats'])[:, [0, 2, 4]])
        coarse = self.coarse_background.transform(raw, p['length'], modes)
        local = self.local_background.transform(local_surprise(p), p['length'], modes)
        cm = (coarse - self.coarse_x) / self.coarse_scale
        lm = (local - self.local_x) / self.local_scale
        fm = (1. - self.local_weight) * cm + self.local_weight * lm - self.fine_shift
        return coarse, local, cm, fm, lm

    def evidence_score(self, p):
        return self._scores(p)[3]

    def transform(self, p):
        result = self.compat.transform(p)
        coarse, local, cm, fm, lm = self._scores(p)
        disagreement = np.tanh(np.abs(fm - cm)).astype(np.float32)
        added = np.column_stack((np.clip(cm, -10., 10.), np.clip(fm, -10., 10.),
                                 np.clip(lm, -10., 10.), disagreement)).astype(np.float32)
        result['global_state'] = np.column_stack((result['global_state'], added)).astype(np.float32)
        result['extra'] = np.column_stack((result['extra'], added)).astype(np.float32)
        result.update(score=fm.astype(np.float32),
            coarse_score=coarse.astype(np.float32), local_score=local.astype(np.float32),
            fine_score=(fm + self.fine_shift).astype(np.float32), coarse_margin=cm.astype(np.float32),
            fine_margin=fm.astype(np.float32), local_margin=lm.astype(np.float32),
            disagreement=disagreement,
            pseudo=np.searchsorted([0., self.fine_severity_margin], fm).astype(np.int64))
        assert all(np.isfinite(a).all() for a in result.values())
        return result

    def rewards(self, p, goal):
        goal = np.asarray(goal, dtype=np.int64)
        assert goal.shape == np.asarray(p['score']).shape
        fine = goal == 1
        margin = np.where(fine, p['fine_margin'], p['coarse_margin'])
        d = np.arcsinh(margin)
        r = np.column_stack((-d, d, d)).astype(np.float32)
        simple = np.asarray(p['fine_margin']) < self.fine_severity_margin
        r[fine & simple, 2] -= .1
        r[fine & ~simple, 1] -= .1
        return r

    def external(self, p, goal):
        return (np.asarray(goal) * (CFG['ambiguity_weight'] * np.asarray(p['ambiguity']) +
            CFG.get('fine_disagreement_weight', .06) * np.asarray(p['disagreement']) -
            CFG['refine_cost'])).astype(np.float32)


def selfcheck():
    """Generated-feature software checks; no dataset or anomaly-label access."""
    rng = np.random.default_rng(12)
    def pack(n):
        length = rng.integers(5, 52, n)
        mode = rng.integers(0, 3, n)
        latent = rng.normal(size=(n, 8)) + mode[:, None] * 2.
        event = rng.gamma(2., .3, (n, 51)) + .01 * length[:, None] + .1 * mode[:, None]
        entropy = np.full_like(event, .5)
        stats = []
        for i, k in enumerate(length):
            a, b = event[i, :k], entropy[i, :k]
            top = np.sort(a)[::-1]
            stats.append([a.mean(), a.std(), top[:3].mean(), top[:5].mean(), a.max(),
                          np.quantile(a, .9), b.mean(), np.maximum(a-b, 0).mean(),
                          np.maximum(a-b, 0).max()])
            event[i, k:] = 0.; entropy[i, k:] = 0.
        stats = np.asarray(stats)
        return dict(latent=latent, length=length, event=event, entropy=entropy,
                    stats=stats, native=stats[:, 0])
    train, reference, query = pack(240), pack(90), pack(32)
    teacher = EvidenceV12().fit(train, reference, 42)
    transformed = teacher.transform(query)
    clone = pickle.loads(pickle.dumps(teacher))
    for key, a in transformed.items():
        assert len(a) == 32 and np.array_equal(a, clone.transform(query)[key]), key
    assert transformed['global_state'].shape[1] == teacher.compat.transform(query)['global_state'].shape[1] + 4
    assert transformed['extra'].shape[1] == teacher.compat.transform(query)['extra'].shape[1] + 4
    assert np.allclose(transformed['global_state'][:, -4], np.clip(transformed['coarse_margin'], -10., 10.))
    assert np.allclose(transformed['global_state'][:, -3], np.clip(transformed['fine_margin'], -10., 10.))
    coarse = teacher.rewards(transformed, np.zeros(32, int))
    fine = teacher.rewards(transformed, np.ones(32, int))
    assert not np.allclose(coarse, fine)
    assert np.allclose(coarse[:, 1] - coarse[:, 0], 2*np.arcsinh(transformed['coarse_margin']))
    assert np.isfinite(teacher.external(transformed, np.ones(32, int))).all()
    assert teacher.threshold_diagnostics['coarse']['observed_reference_positive_fraction'] <= CFG.get('coarse_tail_budget', .25) + 1/90
    assert teacher.threshold_diagnostics['local']['observed_reference_positive_fraction'] <= CFG.get('fine_tail_budget', .20) + 1/90
    assert teacher.threshold_diagnostics['combined_reference_positive_fraction'] <= CFG.get('fine_tail_budget', .20) + 1/90
    # Query batch composition must not refit references or thresholds.
    one = {k: v[:1] for k, v in query.items()}
    assert np.allclose(teacher.transform(one)['score'], transformed['score'][:1])
    return dict(ok=True, fitting_source='generated feature arrays', benchmark_labels_read=False,
                branch_rewards_differ=True, batch_invariant=True, pickle_roundtrip=True,
                coarse_reference_tail=teacher.threshold_diagnostics['coarse']['observed_reference_positive_fraction'],
                local_reference_tail=teacher.threshold_diagnostics['local']['observed_reference_positive_fraction'])


if __name__ == '__main__':
    import json
    print(json.dumps(selfcheck(), indent=2))
