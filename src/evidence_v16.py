"""Bounded same-VAE tail correction of frozen V15 fine reconstruction evidence.

The V15 internal/external reward equations, distance function and raw-reward
ablation are inherited without overrides.  Only the fine evidence and its
reward scale change.  Coarse inputs, coarse reconstruction prior, VAE weights,
latent memberships and all historical reference outputs remain unchanged.

Original held-out sequences are not certified normal.  Their tail budget is
an empirical reference budget, not a false-positive-rate guarantee.  The new
categorical tail mass is reconstruction evidence, not an anomaly p-value.
"""
import copy

import numpy as np

from common import CFG
from evidence_v12 import _Background
from evidence_v15 import EvidenceV15


def _width(values, floor):
    values = np.asarray(values, dtype=np.float64)
    return max(float(np.quantile(values, .9) - np.quantile(values, .1)), float(floor))


def bounded_tail_correction(old_margin, tail_margin, rho, cap):
    """Fixed, symmetric, bounded interpolation; never a test-fitted threshold."""
    old_margin = np.asarray(old_margin, dtype=np.float64)
    tail_margin = np.asarray(tail_margin, dtype=np.float64)
    assert old_margin.shape == tail_margin.shape
    assert 0. <= rho <= .25 and cap > 0.
    correction = float(rho) * np.clip(tail_margin - old_margin, -float(cap), float(cap))
    return old_margin + correction, correction


class EvidenceV16(EvidenceV15):
    def fit(self, reference, train, validation, seed):
        """Fit original-validation tail background with frozen V15 as reference.

        No anomaly labels, test scores, baseline outputs or injection rates are
        accepted.  The 0.25 maximum weight is fixed before benchmark inspection.
        n/(n+32) shrinks a sparse fitted reference towards unchanged V15; it is
        a regularization heuristic, not a confidence bound or a retention
        guarantee.  The cap is one original-V15 reference Q90-Q10 width.
        """
        assert type(reference) is EvidenceV15 and reference.mode == 'optimized'
        assert not any(k in p for p in (train, validation)
                       for k in ('y', 'labels', 'y_three', 'test'))
        assert len(train['length']) >= 3 and len(validation['length']) >= 3
        # The stored wrapper is read-only.  Preserve its globals and prior by
        # calling it during transform rather than recomputing the prior with
        # the new fine reward scale.
        self.__dict__ = reference.__dict__.copy()
        self.frozen_v15 = reference
        self.reward_diagnostics = copy.deepcopy(reference.reward_diagnostics)
        self.threshold_diagnostics = copy.deepcopy(reference.threshold_diagnostics)
        self.optimization_diagnostics = copy.deepcopy(reference.optimization_diagnostics)
        self.tail_source = 'masked' if bool(reference.reference.masked_enabled) else 'clean'
        self.tail_floor = float(CFG.get('tail_evidence_scale_floor', .1))
        self.tail_weight = float(CFG.get('tail_evidence_weight', .25))
        self.tail_shrinkage_cases = float(CFG.get('tail_evidence_shrinkage_cases', 32.))
        assert self.tail_floor > 0. and 0. <= self.tail_weight <= .25
        assert self.tail_shrinkage_cases >= 0.
        n = len(validation['length'])
        self.tail_rho = float(self.tail_weight * n / (n + self.tail_shrinkage_cases))
        old = reference.transform(validation)
        old_margin = np.asarray(old['fine_margin'], dtype=np.float64)
        self.tail_cap = _width(old_margin, self.tail_floor)
        modes = reference.reference.clean._modes(validation)
        self.tail_background = _Background(reference.reference.compat.edges).fit(
            self._tail_raw(validation), validation['length'], modes)
        tail_score = self.tail_background.transform(self._tail_raw(validation),
                                                     validation['length'], modes)
        self.tail_x, self.tail_y, cutoff_info = reference.reference.clean._cutoff(
            tail_score, float(CFG['fine_tail_budget']), seed)
        self.tail_scale = _width(tail_score, self.tail_floor)
        tail_margin = (tail_score - self.tail_x) / self.tail_scale
        corrected, correction = bounded_tail_correction(old_margin, tail_margin,
                                                         self.tail_rho, self.tail_cap)
        old_scale = float(reference.reward_scale)
        self.reward_scale = _width(corrected, float(CFG['paper_reward_scale_floor']))
        self.tail_diagnostics = dict(
            source='original unlabeled validation only', validation_cases=n,
            original_sequences_verified_normal=False,
            feature_source=self.tail_source + '_tail_stats',
            feature_statistics=['mean', 'top_three_mean', 'maximum'],
            feature_transform='log1p of raw negative-log categorical tail mass',
            categorical_tail='sum probability of all categories <= observed probability, including ties',
            categorical_tail_is_anomaly_p_value=False,
            target_visibility='same frozen VAE target-hidden view when enabled; otherwise clean reconstruction',
            background=self.tail_background.diagnostics,
            background_modes='frozen V14 latent modes; training-fitted centers',
            background_groups='frozen V14 training length-bin edges',
            cutoff=cutoff_info, tail_x=float(self.tail_x), tail_y=float(self.tail_y),
            tail_scale=float(self.tail_scale),
            fixed_maximum_weight=float(self.tail_weight),
            shrinkage_cases=float(self.tail_shrinkage_cases), effective_weight=float(self.tail_rho),
            shrinkage_rule='weight * validation_cases / (validation_cases + shrinkage_cases)',
            cap=float(self.tail_cap), cap_units='original V15 validation fine-margin Q90-Q10, floored',
            maximum_absolute_correction=float(self.tail_rho*self.tail_cap),
            observed_maximum_absolute_correction=float(np.max(np.abs(correction))),
            correction_formula='rho * clip(tail_margin - old_fine_margin, -cap, +cap)',
            correction_quantiles=np.quantile(correction, [0., .1, .5, .9, 1.]).tolist(),
            old_reference_positive_fraction=float(np.mean(old_margin > 0.)),
            corrected_reference_positive_fraction=float(np.mean(corrected > 0.)),
            reference_sign_flip_fraction=float(np.mean((old_margin > 0.) != (corrected > 0.))),
            reference_positive_to_negative_fraction=float(np.mean((old_margin > 0.) & (corrected <= 0.))),
            reference_negative_to_positive_fraction=float(np.mean((old_margin <= 0.) & (corrected > 0.))),
            correction_direction='symmetric; no global threshold tightening',
            correction_cannot_flip_beyond_original_margin_magnitude=float(self.tail_rho*self.tail_cap),
            original_reward_scale=old_scale, corrected_reward_scale=float(self.reward_scale),
            x=float(self.x), y=float(self.y), anomaly_boundary_unchanged=True,
            severity_boundary_unchanged=True,
            severity_boundary_limitation='old numerical y retained; corrected margin is not ground-truth severity',
            globals_and_coarse_prior='frozen V15 arithmetic, including original coarse reward scale',
            new_local_features=['asinh(tail_margin)', 'asinh(tail_correction)'],
            internal_external_reward_equations='inherited V15 without overrides',
            fine_reward_scale_reestimated=True,
            no_post_policy_score_override=True,
            no_new_detector_or_model_parameters=True,
            unknown_token_probability_adjustment=False,
            anomaly_labels_used=False, test_used=False, baseline_outputs_used=False,
            synthetic_anomaly_injection_used=False,
            selection_on_test_f1=False, dataset_specific_performance_tuning=False,
            limitations=[
                'Tail mass is model-dependent reconstruction evidence, not a calibrated anomaly probability.',
                'Original validation can contain natural abnormalities and is not a verified normal sample.',
                'Reference-count shrinkage and bounded corrections constrain changes but do not guarantee F1 retention.',
                'Rare legitimate activities can still have small tail mass; true anomalies can remain probable.',
                'A preserved coarse prior may require residual adjustment to the new fine reward scale.',
                'Repeatedly studied historical benchmarks remain exploratory evidence.'
            ])
        self.threshold_diagnostics['v16_tail_correction'] = self.tail_diagnostics
        self.optimization_diagnostics['v16_tail_correction'] = self.tail_diagnostics
        self.reward_diagnostics.update(
            delta='frozen V15 fine margin plus bounded same-VAE categorical-tail correction; x/y unchanged',
            scale_reference='corrected original unlabeled validation Q90-Q10',
            v16_reward_equations_unchanged=True,
            v16_tail_source=self.tail_source,
            v16_effective_tail_weight=float(self.tail_rho),
            v16_maximum_absolute_correction=float(self.tail_rho*self.tail_cap),
            coarse_prior_uses_frozen_v15_scale=True)
        return self

    def _tail_raw(self, p):
        values = np.asarray(p[self.tail_source + '_tail_stats'], dtype=np.float64)
        assert values.shape == (len(p['length']), 3)
        assert np.isfinite(values).all() and np.all(values >= 0.)
        return np.log1p(values)

    def transform(self, p):
        # raw() is inherited unchanged: it selects original linear distances
        # and original minimum membership, while sharing V16 corrected evidence.
        base = self.frozen_v15.raw() if self.mode == 'paper_raw' else self.frozen_v15
        result = base.transform(p)
        modes = self.reference.clean._modes(p)
        tail_score = self.tail_background.transform(self._tail_raw(p), p['length'], modes)
        tail_margin = (tail_score - self.tail_x) / self.tail_scale
        corrected, correction = bounded_tail_correction(result['fine_margin'], tail_margin,
                                                         self.tail_rho, self.tail_cap)
        fm = corrected.astype(np.float32)
        disagreement = np.tanh(np.abs(corrected - result['coarse_margin'])).astype(np.float32)
        extra = result['extra'].copy()
        assert extra.shape[1] >= 5
        # V14's last five local fields are cm, fm, lm, masked margin, disagreement.
        extra[:, -4] = np.arcsinh(corrected).astype(np.float32)
        extra[:, -1] = disagreement
        result['extra'] = np.column_stack((extra, np.arcsinh(tail_margin),
                                           np.arcsinh(correction))).astype(np.float32)
        result.update(score=fm.copy(), fine_margin=fm,
                      fine_score=(np.asarray(result['fine_score'], dtype=np.float64)+correction).astype(np.float32),
                      disagreement=disagreement,
                      pseudo=((fm > self.x).astype(np.int64)+(fm > self.y).astype(np.int64)),
                      tail_margin=tail_margin.astype(np.float32),
                      tail_correction=correction.astype(np.float32))
        assert all(np.isfinite(v).all() for v in result.values())
        return result


def selfcheck():
    """Generated-array correction checks, without experiment data or labels."""
    old = np.array([-10., -1., -.1, 0., .1, 1., 10.])
    tail = np.array([20., -20., 1., 0., -1., 20., -20.])
    corrected, correction = bounded_tail_correction(old, tail, .25, 1.)
    assert np.all(np.abs(correction) <= .25)
    assert np.all((corrected[np.abs(old) > .25] > 0.) == (old[np.abs(old) > .25] > 0.))
    assert corrected[2] > 0. and corrected[4] < 0.
    identity, zero = bounded_tail_correction(old, tail, 0., 1.)
    assert np.array_equal(identity, old) and np.array_equal(zero, np.zeros_like(old))
    unchanged, zero = bounded_tail_correction(old, old, .25, 1.)
    assert np.array_equal(unchanged, old) and np.array_equal(zero, np.zeros_like(old))
    assert EvidenceV16.rewards is EvidenceV15.rewards
    assert EvidenceV16.external is EvidenceV15.external
    assert EvidenceV16.distance is EvidenceV15.distance
    assert EvidenceV16.raw is EvidenceV15.raw
    return dict(ok=True, generated_arrays_only=True, correction_bounded=True,
                symmetric_boundary_changes=True, far_from_boundary_signs_preserved=True,
                zero_weight_exact_identity=True, original_reward_methods_inherited=True)


if __name__ == '__main__':
    import json
    print(json.dumps(selfcheck(), indent=2))
