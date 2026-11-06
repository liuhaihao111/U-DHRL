"""Residual hierarchical Q learning in a common reconstruction-reward unit.

The initial low-level values are the coarse reconstruction prior. Both evidence
levels share a binary reward; fine anomalies additionally learn the paper's
conditional subtype reward. The fine teacher score is never added to the
output. The coarse branch cannot read local states. Frozen masked inference uses
the unchanged policy_masked_source.py and policy.py.
"""
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import CFG, seed_all, fingerprint, write
from policy import (LowQ as LegacyLowQ, Replay, mask_q,
                    coarse_fine_advantage, subset)
from evidence_reward import to_tensor, teacher_inputs


def hierarchical(variant):
    return variant in ('U-DHRL', 'Paper-Raw')


def _weights():
    return (float(CFG.get('policy_advantage_weight', .5)),
            float(CFG.get('policy_boundary_weight', .05)))


def coarse_prior(b):
    """Only globally available coarse evidence enters this analytic prior."""
    d = b['coarse_reward_prior']
    assert d.ndim == 1
    return torch.stack((-d, d, d), dim=1)


class LowQ(LegacyLowQ):
    def __init__(self, gdim, edim, mlp=False):
        super().__init__(gdim, edim, mlp)
        # Zero residual is an interpretable starting policy, not an output
        # interpolation with the teacher after training.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, b, goal):
        return coarse_prior(b) + super().forward(b, goal)


class Agent(nn.Module):
    def __init__(self, gdim, edim, variant, device):
        super().__init__()
        assert variant in ('U-DHRL', 'Paper-Raw', 'Flat-Q', 'Pseudo-MLP')
        self.variant, self.device = variant, device
        self.low = LowQ(gdim, edim, variant == 'Pseudo-MLP')
        self.high = nn.Sequential(nn.Linear(gdim, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        # Kept for the checkpoint interface. hierarchical never fits or applies offsets.
        self.register_buffer('decision_offsets', torch.zeros(2))
        self.to(device)

    @torch.no_grad()
    def predict(self, p):
        self.eval()
        assert not bool(torch.any(self.decision_offsets != 0)), 'hierarchical has no decision offsets'
        scores, classes, goals = [], [], []
        for start in range(0, len(p['score']), 128):
            ix = np.arange(start, min(start + 128, len(p['score'])))
            b = to_tensor(p, ix, self.device)
            goal = (self.high(b['global_state']).argmax(1) if hierarchical(self.variant)
                    else torch.ones(len(ix), dtype=torch.long, device=self.device))
            q = mask_q(self.low(b, goal), goal)
            score = coarse_fine_advantage(q, goal)
            label = q.argmax(1)
            label[(goal == 0) & (label > 0)] = -1
            assert torch.equal(label != 0, score > 0)
            scores.append(score.cpu().numpy())
            classes.append(label.cpu().numpy())
            goals.append(goal.cpu().numpy())
        return np.concatenate(scores), np.concatenate(classes), np.concatenate(goals)


def load_policy_weights(agent, saved):
    """hierarchical checkpoints are not interchangeable with historical plain Q heads."""
    assert saved['variant'] == agent.variant
    assert saved.get('q_parameterization') == 'paper_coarse_prior_plus_learned_residual'
    agent.load_state_dict(saved['state'], strict=True)
    assert not bool(torch.any(agent.decision_offsets != 0))


def _decision_terms(q, target, goal):
    advantage = coarse_fine_advantage(q, goal)
    desired = coarse_fine_advantage(target, goal)
    confidence = desired.detach().abs() / (1. + desired.detach().abs())
    boundary = F.binary_cross_entropy_with_logits(
        advantage, (desired > 0).to(advantage.dtype), reduction='none') * confidence
    error = F.smooth_l1_loss(advantage, desired, reduction='none')
    wrong = ((advantage > 0) != (desired > 0)).to(advantage.dtype) * confidence
    return advantage, desired, error, boundary, wrong


def _decision_training_loss(q, target, goal):
    advantage, desired, error, boundary, _ = _decision_terms(q, target, goal)
    advantage_weight, boundary_weight = _weights()
    result = advantage_weight * error.mean() + boundary_weight * boundary.mean()
    # Keep the historical within-branch pair objective for every trainable
    # control, including the full-information Pseudo-MLP control.
    pair_losses = []
    for g in (0, 1):
        use = goal == g
        if int(use.sum()) > 1:
            a, d = advantage[use], desired[use]
            pair_losses.append(F.smooth_l1_loss(a - a.roll(1), d - d.roll(1)))
    if pair_losses:
        # Two branches must not silently double this coefficient relative to
        # the flat controls, whose replay contains only the fine branch.
        result = result + float(CFG['ranking_weight']) * torch.stack(pair_losses).mean()
    return result


@torch.no_grad()
def validation_metrics(agent, p, teacher):
    """Selection checks both branch fitting and the actually deployed route.

    Reward signs are unlabeled pseudo-targets, not verified anomaly labels.
    Route consistency makes it harder for lower absolute Q error to hide a
    degraded deployment boundary. It is not proof of true F1 improvement.
    """
    agent.eval()
    sums = dict(objective=0., branch_q_mse=0., branch_advantage_huber=0.,
                route_advantage_huber=0., route_boundary_loss=0.,
                route_confidence_weighted_sign_error=0.,
                route_pseudo_sign_agreement=0., high_return_mse=0.)
    count = 0
    advantage_weight, boundary_weight = _weights()
    sign_weight = float(CFG.get('policy_validation_sign_weight', .1))
    route_weight = float(CFG.get('policy_validation_route_weight', 1.))
    branch_ids = (0, 1) if hierarchical(agent.variant) else (1,)
    for start in range(0, len(p['score']), 128):
        ix = np.arange(start, min(start + 128, len(p['score'])))
        b = to_tensor(p, ix, agent.device)
        qs, targets, returns, branch_losses = [], [], [], []
        q_errors, advantage_errors = [], []
        for g in branch_ids:
            goal = torch.full((len(ix),), g, dtype=torch.long, device=agent.device)
            q = agent.low(b, goal)
            rr = torch.as_tensor(teacher.rewards(teacher_inputs(p, ix), np.full(len(ix), g)),
                                 dtype=q.dtype, device=agent.device)
            valid_q, valid_rr = (q[:, :2], rr[:, :2]) if g == 0 else (q, rr)
            q_error = F.mse_loss(valid_q, valid_rr, reduction='none').mean(1)
            _, _, ae, be, wrong = _decision_terms(q, rr, goal)
            branch_losses.append(q_error + advantage_weight * ae + boundary_weight * be
                                 + sign_weight * wrong)
            q_errors.append(q_error)
            advantage_errors.append(ae)
            selected = mask_q(q, goal).argmax(1)
            binary_rewards = torch.as_tensor(teacher.gate_rewards(teacher_inputs(p, ix)), dtype=q.dtype, device=agent.device)
            actual = binary_rewards.gather(1, (selected > 0).long()[:, None]).squeeze(1)
            external = torch.as_tensor(teacher.external(teacher_inputs(p, ix), np.full(len(ix), g)),
                                       dtype=q.dtype, device=agent.device)
            returns.append(external + float(CFG['gamma']) * actual)
            qs.append(q)
            targets.append(rr)
        if hierarchical(agent.variant):
            high = agent.high(b['global_state'])
            route = high.argmax(1)
            high_error = F.mse_loss(high, torch.stack(returns, 1), reduction='none').mean(1)
            row = torch.arange(len(ix), device=agent.device)
            chosen_q = torch.stack(qs, 1)[row, route]
            chosen_target = torch.stack(targets, 1)[row, route]
        else:
            route = torch.ones(len(ix), dtype=torch.long, device=agent.device)
            high_error = torch.zeros(len(ix), device=agent.device)
            chosen_q, chosen_target = qs[0], targets[0]
        adv, desired, ae, be, wrong = _decision_terms(chosen_q, chosen_target, route)
        route_loss = advantage_weight * ae + boundary_weight * be + sign_weight * wrong
        objective = torch.stack(branch_losses).mean(0) + route_weight * route_loss + .1 * high_error
        values = dict(objective=objective,
            branch_q_mse=torch.stack(q_errors).mean(0),
            branch_advantage_huber=torch.stack(advantage_errors).mean(0),
            route_advantage_huber=ae, route_boundary_loss=be,
            route_confidence_weighted_sign_error=wrong,
            route_pseudo_sign_agreement=((adv > 0) == (desired > 0)).float(),
            high_return_mse=high_error)
        for key, value in values.items():
            sums[key] += float(value.sum())
        count += len(ix)
    assert count > 0
    return {key: value / count for key, value in sums.items()}


def validation_objective(agent, p, teacher):
    return validation_metrics(agent, p, teacher)['objective']


def train_policy(train, val, teacher, seed, device, variant, out, epochs=None, warm_path=None):
    """Fresh, equal-budget fitting; the only pseudo-target source is teacher."""
    assert variant in CFG['policies'], 'Frozen references cannot be fitted by hierarchical'
    assert variant in ('U-DHRL', 'Paper-Raw', 'Flat-Q', 'Pseudo-MLP')
    assert warm_path is None, 'hierarchical uses fresh residual policies for every control'
    assert all(k in p for p in (train, val) for k in ('coarse_margin', 'fine_margin'))
    assert all(not any(k in p for k in ('labels', 'y', 'y_three', 'test')) for p in (train, val))
    best, chosen, history = float('inf'), None, []
    max_epochs = int(epochs or CFG['policy_epochs'])
    attempted_low, attempted_high = 0, 0
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def snapshot(agent, row, before):
        return dict(state={k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
                    gdim=train['global_state'].shape[1], edim=train['extra'].shape[1],
                    variant=variant, q_parameterization='paper_coarse_prior_plus_learned_residual',
                    **row, low_changed=fingerprint(agent.low) != before[0],
                    high_changed=fingerprint(agent.high) != before[1])

    for lr in CFG['learning_rates']:
        seed_all(seed)
        rng = np.random.default_rng(seed)
        agent = Agent(train['global_state'].shape[1], train['extra'].shape[1], variant, device)
        low_target = copy.deepcopy(agent.low).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(agent.low.parameters(), lr=lr)
        high_optimizer = torch.optim.Adam(agent.high.parameters(), lr=lr)
        replay, high_replay = Replay(rng), Replay(rng)
        updates, high_updates, stale, local_best = 0, 0, 0, float('inf')
        before = [fingerprint(agent.low), fingerprint(agent.high)]
        all_b = to_tensor(train, np.arange(len(train['score'])), device)
        initial_metrics = validation_metrics(agent, val, teacher)
        # Record the prior for auditing. A trained checkpoint, not epoch zero,
        # must be selected for the formal policy and update claims.
        history.append(dict(lr=lr, epoch=0, training_loss=None,
                            validation_objective=initial_metrics['objective'],
                            validation_metrics=initial_metrics, low_updates=0, high_updates=0,
                            eligible_checkpoint=False))
        for epoch in range(max_epochs):
            agent.train()
            order = rng.permutation(len(train['score']))
            epsilon = max(.05, 1 - epoch / max(1, .7 * max_epochs))
            losses = []
            for start in range(0, len(order), int(CFG['batch_size'])):
                ix = order[start:start + int(CFG['batch_size'])]
                indices = np.r_[ix, ix] if hierarchical(variant) else ix
                goals = (np.r_[np.zeros(len(ix), int), np.ones(len(ix), int)]
                         if hierarchical(variant) else np.ones(len(ix), int))
                b = subset(all_b, indices)
                goal = torch.as_tensor(goals, dtype=torch.long, device=device)
                rewards = teacher.rewards(teacher_inputs(train, indices), goals)
                with torch.no_grad():
                    actions = mask_q(agent.low(b, goal), goal).argmax(1).cpu().numpy()
                    explore = rng.random(len(indices)) < epsilon
                    for g in (0, 1):
                        use = explore & (goals == g)
                        actions[use] = rng.integers(2 if g == 0 else 3, size=int(use.sum()))
                replay.add(indices, goals, actions, rewards[np.arange(len(indices)), actions])
                if hierarchical(variant):
                    hi = np.r_[ix, ix]
                    hg = np.r_[np.zeros(len(ix), int), np.ones(len(ix), int)]
                    high_replay.add(hi, hg, teacher.external(teacher_inputs(train, hi), hg))
                if len(replay.rows) < 128:
                    continue
                rows = replay.sample()
                ii = rows[:, 0].astype(int)
                goals_sampled = rows[:, 1].astype(int)
                go = torch.as_tensor(goals_sampled, dtype=torch.long, device=device)
                actions_sampled = torch.as_tensor(rows[:, 2], dtype=torch.long, device=device)
                terminal_reward = torch.as_tensor(rows[:, 3], dtype=torch.float32, device=device)
                if variant == 'Pseudo-MLP':
                    # Equal data, LR/epoch budget and deployment losses. This
                    # control directly distills full rewards, without chosen-
                    # action TD or a hierarchical routing policy.
                    q = agent.low(subset(all_b, ix), torch.ones(len(ix), dtype=torch.long, device=device))
                    go = torch.ones(len(ix), dtype=torch.long, device=device)
                    rr = torch.as_tensor(teacher.rewards(teacher_inputs(train, ix), np.ones(len(ix), int)),
                                         dtype=q.dtype, device=device)
                    loss = F.smooth_l1_loss(q, rr)
                else:
                    q = agent.low(subset(all_b, ii), go)
                    rr = torch.as_tensor(teacher.rewards(teacher_inputs(train, ii), goals_sampled),
                                         dtype=q.dtype, device=device)
                    # This terminal decision never bootstraps another case.
                    loss = F.smooth_l1_loss(q.gather(1, actions_sampled[:, None]).squeeze(1), terminal_reward)
                    valid = torch.ones_like(q, dtype=torch.bool)
                    valid[go == 0, 2] = False
                    loss = loss + float(CFG['full_reward_weight']) * F.smooth_l1_loss(q[valid], rr[valid])
                loss = loss + _decision_training_loss(q, rr, go)
                assert bool(torch.isfinite(loss))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.low.parameters(), 5)
                optimizer.step()
                updates += 1
                losses.append(float(loss.detach()))
                if hierarchical(variant) and len(high_replay.rows) >= 128:
                    rows = high_replay.sample()
                    hi = rows[:, 0].astype(int)
                    hg = rows[:, 1].astype(int)
                    gg = torch.as_tensor(hg, dtype=torch.long, device=device)
                    external = torch.as_tensor(rows[:, 2], dtype=torch.float32, device=device)
                    hb = subset(all_b, hi)
                    with torch.no_grad():
                        # Double-Q: the online low policy chooses an action;
                        # its frozen target estimates that SAME case's return.
                        next_action = mask_q(agent.low(hb, gg), gg).argmax(1)
                        # Compare the same binary task for both information levels.
                        # Subtype penalties train low-Q, never create a routing tax.
                        tq = low_target(hb, gg)
                        target_gate = torch.stack((tq[:, 0], mask_q(tq, gg)[:, 1:].max(1).values), 1)
                        binary_action = (next_action > 0).long()
                        next_value = target_gate.gather(1, binary_action[:, None]).squeeze(1)
                        observed = torch.as_tensor(teacher.gate_rewards(teacher_inputs(train, hi)),
                                                   dtype=next_value.dtype, device=device)
                        actual = observed.gather(1, binary_action[:, None]).squeeze(1)
                        mix = float(CFG['high_return_mix'])
                        high_target = external + float(CFG['gamma']) * ((1 - mix) * next_value + mix * actual)
                    high_q = agent.high(hb['global_state']).gather(1, gg[:, None]).squeeze(1)
                    high_loss = F.smooth_l1_loss(high_q, high_target)
                    assert bool(torch.isfinite(high_loss))
                    high_optimizer.zero_grad()
                    high_loss.backward()
                    nn.utils.clip_grad_norm_(agent.high.parameters(), 5)
                    high_optimizer.step()
                    high_updates += 1
                if updates % int(CFG['target_every']) == 0:
                    low_target.load_state_dict(agent.low.state_dict())
            if (epoch + 1) % 5 and epoch + 1 != max_epochs:
                continue
            metrics = validation_metrics(agent, val, teacher)
            value = metrics['objective']
            row = dict(lr=lr, epoch=epoch + 1,
                       training_loss=float(np.mean(losses)) if losses else None,
                       validation_objective=value, validation_metrics=metrics,
                       low_updates=updates, high_updates=high_updates,
                       initial_validation_objective=initial_metrics['objective'])
            history.append(dict(row, eligible_checkpoint=True))
            print('POLICY_PROGRESS', variant, 'lr='+str(lr), 'epoch='+str(epoch+1),
                  'validation='+format(value, '.6g'), 'low_updates='+str(updates),
                  'high_updates='+str(high_updates), flush=True)
            if value < best and updates > 0:
                best, chosen = value, snapshot(agent, row, before)
            if value < local_best - 1e-6:
                local_best, stale = value, 0
            else:
                stale += 1
            write(out / (variant + '_history.json'), history)
            if stale >= int(CFG['policy_patience']):
                break
        attempted_low += updates
        attempted_high += high_updates
        del all_b
    assert chosen is not None and attempted_low > 0 and chosen['low_changed']
    if hierarchical(variant):
        assert attempted_high > 0 and chosen['high_changed']
    chosen.update(warm_source=None, validation_fallback_to_reference_weights=False,
                  attempted_low_updates=attempted_low, attempted_high_updates=attempted_high,
                  selection_basis='unlabeled shared reward, branch and deployed-route advantage loss; no test F1',
                  warm_offsets_reset=False, decision_boundary=0., offsets_fitted=False,
                  residual_prior='normalized coarse reconstruction margin only; no fine teacher output bypass',
                  auxiliary_targets='paper threshold-distance reward with anomaly-conditional subtype; no labels',
                  reward_mode=teacher.mode, reward_scale=teacher.reward_scale,
                  high_reward='paper external plus binary projected terminal internal reward')
    agent = Agent(chosen['gdim'], chosen['edim'], variant, device)
    load_policy_weights(agent, chosen)
    agent.eval()
    torch.save(chosen, out / (variant + '.pt'))
    return agent, {k: v for k, v in chosen.items() if k != 'state'}
