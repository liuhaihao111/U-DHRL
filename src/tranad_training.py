"""Numerically stable training of the existing activity-only TranAD adapter.

The two decoders, self-conditioning, sigmoid output, epoch-weighted MSE, and
case-local windows are unchanged. This is a disclosed activity adaptation,
not an assertion of complete equivalence to the upstream time-series setup.
Only original training/validation sequences are accepted; no anomaly labels,
injected cases, test arrays, F1 scores or desired ranks enter this module.
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch

from common import CFG, seed_all, write
from native_models import TranAD


LEARNING_RATES = tuple(CFG['tranad_learning_rates'])
BATCH_SIZE = 64
PATIENCE = int(CFG['tranad_patience'])
HEALTH_RULES = {
    'near_zero_probability': 1e-7,
    'near_zero_fraction': .999,
    'constant_case_score_tolerance': 1e-12,
    'near_perfect_mse': 1e-8,
    'conditional_probability_rms_tolerance': 1e-7,
    'frequency_null_relative_tolerance': 1e-4,
}


def _frequency_prior(windows, vocab):
    counts = np.bincount(windows[:, -1], minlength=vocab).astype(np.float64)
    return counts / counts.sum()


def _initialize_output(model, frequency):
    # Sparse one-hot MSE otherwise initially pushes hundreds of sigmoid outputs
    # down together. Small nonzero weights keep both decoder gradient paths live.
    linear = model.output[0]
    with torch.no_grad():
        torch.nn.init.normal_(linear.weight, mean=0., std=1e-3)
        prior = torch.as_tensor(np.clip(frequency, 1e-4, 1 - 1e-4),
                                dtype=linear.bias.dtype, device=linear.bias.device)
        linear.bias.copy_(torch.logit(prior))


def _health_decision(health, distinct_cases):
    """Fixed training-health rules; zero F1 is never a rejection criterion."""
    reasons = []
    numeric = [v for v in health.values() if isinstance(v, (int, float))]
    if not all(np.isfinite(v) for v in numeric):
        reasons.append('nonfinite_validation_output')
    if (health['probability_near_zero_fraction'] >= HEALTH_RULES['near_zero_fraction']
            and health['observed_target_probability_mean'] <= HEALTH_RULES['near_zero_probability']):
        reasons.append('saturated_all_zero_reconstruction')
    if (distinct_cases > 1
            and health['case_score_range'] <= HEALTH_RULES['constant_case_score_tolerance']
            and health['validation_second_phase_mse'] > HEALTH_RULES['near_perfect_mse']):
        reasons.append('constant_nonperfect_case_scores')
    if (health['conditional_probability_rms'] <= HEALTH_RULES['conditional_probability_rms_tolerance']
            and health['frequency_null_relative_mse'] >= 1 - HEALTH_RULES['frequency_null_relative_tolerance']):
        reasons.append('constant_frequency_or_worse_reconstruction')
    return not reasons, reasons


@torch.no_grad()
def _validation_health(model, windows, owners, case_count, frequency, device):
    model.eval()
    scores, target_probability = [], []
    probability_sum = np.zeros(model.vocab, np.float64)
    probability_square_sum = np.zeros(model.vocab, np.float64)
    below, elements = 0, 0
    for start in range(0, len(windows), BATCH_SIZE):
        batch = torch.as_tensor(windows[start:start + BATCH_SIZE], device=device)
        _, second, target = model(batch)
        probability = second[0].double().cpu().numpy()
        score = (second - target).square().mean(2)[0].double().cpu().numpy()
        scores.append(score)
        target_probability.append(probability[np.arange(len(batch)), windows[start:start + BATCH_SIZE, -1]])
        probability_sum += probability.sum(0)
        probability_square_sum += np.square(probability).sum(0)
        below += int((probability <= HEALTH_RULES['near_zero_probability']).sum())
        elements += probability.size
    event_scores = np.concatenate(scores)
    case_scores = np.zeros(case_count, np.float64)
    np.maximum.at(case_scores, owners, event_scores)
    conditional_variance = np.maximum(
        probability_square_sum / len(windows) - np.square(probability_sum / len(windows)), 0.)
    # Unconditional training-frequency predictor is a diagnostic null model;
    # its loss is evaluated on the original held-out events, without labels.
    null = (np.square(frequency).sum() + 1 - 2 * frequency[windows[:, -1]]) / model.vocab
    value = float(event_scores.mean())
    return dict(validation_second_phase_mse=value,
                frequency_null_mse=float(null.mean()),
                frequency_null_relative_mse=value / max(float(null.mean()), 1e-30),
                all_zero_mse=1. / model.vocab,
                relative_distance_from_all_zero_mse=abs(value * model.vocab - 1),
                event_score_std=float(event_scores.std()),
                event_score_range=float(np.ptp(event_scores)),
                case_score_std=float(case_scores.std()),
                case_score_range=float(np.ptp(case_scores)),
                case_score_unique=int(len(np.unique(case_scores))),
                conditional_probability_rms=float(np.sqrt(conditional_variance.mean())),
                probability_near_zero_fraction=below / elements,
                observed_target_probability_mean=float(np.concatenate(target_probability).mean()),
                validation_windows=len(windows), validation_cases=case_count)


def train_tranad(train, val, vocab, seed, device, out, epochs=None):
    """Return (frozen_eval_model, info), saving a fresh out/TranAD.pt checkpoint.

    Both learning rates use identical train/validation sequences and seed.
    The lowest held-out second-phase MSE among healthy checkpoints wins.
    If both candidates collapse, stop explicitly instead of changing ties,
    inventing nonzero scores, reusing a failed model, or consulting test labels.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'TranAD.pt').exists():
        raise FileExistsError('Preserve existing TranAD.pt; choose a new development folder')
    train, val = np.asarray(train), np.asarray(val)
    if train.ndim != 2 or val.ndim != 2 or not len(train) or not len(val):
        raise ValueError('TranAD needs nonempty two-dimensional original case arrays')
    epoch_limit = int(CFG['tranad_epochs']) if epochs is None else int(epochs)
    if epoch_limit < 1:
        raise ValueError('epochs must be positive')
    seed_all(seed)
    window_model = TranAD(vocab, train.shape[1])
    train_windows = window_model.windows(torch.as_tensor(train))[0].numpy()
    val_windows, val_owners = window_model.windows(torch.as_tensor(val))
    val_windows, val_owners = val_windows.numpy(), val_owners.numpy()
    del window_model
    frequency = _frequency_prior(train_windows, vocab)
    distinct_cases = len({row.tobytes() for row in val})
    history, candidates, best_state = [], [], None
    best_loss, started = float('inf'), time.time()
    budget = dict(learning_rates=list(LEARNING_RATES), maximum_epochs_per_candidate=epoch_limit,
                  patience=PATIENCE, batch_size=BATCH_SIZE, optimizer='AdamW', weight_decay=1e-5,
                  scheduler='StepLR(step_size=5,gamma=0.9)', gradient_clip=5., dtype='float32',
                  output_initialization='weight Normal(0,0.001), bias logit(train event frequency clipped to [1e-4,1-1e-4])',
                  selection='minimum original-validation second-phase MSE among healthy checkpoints',
                  health_rules=HEALTH_RULES, training_cases=len(train), training_windows=len(train_windows),
                  validation_cases=len(val), validation_windows=len(val_windows), ground_truth_labels_used=False,
                  categorical_adapter=True, upstream_architecture_and_mse_unchanged=True)
    for lr in LEARNING_RATES:
        seed_all(seed)
        model = TranAD(vocab, train.shape[1]).to(device)
        _initialize_output(model, frequency)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=.9)
        rng = np.random.default_rng(seed)
        local_best, stale, healthy_epochs = float('inf'), 0, 0
        candidate = dict(lr=lr, best_healthy_validation_mse=None, best_epoch=None)
        for epoch in range(epoch_limit):
            model.train()
            order = rng.permutation(len(train_windows))
            total, count, maximum_gradient = 0., 0, 0.
            for start in range(0, len(order), BATCH_SIZE):
                batch = torch.as_tensor(train_windows[order[start:start + BATCH_SIZE]], device=device)
                opt.zero_grad(set_to_none=True)
                loss = model.objective(batch, epoch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite TranAD training loss')
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if not torch.isfinite(gradient):
                    raise FloatingPointError('Nonfinite TranAD training gradient')
                maximum_gradient = max(maximum_gradient, float(gradient))
                opt.step()
                total += float(loss.detach()) * len(batch)
                count += len(batch)
            health = _validation_health(model, val_windows, val_owners, len(val), frequency, device)
            healthy, reasons = _health_decision(health, distinct_cases)
            value = health['validation_second_phase_mse']
            row = dict(lr=lr, epoch=epoch + 1, effective_lr=opt.param_groups[0]['lr'],
                       train_loss=total / count, maximum_gradient_norm_before_clip=maximum_gradient,
                       healthy=healthy, rejection_reasons=reasons, **health)
            history.append(row)
            scheduler.step()
            if healthy:
                healthy_epochs += 1
                if value < local_best - max(abs(local_best) * 1e-6, 1e-12) or not np.isfinite(local_best):
                    local_best, stale = value, 0
                else:
                    stale += 1
                if candidate['best_healthy_validation_mse'] is None or value < candidate['best_healthy_validation_mse']:
                    candidate.update(best_healthy_validation_mse=value, best_epoch=epoch + 1, best_health=health)
                if value < best_loss:
                    best_loss = value
                    best_state = dict(state_dict=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                                      lr=lr, epoch=epoch + 1, validation_loss=value, health=health,
                                      model='TranAD', vocab=vocab, width=train.shape[1], budget=budget)
            else:
                stale += 1
            write(out / 'TranAD_history.json', history)
            print('TRANAD_TRAIN', seed, lr, epoch + 1, value, 'healthy' if healthy else ','.join(reasons), flush=True)
            if stale >= PATIENCE:
                break
        candidate.update(epochs_executed=epoch + 1, healthy_epochs=healthy_epochs,
                         final_health=health, final_rejection_reasons=reasons)
        candidates.append(candidate)
        del model, opt, scheduler
    info = dict(budget=budget, candidates=candidates, seconds=time.time() - started,
                all_candidates_failed=best_state is None)
    if best_state is None:
        write(out / 'TranAD_training.json', info)
        raise RuntimeError('Both TranAD candidates failed fixed numerical-health checks; inspect TranAD_training.json')
    info.update(selected_lr=best_state['lr'], selected_epoch=best_state['epoch'],
                selected_validation_mse=best_state['validation_loss'], selected_health=best_state['health'])
    best_state['training_info'] = info
    torch.save(best_state, out / 'TranAD.pt')
    write(out / 'TranAD_training.json', info)
    result = TranAD(vocab, train.shape[1]).to(device)
    result.load_state_dict(best_state['state_dict'])
    result.eval()
    result.requires_grad_(False)
    return result, info
