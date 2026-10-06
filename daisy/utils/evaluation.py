"""Optional rigorous evaluation extensions for the CA6124 project.

All functions in this module are opt-in. DaisyRec's original candidate-based
evaluation remains the default path.
"""

import numpy as np


def filter_warm_start(eval_set, train_set, uid='user', iid='item'):
    """Keep only interactions whose user and item occur in the training set."""
    train_users = set(train_set[uid].unique())
    train_items = set(train_set[iid].unique())
    user_mask = eval_set[uid].isin(train_users)
    item_mask = eval_set[iid].isin(train_items)
    filtered = eval_set[user_mask & item_mask].copy().reset_index(drop=True)

    stats = {
        'input_interactions': int(len(eval_set)),
        'warm_interactions': int(len(filtered)),
        'cold_user_interactions': int((~user_mask).sum()),
        'cold_item_interactions': int((~item_mask).sum()),
        'input_users': int(eval_set[uid].nunique()),
        'warm_users': int(filtered[uid].nunique()),
    }
    return filtered, stats


def log_warm_start_stats(logger, split_name, stats):
    """Log enough information to audit the warm-start evaluation cohort."""
    logger.info(
        'Warm-start %s: %d/%d interactions and %d/%d users retained; '
        '%d cold-user and %d cold-item interactions removed',
        split_name,
        stats['warm_interactions'], stats['input_interactions'],
        stats['warm_users'], stats['input_users'],
        stats['cold_user_interactions'], stats['cold_item_interactions'])


def _score_all_items(model, users, algo_name):
    """Return a dense user-by-item score matrix without changing baselines."""
    algo_name = algo_name.lower()
    if algo_name == 'ease':
        scores = model.interaction_matrix[users, :] @ model.item_similarity
        return np.asarray(scores)

    import torch

    user_tensor = torch.as_tensor(users, device=model.device, dtype=torch.long)
    with torch.no_grad():
        rating_matrix = model.get_user_rating_matrix(user_tensor)
        if algo_name == 'multi-vae':
            scores, _, _ = model.forward(rating_matrix)
        elif algo_name == 'cdae':
            scores = model.forward(user_tensor, rating_matrix)
        else:
            raise NotImplementedError(
                f'Full-ranking extension is not implemented for {algo_name}'
            )
    return scores.detach().cpu().numpy()


def full_rank_predictions(model, users, train_ur, train_items, config,
                          batch_size=128):
    """Rank every train-seen item while excluding each user's train history."""
    if hasattr(model, 'eval'):
        model.eval()

    users = list(users)
    train_items = np.asarray(sorted(set(train_items)), dtype=np.int64)
    item_num = int(config['item_num'])
    topk = int(config['topk'])
    allowed = np.zeros(item_num, dtype=bool)
    allowed[train_items] = True

    predictions = []
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        scores = _score_all_items(model, batch_users, config['algo_name'])
        scores[:, ~allowed] = -np.inf
        for row, user in enumerate(batch_users):
            seen = list(train_ur[user])
            if seen:
                scores[row, seen] = -np.inf
            available = int(np.isfinite(scores[row]).sum())
            if available < topk:
                raise ValueError(
                    f'User {user} has only {available} eligible full-ranking '
                    f'items, fewer than topk={topk}')
            ranked = np.argsort(-scores[row], kind='mergesort')[:topk]
            predictions.append(ranked)

    return np.asarray(predictions, dtype=np.int64)


def evaluation_users(eval_ur):
    """Return a deterministic user order for metric calculation."""
    return sorted(eval_ur.keys())


def evaluation_suffix(config):
    """Distinguish opt-in experiment artifacts from original DaisyRec output."""
    parts = []
    if config.get('split_boundary', 'interaction') != 'interaction':
        parts.append(config['split_boundary'])
    if config.get('warm_start', False):
        parts.append('warm')
    if config.get('ranking_mode', 'sampled') != 'sampled':
        parts.append(config['ranking_mode'])
    return '' if not parts else '_' + '_'.join(parts)
