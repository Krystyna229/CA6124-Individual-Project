"""Reproduce the pre-model dataset diagnostics used in the CA6124 report."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ['user', 'item', 'rating', 'timestamp']


def read_dataset(root, dataset):
    if dataset == 'amazon-music':
        path = root / dataset / 'ratings_Digital_Music.csv'
        return pd.read_csv(path, names=COLUMNS), path
    if dataset == 'ml-1m':
        path = root / dataset / 'ratings.dat'
        return pd.read_csv(
            path, sep='::', names=COLUMNS, engine='python'), path
    raise ValueError(f'Unsupported diagnostic dataset: {dataset}')


def preprocess(df, method):
    """Mirror DaisyRec's deduplication and ui-level filter/core logic."""
    result = df.drop_duplicates(
        ['user', 'item'], keep='last', ignore_index=True).copy()
    threshold = int(''.join(character for character in method
                            if character.isdigit()))

    if method.endswith('filter'):
        user_counts = result.groupby('user').size()
        item_counts = result.groupby('item').size()
        keep = (result['user'].map(user_counts).ge(threshold)
                & result['item'].map(item_counts).ge(threshold))
        result = result[keep].copy()
    elif method.endswith('core'):
        while True:
            user_counts = result.groupby('user').size()
            item_counts = result.groupby('item').size()
            keep = (result['user'].map(user_counts).ge(threshold)
                    & result['item'].map(item_counts).ge(threshold))
            if keep.all():
                break
            result = result[keep].copy()
    else:
        raise ValueError('Expected an Nfilter or Ncore preprocessing method')

    return result.sort_values('timestamp', kind='mergesort').reset_index(
        drop=True)


def day_split(df, holdout=0.2):
    days = df['timestamp'].astype(np.int64).to_numpy() // 86400
    _, day_counts = np.unique(days, return_counts=True)
    cumulative = np.cumsum(day_counts)
    target = len(df) * (1 - holdout)
    cut = int(cumulative[:-1][np.argmin(
        np.abs(cumulative[:-1] - target))])
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def summarize(df, method):
    processed = preprocess(df, method)
    train, test = day_split(processed)
    warm_mask = (test['user'].isin(train['user'].unique())
                 & test['item'].isin(train['item'].unique()))
    warm = test[warm_mask]
    users = processed['user'].nunique()
    items = processed['item'].nunique()
    bytes_per_dense_matrix = items * items * 8

    return {
        'preprocessing': method,
        'interactions': int(len(processed)),
        'users': int(users),
        'items': int(items),
        'density': float(len(processed) / (users * items)),
        'minimum_user_degree_after_preprocessing': int(
            processed.groupby('user').size().min()),
        'minimum_item_degree_after_preprocessing': int(
            processed.groupby('item').size().min()),
        'train_interactions': int(len(train)),
        'test_interactions': int(len(test)),
        'actual_train_ratio': float(len(train) / len(processed)),
        'train_users': int(train['user'].nunique()),
        'train_items': int(train['item'].nunique()),
        'test_users_before_warm_filter': int(test['user'].nunique()),
        'warm_test_interactions': int(len(warm)),
        'warm_test_users': int(warm['user'].nunique()),
        'warm_test_items': int(warm['item'].nunique()),
        'warm_interaction_retention': float(len(warm) / len(test)),
        'warm_user_retention': float(
            warm['user'].nunique() / test['user'].nunique()),
        'last_train_day_utc': pd.to_datetime(
            train['timestamp'].max(), unit='s', utc=True).date().isoformat(),
        'first_test_day_utc': pd.to_datetime(
            test['timestamp'].min(), unit='s', utc=True).date().isoformat(),
        'ease_one_float64_item_matrix_gib': float(
            bytes_per_dense_matrix / 1024 ** 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    experiment_grid = {
        'ml-1m': ['10filter'],
        'amazon-music': ['10filter', '5filter', '5core'],
    }
    diagnostics = {}
    for dataset, methods in experiment_grid.items():
        frame, path = read_dataset(args.data_root, dataset)
        diagnostics[dataset] = {
            'source_path': str(path),
            'raw_interactions': int(len(frame)),
            'raw_users': int(frame['user'].nunique()),
            'raw_items': int(frame['item'].nunique()),
            'variants': [summarize(frame, method) for method in methods],
        }

    rendered = json.dumps(diagnostics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
