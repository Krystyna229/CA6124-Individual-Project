import logging
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from daisy.utils.evaluation import filter_warm_start, full_rank_predictions
from daisy.utils.parser import json_format_corrector
from daisy.utils.splitter import split_test, split_validation

try:
    import torch
    from daisy.model.CDAERecommender import CDAE
except ModuleNotFoundError:
    torch = None
    CDAE = None


class AssignmentExtensionTests(unittest.TestCase):
    def test_tune_pack_parser_preserves_underscored_keys(self):
        import json

        parsed = json.loads(json_format_corrector(
            '{reg_1: [0.0], latent_dim: [16, 32]}'))

        self.assertEqual(parsed, {
            'reg_1': [0.0],
            'latent_dim': [16, 32],
        })

    def test_original_interaction_timestamp_split_is_unchanged(self):
        df = pd.DataFrame({
            'user': np.arange(10),
            'timestamp': np.arange(10),
        })

        train_ids, test_ids = split_test(
            df, test_method='tsbr', test_size=0.2)

        np.testing.assert_array_equal(train_ids, np.arange(8))
        np.testing.assert_array_equal(test_ids, np.arange(8, 10))

    def test_day_boundary_split_keeps_equal_days_together(self):
        seconds_per_day = 86400
        df = pd.DataFrame({
            'user': np.arange(10),
            'timestamp': [
                *([seconds_per_day] * 4),
                *([2 * seconds_per_day] * 3),
                *([3 * seconds_per_day] * 3),
            ],
        })

        pairs = list(split_validation(
            df, val_method='tsbr', val_size=0.2, split_boundary='day'))
        train_ids, val_ids = pairs[0]

        np.testing.assert_array_equal(train_ids, np.arange(7))
        np.testing.assert_array_equal(val_ids, np.arange(7, 10))
        self.assertTrue(set(df.iloc[train_ids]['timestamp']).isdisjoint(
            set(df.iloc[val_ids]['timestamp'])))

    def test_warm_start_filter_reports_cold_interactions(self):
        train = pd.DataFrame({'user': [0, 1], 'item': [0, 1]})
        evaluate = pd.DataFrame({
            'user': [0, 2, 1, 2],
            'item': [1, 1, 3, 3],
        })

        filtered, stats = filter_warm_start(evaluate, train)

        self.assertEqual(
            filtered[['user', 'item']].values.tolist(), [[0, 1]])
        self.assertEqual(stats, {
            'input_interactions': 4,
            'warm_interactions': 1,
            'cold_user_interactions': 2,
            'cold_item_interactions': 2,
            'input_users': 3,
            'warm_users': 1,
        })

    def test_full_ranking_uses_only_train_items_and_masks_history(self):
        class DummyEASE:
            interaction_matrix = sp.csr_matrix(
                [[1.0, 0.0, 0.0, 0.0, 0.0]])
            item_similarity = np.array([
                [0.1, 0.9, 0.8, 0.2, 10.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ])

        predictions = full_rank_predictions(
            DummyEASE(), [0], {0: {1}}, [0, 1, 2, 3],
            {'algo_name': 'ease', 'item_num': 5, 'topk': 2})

        np.testing.assert_array_equal(predictions, np.array([[2, 3]]))

    @unittest.skipIf(torch is None, 'PyTorch is not installed')
    def test_cdae_eval_is_deterministic_and_has_expected_shape(self):
        config = {
            'gpu': '0',
            'logger': logging.getLogger('cdae-test'),
            'epochs': 1,
            'lr': 0.001,
            'dropout': 0.9,
            'reg_1': 0.0,
            'reg_2': 0.0,
            'user_num': 2,
            'item_num': 4,
            'latent_dim': 3,
            'history_item_id': torch.tensor([[0, 1], [2, 0]]),
            'history_item_value': torch.tensor(
                [[1.0, 1.0], [1.0, 0.0]]),
            'optimizer': 'default',
            'init_method': 'default',
            'early_stop': False,
            'topk': 2,
        }
        model = CDAE(config)
        users = torch.tensor([0, 1], device=model.device)
        ratings = model.get_user_rating_matrix(users)

        model.eval()
        with torch.no_grad():
            first = model.forward(users, ratings)
            second = model.forward(users, ratings)

        self.assertEqual(first.shape, (2, 4))
        self.assertTrue(torch.equal(first, second))


if __name__ == '__main__':
    unittest.main()
