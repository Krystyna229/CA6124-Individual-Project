"""Collaborative Denoising Auto-Encoder for top-N recommendation.

Reference: Wu et al., Collaborative Denoising Auto-Encoders for Top-N
Recommender Systems, WSDM 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from daisy.model.AbstractRecommender import AERecommender


class CDAE(AERecommender):
    """One-hidden-layer collaborative denoising auto-encoder.

    DaisyRec already supplies the user interaction history through
    :class:`AERecommender`. CDAE adds a learned user embedding to the encoded
    corrupted interaction vector, then reconstructs all items.
    """

    tunable_param_names = [
        'latent_dim', 'dropout', 'batch_size', 'lr', 'reg_1', 'reg_2'
    ]

    def __init__(self, config):
        super(CDAE, self).__init__(config)

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.dropout = config['dropout']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.latent_dim = config['latent_dim']

        self.history_item_id = config['history_item_id'].to(self.device)
        self.history_item_value = config['history_item_value'].to(self.device)

        self.encoder = nn.Linear(self.item_num, self.latent_dim)
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.item_num)

        self.optimizer = (
            config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        )
        self.initializer = (
            config['init_method']
            if config['init_method'] != 'default'
            else 'xavier_normal'
        )
        self.early_stop = config['early_stop']
        self.topk = config['topk']

        self.apply(self._init_weight)

    def forward(self, users, rating_matrix):
        corrupted = F.dropout(
            rating_matrix, p=self.dropout, training=self.training
        )
        hidden = F.relu(self.encoder(corrupted) + self.user_embedding(users))
        return self.decoder(hidden)

    def calc_loss(self, batch):
        users = batch.to(self.device).long()
        rating_matrix = self.get_user_rating_matrix(users)
        logits = self.forward(users, rating_matrix)
        reconstruction = F.binary_cross_entropy_with_logits(
            logits, rating_matrix, reduction='sum'
        )

        parameters = (
            self.encoder.weight,
            self.user_embedding.weight,
            self.decoder.weight,
        )
        l1_penalty = sum(param.abs().sum() for param in parameters)
        l2_penalty = sum(param.pow(2).sum() for param in parameters)
        return reconstruction + self.reg_1 * l1_penalty + self.reg_2 * l2_penalty

    def predict(self, u, i):
        self.eval()
        with torch.no_grad():
            users = torch.tensor([u], device=self.device)
            rating_matrix = self.get_user_rating_matrix(users)
            scores = self.forward(users, rating_matrix)
        return scores[0, i].cpu().item()

    def rank(self, test_loader):
        self.eval()
        rec_ids = []
        with torch.no_grad():
            for users, candidate_ids in test_loader:
                users = users.to(self.device).long()
                candidate_ids = candidate_ids.to(self.device).long()
                rating_matrix = self.get_user_rating_matrix(users)
                scores = self.forward(users, rating_matrix)
                scores = torch.gather(scores, 1, candidate_ids)
                order = torch.argsort(scores, descending=True)
                ranked = torch.gather(candidate_ids, 1, order)[:, :self.topk]
                rec_ids.append(ranked.cpu())
        return torch.cat(rec_ids, dim=0).numpy()

    def full_rank(self, u):
        self.eval()
        with torch.no_grad():
            users = torch.tensor([u], device=self.device)
            rating_matrix = self.get_user_rating_matrix(users)
            scores = self.forward(users, rating_matrix).view(-1)
        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()
