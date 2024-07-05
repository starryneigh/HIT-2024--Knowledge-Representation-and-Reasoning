import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class TransBase(nn.Module):
    def __init__(self, ent_num, rel_num, device, dim=100, norm=1, margin=2.0, alpha=0.01):
        super(TransBase, self).__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.dim = dim
        self.norm = norm
        self.margin = margin
        self.alpha = alpha

        # 初始化实体和关系表示向量
        self.ent_embeddings = nn.Embedding(self.ent_num, self.dim)
        torch.nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, 1)

        self.rel_embeddings = nn.Embedding(self.rel_num, self.dim)
        torch.nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, 1)
        
        # 损失函数
        self.criterion = nn.MarginRankingLoss(margin=self.margin)

    def get_ent_resps(self, ent_idx): #[batch]
        return self.ent_embeddings(ent_idx) # [batch, emb]

    def distance(self, h_idx, r_idx, t_idx):
        raise NotImplementedError("This method should be overridden by subclasses")

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.float, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)
    
    def negative_sampling(self, pos_triples, num_negatives=1):
        neg_triples = []
        for h, r, t in pos_triples:
            for _ in range(num_negatives):
                if torch.rand(1) > 0.5:
                    # Replace head
                    neg_h = np.random.randint(0, self.ent_num)
                    neg_t = t
                else:
                    # Replace tail
                    neg_h = h
                    neg_t = np.random.randint(0, self.ent_num)
                # Append negative triple
                neg_triples.append((neg_h, r, neg_t))
        return torch.tensor(neg_triples, dtype=torch.long, device=self.device)
    
    def forward(self, pos_triples):
        # Negative sampling
        neg_triples = self.negative_sampling(pos_triples)
        # Unpack positive triples
        pos_h_idx, pos_r_idx, pos_t_idx = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        # Unpack negative triples
        neg_h_idx, neg_r_idx, neg_t_idx = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        
        # Calculate distances for positive triples
        pos_distances, pos_norms = self.distance(pos_h_idx, pos_r_idx, pos_t_idx)
        # Calculate distances for negative triples
        neg_distances, neg_norms = self.distance(neg_h_idx, neg_r_idx, neg_t_idx)
        
        # Compute the loss
        loss_value = self.loss(pos_distances, neg_distances)
        
        # Regularize
        reg_term = self.alpha * (pos_norms + neg_norms)
        # reg_term = 0
        
        # Total loss
        total_loss = loss_value + reg_term
        
        return total_loss
    
    def evaluate(self, test_triples, k=10):
        hits = 0
        total = len(test_triples)
        test_bar = tqdm(test_triples)
        ranks = []
        
        for h, r, t in test_bar:
            h_idx = torch.tensor([h], dtype=torch.long, device=self.device)
            r_idx = torch.tensor([r], dtype=torch.long, device=self.device)
            t_idx = torch.tensor([t], dtype=torch.long, device=self.device)
            
            # Positive score
            pos_score, _ = self.distance(h_idx, r_idx, t_idx)
            
            # Replace tail
            all_t_idx = torch.arange(self.ent_num, dtype=torch.long, device=self.device)
            all_scores, _ = self.distance(h_idx.expand(self.ent_num), r_idx.expand(self.ent_num), all_t_idx)
            
            # Ranking
            sorted_indices = torch.argsort(all_scores)
            rank = (sorted_indices == t).nonzero().item()
            ranks.append(rank)
            
            if rank < k:
                hits += 1
        
        return hits / total, sum(ranks) / total