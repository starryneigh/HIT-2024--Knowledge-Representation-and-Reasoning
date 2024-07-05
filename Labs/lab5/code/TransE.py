import torch
from TransBase import TransBase

class TransE(TransBase):
    def distance(self, h_idx, r_idx, t_idx):
        h_embs = self.ent_embeddings(h_idx) # [batch, emb]
        r_embs = self.rel_embeddings(r_idx) # [batch, emb]
        t_embs = self.ent_embeddings(t_idx) # [batch, emb]
        scores = h_embs + r_embs - t_embs

        # norm 是计算 loss 时的正则化项
        norms = (torch.mean(h_embs.norm(p=self.norm, dim=1) - 1.0)
                 + torch.mean(r_embs ** 2) +
                 torch.mean(t_embs.norm(p=self.norm, dim=1) - 1.0)) / 3
        
        return scores.norm(p=self.norm, dim=1), norms