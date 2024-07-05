import torch
import torch.nn as nn
import torch.nn.functional as F
from TransBase import TransBase

class TransH(TransBase):
    def __init__(self, ent_num, rel_num, device, dim=100, norm=1, margin=2.0, alpha=0.01):
        super(TransH, self).__init__(ent_num, rel_num, device, dim, norm, margin, alpha)
        self.epsilon = 1e-5
        
        # 关系的法向量
        self.norm_vectors = nn.Embedding(self.rel_num, self.dim)
        torch.nn.init.xavier_uniform_(self.norm_vectors.weight.data)
        self.norm_vectors.weight.data = F.normalize(self.norm_vectors.weight.data, 2, 1)
    
    def project_to_hyperplane(self, ent_embeddings, norm_vectors):
        return ent_embeddings - torch.sum(ent_embeddings * norm_vectors, dim=1, keepdim=True) * norm_vectors

    def distance(self, h_idx, r_idx, t_idx):
        h_embs = self.ent_embeddings(h_idx) # [batch, emb]
        r_embs = self.rel_embeddings(r_idx) # [batch, emb]
        t_embs = self.ent_embeddings(t_idx) # [batch, emb]
        norm_vectors = self.norm_vectors(r_idx) # [batch, emb]

        # 投影到超平面
        h_proj = self.project_to_hyperplane(h_embs, norm_vectors)
        t_proj = self.project_to_hyperplane(t_embs, norm_vectors)

        scores = h_proj + r_embs - t_proj

        # norm 是计算 loss 时的正则化项
        ety_norms = (torch.mean(abs(h_proj.norm(p=2, dim=1) ** 2 - 1.0))
                     + torch.mean(abs(t_proj.norm(p=2, dim=1) ** 2 - 1.0)))
        temp = torch.sum(r_embs * norm_vectors, dim=1) ** 2 / norm_vectors.norm(p=2, dim=1) ** 2 - self.epsilon
        rel_norms = torch.mean(abs(temp))
        # norms = (torch.mean(h_embs.norm(p=2, dim=1) ** 2 - 1.0)
        #          + torch.mean(r_embs ** 2) +
        #          torch.mean(t_embs.norm(p=2, dim=1) ** 2 - 1.0)) / 3
        # norms = (ety_norms + rel_norms) / 3

        norms = torch.mean(h_embs ** 2 - 1.0) + torch.mean(r_embs ** 2) + torch.mean(t_embs ** 2 - 1.0) + torch.mean(norm_vectors ** 2)
        norms = norms / 4
        
        return scores.norm(p=self.norm, dim=1), norms
    