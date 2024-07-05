import torch
import torch.nn as nn
from TransBase import TransBase

class TransR(TransBase):
    def __init__(self, ent_num, rel_num, device, dim=100, norm=1, margin=2.0, alpha=0.01):
        super(TransR, self).__init__(ent_num, rel_num, device, dim, norm, margin, alpha)
        
        # 关系的投影矩阵
        self.proj_matrix = nn.Embedding(self.rel_num, self.dim * self.dim)
        torch.nn.init.xavier_uniform_(self.proj_matrix.weight.data)
    
    def project(self, ent_embs, proj_matrix):
        ent_embs = ent_embs.view(-1, self.dim, 1)
        proj_matrix = proj_matrix.view(-1, self.dim, self.dim)
        proj_ent_embs = torch.bmm(proj_matrix, ent_embs).view(-1, self.dim)
        return proj_ent_embs

    def distance(self, h_idx, r_idx, t_idx):
        h_embs = self.ent_embeddings(h_idx) # [batch, emb]
        r_embs = self.rel_embeddings(r_idx) # [batch, emb]
        t_embs = self.ent_embeddings(t_idx) # [batch, emb]
        proj_matrix = self.proj_matrix(r_idx) # [batch, dim*dim]

        # 投影到关系特定空间
        h_proj = self.project(h_embs, proj_matrix)
        t_proj = self.project(t_embs, proj_matrix)

        scores = h_proj + r_embs - t_proj

        # norm 是计算 loss 时的正则化项
        # norms = (torch.mean(h_embs.norm(p=self.norm, dim=1) - 1.0)
        #          + torch.mean(r_embs ** 2) +
        #          torch.mean(t_embs.norm(p=self.norm, dim=1) - 1.0)) / 3
        norms = torch.mean(h_embs ** 2 - 1.0) + torch.mean(r_embs ** 2) + torch.mean(t_embs ** 2 - 1.0) + torch.mean(proj_matrix ** 2)
        
        return scores.norm(p=self.norm, dim=1), norms
