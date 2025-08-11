import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

##########################################
# Soft Graph over Triplets with Edge Type Embedding
##########################################
class SoftTripletGraph(nn.Module):
    def __init__(self, hidden_dim, sim_threshold=0.7):
        super(SoftTripletGraph, self).__init__()
        self.hidden_dim = hidden_dim
        self.sim_threshold = sim_threshold
        self.edge_type_embed = nn.Embedding(2, hidden_dim)  # 0: aspect, 1: opinion
        self.triplet_proj = nn.Linear(hidden_dim * 2 + 3, hidden_dim)

        self.attn_fc = nn.Linear(hidden_dim * 2 + hidden_dim, 1)  # for attention weight α_ij
        self.gat_proj = nn.Linear(hidden_dim + hidden_dim, hidden_dim)  # W_m
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embeddings, triplets_batch):
        B, L, H = embeddings.size()
        enhanced = embeddings.clone()

        for b_idx, triplets in enumerate(triplets_batch):
            triplet_nodes = []
            edge_list = []
            edge_types = []

            for idx, (asp_spans, opi_spans, sent_id) in enumerate(triplets):
                a_st, a_ed = asp_spans[0]
                o_st, o_ed = opi_spans[0]
                if a_ed < L and o_ed < L:
                    asp_repr = embeddings[b_idx, a_st:a_ed+1].mean(0)
                    opi_repr = embeddings[b_idx, o_st:o_ed+1].mean(0)
                    sent_vec = torch.zeros(3, device=embeddings.device)
                    sent_vec[sent_id - 2] = 1
                    node_feat = torch.cat([asp_repr, opi_repr, sent_vec], dim=-1)
                    triplet_nodes.append(self.triplet_proj(node_feat))

            N = len(triplet_nodes)
            if N == 0:
                continue
            node_feats = torch.stack(triplet_nodes, dim=0)

            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    asp_i, opi_i, _ = triplets[i]
                    asp_j, opi_j, _ = triplets[j]

                    sim = F.cosine_similarity(node_feats[i], node_feats[j], dim=0)
                    if sim > self.sim_threshold:
                        if asp_i[0] == asp_j[0]:
                            edge_list.append((i, j))
                            edge_types.append(0)
                        if opi_i[0] == opi_j[0]:
                            edge_list.append((i, j))
                            edge_types.append(1)

            if not edge_list:
                continue

            updated_feats = []
            for i in range(N):
                neigh_msgs = []
                attn_scores = []

                for (src, tgt), etype in zip(edge_list, edge_types):
                    if tgt == i:
                        edge_emb = self.edge_type_embed(torch.tensor(etype, device=embeddings.device))
                        concat_vec = torch.cat([node_feats[i], node_feats[src], edge_emb], dim=-1)
                        attn_score = self.attn_fc(self.leaky_relu(concat_vec))
                        msg = torch.cat([node_feats[src], edge_emb], dim=-1)
                        neigh_msgs.append(msg)
                        attn_scores.append(attn_score)

                if neigh_msgs:
                    attn_scores = torch.stack(attn_scores, dim=0).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=0)
                    neigh_msgs = torch.stack(neigh_msgs, dim=0)
                    agg = torch.sum(attn_weights.unsqueeze(-1) * neigh_msgs, dim=0)
                    updated = self.relu(self.gat_proj(agg))
                else:
                    updated = node_feats[i]
                updated_feats.append(updated)

            node_feats = torch.stack(updated_feats, dim=0)

            for i, (asp_spans, opi_spans, _) in enumerate(triplets):
                center = (asp_spans[0][0] + opi_spans[0][0]) // 2
                if center < L:
                    enhanced[b_idx, center] += node_feats[i]

        return enhanced


##########################################
# Final Model
##########################################
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.bert = RobertaModel.from_pretrained(args.model_name_or_path)
        self.norm0 = nn.LayerNorm(args.bert_feature_dim)

        self.triplet_graph = SoftTripletGraph(hidden_dim=args.bert_feature_dim)

        self.linear1 = nn.Linear(args.bert_feature_dim * 2, self.args.max_sequence_len)
        self.norm1 = nn.LayerNorm(self.args.max_sequence_len)
        self.cls_linear = nn.Linear(self.args.max_sequence_len, args.class_num)
        self.cls_linear1 = nn.Linear(self.args.max_sequence_len, 2)
        self.gelu = nn.GELU()

    def forward(self, tokens, masks, word_spans_batch=None, triplets_batch=None, return_word_emb=False):
        bert_feature, _ = self.bert(tokens, masks, return_dict=False)
        emb1 = self.norm0(bert_feature)

        if triplets_batch is not None:
            emb = self.triplet_graph(emb1, triplets_batch)
        else:
            emb = emb1
        #emb = emb1
        B, L, H = emb.size()
        feat_exp = emb.unsqueeze(2).expand(-1, -1, self.args.max_sequence_len, -1)
        feat_expT = feat_exp.transpose(1, 2)
        features = torch.cat([feat_exp, feat_expT], dim=3)

        hidden = self.linear1(features)
        hidden = self.norm1(hidden)
        hidden = self.gelu(hidden)

        logits = self.cls_linear(hidden)
        logits1 = self.cls_linear1(hidden)

        masks0 = masks.unsqueeze(3).expand(-1, -1, self.args.max_sequence_len, self.args.class_num)
        masks1 = masks.unsqueeze(3).expand(-1, -1, self.args.max_sequence_len, 2)
        logits = logits * masks0
        logits1 = logits1 * masks1

        if not return_word_emb:
            return logits, logits1
        else:
            word_emb_list = []
            for b_idx in range(B):
                sub_emb_b = emb[b_idx]
                if word_spans_batch is None:
                    word_emb_list.append(sub_emb_b)
                    continue
                spans_b = word_spans_batch[b_idx]
                word_vecs = []
                for (st, ed) in spans_b:
                    subpiece = sub_emb_b[st: ed + 1] if ed < sub_emb_b.size(0) else torch.zeros((1, H)).to(sub_emb_b.device)
                    pooled = subpiece.mean(dim=0)
                    word_vecs.append(pooled)
                word_emb_b = torch.stack(word_vecs, dim=0)
                word_emb_list.append(word_emb_b)
            return logits, logits1, word_emb_list

