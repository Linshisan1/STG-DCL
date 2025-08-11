import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils.common_utils import stop_words
from tools.evaluate import evaluate
import torch.nn as nn
from tqdm import trange
from utils.plot_utils import gather_features, plot_pca, plot_pca_3d
import copy
import torch.nn.functional as F
from collections import deque

import os


def compute_dep_distance(head_list, L):
    """
    根据 head_list（长度为 L 的列表，head_list[i] 为 token i 的父节点索引）构造无向依存图，
    并通过 BFS 计算每个 token 对之间的最短依存距离。返回形状 [L, L] 的 tensor，
    距离不可达的设为 +inf。
    """
    import math
    # 初始化距离矩阵，自己到自己距离为 0，其它为 +inf
    dist = [[math.inf] * L for _ in range(L)]
    for i in range(L):
        dist[i][i] = 0
    # 构建无向图（假设 head_list 中的值有效且 root 的 head 通常为 0 或 -1，根据数据情况调整）
    adj = [[] for _ in range(L)]
    for i in range(L):
        h = head_list[i]
        # 此处假设有效 head 范围在 [0, L)，且不与自己相同
        if h >= 0 and h < L and h != i:
            adj[i].append(h)
            adj[h].append(i)
    # 对每个 token，执行 BFS 计算最短距离
    for i in range(L):
        from collections import deque
        q = deque([i])
        visited = {i: 0}
        while q:
            cur = q.popleft()
            cur_dist = visited[cur]
            for neighbor in adj[cur]:
                if neighbor not in visited:
                    visited[neighbor] = cur_dist + 1
                    q.append(neighbor)
        for j in range(L):
            if j in visited:
                dist[i][j] = visited[j]
    return torch.tensor(dist, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float)
# ------------------ 结束依存距离计算辅助函数 ------------------



class Trainer():
    def __init__(
            self, model, trainset, devset, testset, optimizer, criterion, lr_scheduler,
            args, logging, beta_1, beta_2, bear_max, last, plot=False
    ):
        """
        This version does cross-sentence SCL + Syntax adjacency InfoNCE.

        Requirements:
         - roberta.py forward => (logits, logits1, embeddings)
           where embeddings => [B, L, hidden_dim]
         - data_preparing => each Instance has token_classes, head, etc.
         - get_batch => returns postag_list, head_list, token_classes_list, etc. for the entire batch.

        Additional hyperparams in args:
         - args.lambda_scl (float)   # scale for supervised contrast
         - args.lambda_syntax (float)# scale for syntax adjacency contrast
         - args.temp (float)         # temperature for InfoNCE
        """

        self.model = model
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.optimizer = optimizer

        # (f_loss, f_loss1) => multi-class, binary heads
        self.f_loss = criterion[0]
        self.f_loss1 = criterion[1]

        self.lr_scheduler = lr_scheduler
        self.best_joint_f1 = 0
        self.best_joint_f1_test = 0
        self.best_joint_epoch = 0
        self.best_joint_epoch_test = 0

        self.writer = SummaryWriter()
        self.args = args
        self.logging = logging

        self.evaluate = evaluate
        self.stop_words = stop_words

        self.beta_1 = beta_1  # second head loss scale
        self.beta_2 = beta_2  # old or free to reuse
        self.plot = plot
        self.bear_max = bear_max
        self.last = last
        self.contrastive = True  # old code, can remove if you wish
        #self.use_scl = args.use_scl
        #self.use_syntaxcl = args.use_syntaxcl
        # self.lambda_scl = args.lambda_scl
        # self.lambda_syntax = args.lambda_syntax


    def train(self):
        bear = 0
        last = self.last
        for i in range(self.args.epochs):

            if self.plot:
                if i % 10 == 0:
                    model_copy = copy.deepcopy(self.model)
                    g0, g1, g2, g3, g4 = gather_features(model_copy, self.testset)
                    plot_pca(g0, g1, g2, g3, g4, i)
                    plot_pca_3d(g0, g1, g2, g3, g4, i)

            self.logging(f"\n\nEpoch: {i + 1}")
            self.logging(f"contrastive: {self.contrastive} | bear/max: {bear}/{self.bear_max} | last: {last}")

            epoch_sum_loss = []

            for j in trange(self.trainset.batch_count):
                self.model.train()
                batch_data = self.trainset.get_batch(j)
                (
                    sentence_ids,
                    bert_tokens,
                    masks,
                    word_spans,
                    tagging_matrices,
                    tokenized,
                    token_classes_list,
                    postag_list,
                    head_list,
                    deprel_list,
                    triplets_in_spans_list
                ) = batch_data

                logits, logits1, word_emb_list = self.model(
                    bert_tokens, masks,
                    word_spans_batch=word_spans,
                    triplets_batch=triplets_in_spans_list,
                    return_word_emb=True
                )

                logits_flat = logits.reshape(-1, logits.shape[-1])
                gold_tag_flat = tagging_matrices.reshape(-1)
                loss0 = self.f_loss(logits_flat, gold_tag_flat)

                tags1 = tagging_matrices.clone()
                tags1[tags1 > 0] = 1
                logits1_flat = logits1.reshape(-1, logits1.shape[-1])
                tags1_flat = tags1.reshape(-1).to(self.args.device)
                loss1 = self.f_loss1(logits1_flat.float(), tags1_flat)


                all_emb, all_label, batch_lens = [], [], []
                for emb_b, lab_b in zip(word_emb_list, token_classes_list):
                    all_emb.append(emb_b)
                    all_label.extend(lab_b)
                    batch_lens.append(emb_b.size(0))

                # 拼接整个 batch 的表示，用于 SCL
                all_emb_cat = torch.cat(all_emb, dim=0)  # [N, H]
                all_emb_cat = F.normalize(all_emb_cat, dim=-1)
                all_label_t = torch.tensor(all_label, device=self.args.device)

                # ========== SCL 构造：整个 batch 拼接后构造正负样本 ==========

                if self.args.use_scl:
                    scl_loss = self.build_scl_loss_batch(all_emb_cat, all_label_t, temp=self.args.temp)
                else:
                    scl_loss = torch.tensor(0.0, device=self.args.device)
                # ========== SyntaxCL 构造：依句子构建依存邻接矩阵并计算 ==========
                syntax_loss_list = []
                if self.args.use_syntaxcl:
                    for b_idx, emb_b in enumerate(all_emb):
                        head_b = head_list[b_idx]
                        L_b = emb_b.size(0)

                        # 依存图距离矩阵
                        dep_dist = compute_dep_distance(head_b, L_b)
                        max_distance = getattr(self.args, 'dep_threshold')
                        alpha_dep = getattr(self.args, 'alpha_dep')

                        # Soft edge weights: 权重越远越小，阈值外置0
                        weight_matrix = torch.exp(-alpha_dep * dep_dist)
                        weight_matrix[dep_dist > max_distance] = 0.0

                        emb_norm = F.normalize(emb_b, dim=-1)
                        loss_b = self.build_syntax_loss_batch(emb_norm, weight_matrix, temp=self.args.temp)
                        syntax_loss_list.append(loss_b)

                    syn_loss = torch.stack(syntax_loss_list).mean()
                else:
                    syn_loss = torch.tensor(0.0, device=self.args.device)

                loss_cl = self.args.lambda_scl * scl_loss + self.args.lambda_syntax * syn_loss

                loss = loss0 + self.beta_1 * loss1 + loss_cl
                epoch_sum_loss.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step_id = i * self.trainset.batch_count + j + 1
                self.writer.add_scalar('train/loss', loss, step_id)
                self.writer.add_scalar('train/loss0', loss0, step_id)
                self.writer.add_scalar('train/loss1', loss1, step_id)
                self.writer.add_scalar('train/scl_loss', scl_loss.item(), step_id)
                self.writer.add_scalar('train/syn_loss', syn_loss, step_id)
                for idx_pg, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f'lr/group{idx_pg}', param_group['lr'], step_id)
                self.writer.flush()
            epoch_avg_loss = sum(epoch_sum_loss) / len(epoch_sum_loss)
            self.logging(
                f"Epoch:{i + 1}, Step:{j + 1}, Loss:{loss.item():.6f}, Loss0:{loss0.item():.6f}, Loss1:{loss1.item():.6f}, SCL:{scl_loss.item():.6f}, SyntaxCL:{syn_loss.item():.6f}")
            self.logging(f"Epoch average loss: {epoch_avg_loss:.6f}")

            # do evaluate
            joint_precision, joint_recall, joint_f1 = self.evaluate(self.model, self.devset, self.stop_words,
                                                                    self.logging, self.args)
            joint_precision_test, joint_recall_test, joint_f1_test = self.evaluate(self.model, self.testset,
                                                                                   self.stop_words, self.logging,
                                                                                   self.args)

            if joint_f1 > self.best_joint_f1:
                self.best_joint_f1 = joint_f1
                self.best_joint_epoch = i

            if joint_f1_test > self.best_joint_f1_test:
                self.best_joint_f1_test = joint_f1_test
                self.best_joint_epoch_test = i

            self.writer.add_scalar('dev/f1', joint_f1, i + 1)
            self.writer.add_scalar('test/f1', joint_f1_test, i + 1)
            self.writer.add_scalar('dev/precision', joint_precision, i + 1)
            self.writer.add_scalar('test/precision', joint_precision_test, i + 1)
            self.writer.add_scalar('dev/recall', joint_recall, i + 1)
            self.writer.add_scalar('test/recall', joint_recall_test, i + 1)
            self.writer.add_scalar('best/dev_f1', self.best_joint_f1, i + 1)
            self.writer.add_scalar('best/test_f1', self.best_joint_f1_test, i + 1)

            self.lr_scheduler.step()

            self.logging(
                f"best epoch:{self.best_joint_epoch + 1}\t best dev {self.args.task} f1:{self.best_joint_f1:.5f}")
            self.logging(
                f"best epoch:{self.best_joint_epoch_test + 1}\tbest test {self.args.task} f1:{self.best_joint_f1_test:.5f}")

        self.writer.close()


    def build_scl_loss_batch(self, all_emb_cat, all_label_t, temp=0.07):

        device = all_emb_cat.device
        valid_mask = (all_label_t >= 2) & (all_label_t <= 4)
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        valid_indices = valid_indices[valid_indices < all_emb_cat.size(0)]
        if valid_indices.numel() < 2:
            return torch.tensor(0.0, device=device)

        sub_emb = all_emb_cat[valid_indices].float().contiguous()  # [K, hidden_dim]
        #sub_lab = all_label_t[valid_indices] - 2
        sub_lab = all_label_t[valid_indices]

        #sub_emb = self.scl_projection_head(sub_emb)  # shape: [K, proj_dim]
        sim_matrix = torch.matmul(sub_emb, sub_emb.t()) / temp  # [K, K]

        # 超参：hard positive 和 hard negative 数量
        num_hard_positive = getattr(self.args, 'num_hard_positive')
        num_hard_negative = getattr(self.args, 'num_hard_negative')

        K = sub_emb.size(0)
        loss_list = []
        for i in range(K):
            sim_i = sim_matrix[i]  # [K]
            # 正样本mask: 同一情感且排除自己
            pos_mask = ((sub_lab == sub_lab[i]) & (torch.arange(K, device=device) != i))
            neg_mask = (sub_lab != sub_lab[i])
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            if pos_indices.numel() == 0 or neg_indices.numel() == 0:
                continue
            pos_sims = sim_i[pos_indices]
            neg_sims = sim_i[neg_indices]

            if pos_sims.numel() > num_hard_positive:
                _, pos_sorted_idx = torch.sort(pos_sims, descending=False)
                pos_indices = pos_indices[pos_sorted_idx[:num_hard_positive]]

            if neg_sims.numel() > num_hard_negative:
                _, neg_sorted_idx = torch.sort(neg_sims, descending=True)
                neg_indices = neg_indices[neg_sorted_idx[:num_hard_negative]]
            exp_sim = torch.exp(sim_i)
            pos_sum = torch.sum(exp_sim[pos_indices])
            neg_sum = torch.sum(exp_sim[neg_indices])
            denominator = pos_sum + neg_sum + 1e-8
            loss_i = -torch.log(pos_sum / denominator + 1e-8)
            loss_list.append(loss_i)

        if len(loss_list) == 0:
            return torch.tensor(0.0, device=device)
        sup_loss = torch.stack(loss_list).mean()

        return sup_loss

    # =================== build_syntax_loss_batch 保持不变 ===================
    def build_syntax_loss_batch(self, all_emb_cat, adjacency, temp=0.07):
        device = all_emb_cat.device

        N = all_emb_cat.size(0)
        if N < 2:
            return torch.tensor(0.0, device=device)

        sim_matrix = torch.matmul(all_emb_cat, all_emb_cat.t()) / temp
        sim_matrix = sim_matrix - 1e9 * torch.eye(N, device=device)
        exp_sim = torch.exp(sim_matrix)
        denom = torch.sum(exp_sim, dim=-1) + 1e-8
        pos_mask = adjacency
        pos_exp = exp_sim * pos_mask
        numer = torch.sum(pos_exp, dim=-1) + 1e-8
        loss_i = -torch.log(numer / denom)
        syntax_loss = loss_i.mean()
        return syntax_loss







