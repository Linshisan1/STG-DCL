import torch
import torch.nn.functional as F
from utils.common_utils import Logging
from tools.metric import Metric

from utils.eval_utils import get_triplets_set



def evaluate(model, dataset, stop_words, logging, args):
#将模型切换到评估模式，禁用 dropout、BatchNorm 的训练效果，以保证推理稳定
    model.eval()
#在此作用域下，禁止自动求导，节省内存并加速推理。
    with torch.no_grad():
# all_ids：存放每个句子的ID；
# all_preds：模型预测输出（logitsargmax的结果）；
# all_labels：真实标签；
# all_sens_lengths：句子长度（以token数量计）或其他长度信息；
# all_token_ranges：每个句子中 “word_spans” or “token_ranges”，帮助后续解码坐标；
# all_tokenized：原始分词结果，可能在评估时要用。
        all_ids = []
        all_preds = []
        all_labels = []
        # all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        all_tokenized = []


        for i in range(dataset.batch_count):
# 从前面看过的DataIterator.get_batch中拿到该批次的(tokens, masks, tagging_matrix, ...)等。
# 此处我们只提取tags（真实标签网格）和tokens（输入给模型的tokenIDs）等。
            sentence_ids, tokens, masks, token_ranges, tags, tokenized, _, _,_,_,_ = dataset.get_batch(i)

            # sentence_ids, bert_tokens, masks, word_spans, tagging_matrices = trainset.get_batch(i)
#调用 roberta.py 中的 Model.forward，返回 (logits, logits1, sim_matrix)，这里只把第一个返回值命名为 preds
#-,-是用来占位置的，把logits1, sim_matrix两个返回值“读走”但不命名，后续也不会再用它。
            preds, _, _ = model(tokens, masks, return_word_emb=True) #1
#在最后一维（class_num=5）进行 argmax，得到 [batch_size, seq_len, seq_len] 的预测标签整数。
            preds = torch.argmax(preds, dim=3) #2
# preds 是当前 batch 的预测网格；
# tags 是当前 batch 的真实网格；
# 用 all_preds.append(preds), all_labels.append(tags) 存起来，以便后续整合。
            all_preds.append(preds) #3
            all_labels.append(tags) #4
            # all_lengths.append(lengths) #5
# 为每条句子保存句子长度、token 范围、句子ID、分词后 token 列表等辅助信息，用于最终解码或评测时做映射。
            sens_lens = [len(token_range) for token_range in token_ranges]
            all_sens_lengths.extend(sens_lens) #6
            all_token_ranges.extend(token_ranges) #7
            all_ids.extend(sentence_ids) #8
            all_tokenized.extend(tokenized)

# 把收集在 all_preds 里的多批次预测张量（形状 [batch, seq_len, seq_len]）在 batch 维度上拼接成一个大张量。
# 这样就得到 [dataset_size, seq_len, seq_len]
# all_preds：尺寸 [N, seq_len, seq_len]（N=数据集中句子总数），记录了每个 cell 的预测标签 0/1/2/3/4
        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        # all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()
        
        # 引入 metric 计算评价指标
        metric = Metric(args, stop_words, all_tokenized, all_ids, all_preds, all_labels, all_sens_lengths, all_token_ranges, ignore_index=-1, logging=logging)
        predicted_set, golden_set = metric.get_sets()
        
        
        aspect_results = metric.score_aspect(predicted_set, golden_set)
        opinion_results = metric.score_opinion(predicted_set, golden_set)
        pair_results = metric.score_pairs(predicted_set, golden_set)
        
        precision, recall, f1 = metric.score_triplets(predicted_set, golden_set)

        aspect_results = [100 * i for i in aspect_results]
        opinion_results = [100 * i for i in opinion_results]
        pair_results = [100 * i for i in pair_results]
        
        precision = 100 * precision
        recall = 100 * recall
        f1 = 100 * f1
        
        logging('Aspect\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(aspect_results[0], aspect_results[1], aspect_results[2]))
        logging('Opinion\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(opinion_results[0], opinion_results[1], opinion_results[2]))
        logging('Pair\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(pair_results[0], pair_results[1], pair_results[2]))
        logging('Triplet\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1
