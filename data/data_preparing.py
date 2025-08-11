import math
import torch
import json
import os
import copy
import torch.nn.functional as F
import stanza
# 初始化 Stanza 管道（建议只初始化一次，可放在全局变量中）
# 如果需要下载模型请先执行：stanza.download('en')
#nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True)

class Instance(object):
    '''
    After modification:
     1) Each instance stores token_classes => [0..4]
     2) Each instance stores head => python list of length = L_token
     3) Remove old cl_mask logic
     4) Retain GTS tagging_matrix if you need
    '''

    def __init__(self, tokenizer, single_sentence_pack, args):
        self.args = args
        self.sentence = single_sentence_pack['sentence']

        # roberta tokens
        self.tokens = tokenizer.tokenize(self.sentence, add_prefix_space=True)
        self.L_token = len(self.tokens)

        # store head => from data
        # e.g. single_sentence_pack['head'] => list of ints length L
        self.head = single_sentence_pack.get('head', [])
        #self.head = self.get_dependency_heads()
        # store postag, deprel if you want
        self.postag = single_sentence_pack.get('postag', [])
        self.deprel = single_sentence_pack.get('deprel', [])

        self.word_spans = self.get_word_spans()
        self._word_spans = copy.deepcopy(self.word_spans)

        self.id = single_sentence_pack['id']
        self.triplets = single_sentence_pack['triples']

        # from old code => triplets_in_spans => AST decoding
        self.triplets_in_spans = self.get_triplets_in_spans()

        # create token_classes => [L_token], 0..4
        self.token_classes = self.get_token_classes()

        # no old cl_mask
        # self.cl_mask = ...

        # check word_spans
        assert len(self.sentence.strip().split(' ')) == len(self.word_spans), \
            f"Mismatch in word_spans count vs actual words for {self.sentence}"

        # build roberta tokens => padded
        self.bert_tokens = tokenizer.encode(
            self.sentence,
            add_special_tokens=False,
            add_prefix_space=True
        )
        self.bert_tokens_padded = torch.zeros(args.max_sequence_len).long()
        for i in range(len(self.bert_tokens)):
            self.bert_tokens_padded[i] = self.bert_tokens[i]

        # build mask => for GTS
        self.mask = self.get_mask()

        if len(self.bert_tokens) != self._word_spans[-1][-1] + 1:
            print("WARNING mismatch tokens vs word_spans:", self.sentence, self._word_spans)

        self.tagging_matrix = self.get_tagging_matrix()
        self.tagging_matrix = (self.tagging_matrix + self.mask - torch.tensor(1)).long()



    def get_word_spans(self):
        '''
        each 'Ġ' => new word boundary
        '''
        l_indx = 0
        r_indx = 0
        word_spans = []
        while r_indx + 1 < len(self.tokens):
            if self.tokens[r_indx+1][0] == 'Ġ':
                word_spans.append([l_indx, r_indx])
                r_indx += 1
                l_indx = r_indx
            else:
                r_indx += 1
        word_spans.append([l_indx, r_indx])
        print("word_spans", word_spans)
        return word_spans

    def get_triplets_in_spans(self):
        triplets_in_spans = []
        sentiment2id = {'negative':2, 'neutral':3, 'positive':4}

        for triplet in self.triplets:
            aspect_tags = triplet['target_tags']
            opinion_tags= triplet['opinion_tags']
            sentiment = triplet['sentiment']
            s_id = sentiment2id[sentiment]

            aspect_spans = self.get_spans_from_BIO(aspect_tags)
            opinion_spans= self.get_spans_from_BIO(opinion_tags)

            triplets_in_spans.append((aspect_spans, opinion_spans, s_id))
        print("sentence= ", self.sentence)
        print("triplets_in_spans", triplets_in_spans)
        print("head",self.head)
        return triplets_in_spans

    def get_spans_from_BIO(self, tags):
        '''
        parse B/I/O => produce [ [start,end], ... ] from self.word_spans
        '''
        token_ranges = copy.deepcopy(self.word_spans)
        t_list = tags.strip().split()
        spans = []
        for i, tag in enumerate(t_list):
            if tag.endswith('B'):
                spans.append(token_ranges[i])
            elif tag.endswith('I'):
                # extend last
                spans[-1][-1] = token_ranges[i][-1]
            else:
                pass
        return spans

    def get_token_classes(self):
        '''
        define 0=none,1=aspect,2=neg,3=neu,4=pos
        fill from triplets_in_spans
        '''
        token_classes = [0]*self.L_token
        for (asp_spans, opi_spans, s_id) in self.triplets_in_spans:
            # aspect => label=1
            for (al,ar) in asp_spans:
                for idx in range(al, ar+1):
                    token_classes[idx] = 1
            # opinion => label= s_id(2..4)
            for (ol,or_) in opi_spans:
                for idx in range(ol, or_+1):
                    token_classes[idx] = s_id
        print("token_classes", token_classes)
        return token_classes

    def get_mask(self):
        mask = torch.ones((self.args.max_sequence_len, self.args.max_sequence_len))
        actual_len = len(self.bert_tokens)
        mask[:, actual_len:] = 0
        mask[actual_len:, :] = 0
        for i in range(actual_len):
            mask[i][i] = 0
        return mask

    def get_tagging_matrix(self):
        '''
        build GTS => shape (max_seq_len, max_seq_len)
        fill (aspect,opinion) => top-left => sentiment, else ctd=1
        '''
        mat = torch.zeros((self.args.max_sequence_len, self.args.max_sequence_len))
        for (asp_spans, opi_spans, s_id) in self.triplets_in_spans:
            for (al,ar) in asp_spans:
                for (pl,pr) in opi_spans:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            if i==al and j==pl:
                                mat[i][j] = s_id
                            else:
                                mat[i][j] = 1
        return mat


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        # gather for this batch
        sentence_ids = []
        word_spans = []
        bert_tokens = []
        masks = []
        tagging_matrices = []
        tokenized = []

        token_classes_list = []   # store each inst's 0..4
        postag_list = []
        head_list = []
        deprel_list = []

        # [NEW] 用于存放本batch里每个句子的 triplets_in_spans
        triplets_in_spans_list = []  # [NEW]

        start_idx = index*self.args.batch_size
        end_idx   = min((index+1)*self.args.batch_size, len(self.instances))
        for i in range(start_idx, end_idx):
            inst = self.instances[i]
            sentence_ids.append(inst.id)
            word_spans.append(inst.word_spans)
            bert_tokens.append(inst.bert_tokens_padded)
            masks.append(inst.mask)
            tagging_matrices.append(inst.tagging_matrix)
            tokenized.append(inst.tokens)

            token_classes_list.append(inst.token_classes)
            postag_list.append(inst.postag)
            head_list.append(inst.head)
            deprel_list.append(inst.deprel)

            triplets_in_spans_list.append(inst.triplets_in_spans)  # [NEW]

        if len(bert_tokens)==0:
            print("Warning: empty batch??")

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        tagging_matrices = torch.stack(tagging_matrices).long().to(self.args.device)

        # return 10 items => adapt to your usage in trainer
        return (
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
        )



# 删除 / 注释旧对比学习：
#
# 不再保留 get_cl_mask() 函数和 self.cl_mask。
# 同时在 DataIterator 不再返回 cl_masks。
# 新增 self.postag, self.head, self.deprel：
#
# 在 __init__() 里直接取 single_sentence_pack['postag'], 'head', 'deprel'.
# 这样后续在 trainer.py 就能使用这些句法信息进行“SSCL”之类的对比。
# 其余逻辑（word_spans, triplets_in_spans, tagging_matrix 等）依旧保留，用于BIO解析 / 网格标注 / token-level labeling。
#
# 在 DataIterator.get_batch() 中，不再返回 cl_masks。增加 postag, head, deprel，以便 trainer.py 能取到这些字段做依存树对比学习（如果你想在 trainer 里构建 pairing）。
#
# 如果你想只在 Instance 就构建 pairing，也可以，但通常在 trainer 里构造embedding后再 pairing pull/push 也行。这里我仅给你保留 postag, head, deprel → 你就能看在 trainer 里 get_batch(...) 解析它们了。
#
# 删除：旧 get_cl_mask() 和 self.cl_mask（含 cl_masks）部分。
# 新增：
# self.postag = single_sentence_pack.get('postag', [])
# self.head = single_sentence_pack.get('head', [])
# self.deprel = single_sentence_pack.get('deprel', [])
# 并在 DataIterator.get_batch() 返回 postag_list, head_list, deprel_list → 这样下游 trainer.py 能拿到依存句法与词性信息以进行 SSCL 对比学习。
# 保留：
# word_spans, tagging_matrix, token_classes → ASTE 原功能。
if __name__ == "__main__":

    import sys
    from transformers import RobertaTokenizer, RobertaModel
    import argparse
    # import os
    
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    
    # 将上一级目录添加到sys.path
    sys.path.append(parent_dir)
    from utils.data_utils import load_data_instances

    # Load Dataset
    train_sentence_packs = json.load(open(os.path.abspath('D1/res14/train.json')))
    # # random.shuffle(train_sentence_packs)
    # dev_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/dev.json')))
    # test_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/test.json')))


    #加载预训练字典和分词方法
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base",
        cache_dir="../modules/models/",  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
        force_download=False,  # 是否强制下载
    )

    # 创建一个TensorBoard写入器
    torch.cuda.set_device(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequence_len', type=int, default=100, help='max length of the tagging matrix')
    parser.add_argument('--sentiment2id', type=dict, default={'negative': 2, 'neutral': 3, 'positive': 4}, help='mapping sentiments to ids')
    parser.add_argument('--model_cache_dir', type=str, default='./modules/models/', help='model cache path')
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', help='reberta model path')
    parser.add_argument('--batch_size', type=int, default=16, help='json data path')
    parser.add_argument('--device', type=str, default="cuda", help='gpu or cpu')
    parser.add_argument('--prefix', type=str, default="./data/", help='dataset and embedding path prefix')

    parser.add_argument('--data_version', type=str, default="D1", choices=["D1", "D2"], help='dataset and embedding path prefix')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"], help='dataset')

    parser.add_argument('--bert_feature_dim', type=int, default=768, help='dimension of pretrained bert feature')
    parser.add_argument('--epochs', type=int, default=2000, help='training epoch number')
    parser.add_argument('--class_num', type=int, default=5, help='label number')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"], help='option: pair, triplet')
    parser.add_argument('--model_save_dir', type=str, default="./modules/models/saved_models/", help='model path prefix')
    parser.add_argument('--log_path', type=str, default="log.log", help='log path')


    args = parser.parse_known_args()[0]

    
    train_instances = load_data_instances(tokenizer, train_sentence_packs, args)
    # dev_instances = load_data_instances(tokenizer, dev_sentence_packs, args)
    # test_instances = load_data_instances(tokenizer, test_sentence_packs, args)

    trainset = DataIterator(train_instances, args)
    # devset = DataIterator(dev_instances, args)
    # testset = DataIterator(test_instances, args)
