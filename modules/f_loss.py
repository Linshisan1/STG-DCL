import torch
import torch.nn.functional as F

# Loss Func
class FocalLoss(torch.nn.Module):
    def __init__(self, weight, alpha=0.99, gamma=4., size_average=True, ignore_index=-1):
        super(FocalLoss, self).__init__()
#3.2中的参数这是 Focal Loss 中常见的一个超参数，用于平衡容易样本与困难样本的贡献。
        self.alpha = alpha
#3.2中的参数用于放大难分类样本的损失，减弱易分类样本的损失
        self.gamma = gamma
#定在计算损失时要忽略的标签值。比如 -1 表示那个位置的标签无效，不纳入损失计算。
#在 ASTE 里，有些地方可能用于 “mask padding” 或 “不需要预测的区域
        self.ignore_index = ignore_index
#是否对所有样本的损失取平均。如果为 False，则对损失做加和；True 表示平均化
        self.size_average = size_average
#weight就是main函数里设置的那些参数
        self.weight = weight

#一般对应网络输出的 logits 或 logits1
    def forward(self, inputs, targets):
        # F.cross_entropy(x,y)工作过程就是(Log_Softmax+NllLoss)：①对x做softmax,使其满足归一化要求，结果记为x_soft;②对x_soft做对数运算
        # 并取相反数，记为x_soft_log;③对y进行one-hot编码，编码后与x_soft_log进行点乘，只有元素为1的位置有值而且乘的是1，
        # 所以点乘后结果还是x_soft_log
        # 总之，F.cross_entropy(x,y)对应的数学公式就是CE(pt)=-1*log(pt)
        # inputs = inputs.long()
        # targets = targets.long()

        # print(inputs.dtype)
        # inputs = inputs.float()

        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', ignore_index=self.ignore_index)
#inputs 一般是网络输出的 logits 就是roberta中得到的logits1 和logit
#targets 形状与前者相对应（除了最后一维），表示每个位置上的类别 id。若是多分类 [N, CTD, POS, NEU, NEG]，就会给 gold_label_5class 取值 0∼4若是二分类 [0, 1] 用 logits1 时，对应 gold_label_2class 取值 0∼1
#weight = self.weight给每个类别添加额外权重。
#reduction='none'不要在函数内部做平均或求和，而是直接返回与 targets 同形状的 loss 张量，以便后续再做自定义操作（如 focal 公式）
#ignore_index=self.ignore_index忽略标签等于 -1 的位置，不计算损失

        pt = torch.exp(-ce_loss)  # pt是预测该类别的概率，要明白F.cross_entropy工作过程就能够理解
#本质是 -log(pt): 这里 pt = softmax(inputs)[targets]
#所以ce_loss = - log(pt) ⇒ pt = exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#这里就是对应3.2的损失函数

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

