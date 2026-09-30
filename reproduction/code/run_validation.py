"""Faithful re-run of EEG-Conformer conformer.py on BCI IV 2a.

Model classes and the training loop are copied VERBATIM from the released
conformer.py (lines 66-410). Every deviation is marked with  # [MOD].
Protocol A = exactly what the released code does (train on all 288 trials of
session T, evaluate on all 288 trials of session E every epoch).
Protocol B = identical, but 20% of session T is held out as a validation set
and the reported epoch is chosen by validation accuracy instead of test
accuracy.
"""
import argparse, os, sys, random, time, datetime, math
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

gpus = [0]

# ===================== VERBATIM from conformer.py L66-212 =====================
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )
# =================== END VERBATIM model section ===================


class ExP():
    def __init__(self, nsub, root, logpath, n_epochs, val_frac):   # [MOD] args instead of hardcoded
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = n_epochs                                    # [MOD] was 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub
        self.val_frac = val_frac                                    # [MOD] protocol B
        self.start_epoch = 0
        self.root = root                                            # [MOD] was '/Data/strict_TE/'
        self.log_write = open(logpath, "w")                         # [MOD] path
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]
            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]
        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label - 1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']
        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)
        self.allData = self.train_data
        self.allLabel = self.train_label[0]
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)
        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # [MOD] protocol B: carve a validation set out of session T before standardising
        self.valData = self.valLabel = None
        if self.val_frac > 0:
            nval = int(round(self.val_frac * len(self.allData)))
            self.valData, self.valLabel = self.allData[:nval], self.allLabel[:nval]
            self.allData, self.allLabel = self.allData[nval:], self.allLabel[nval:]

        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std
        if self.valData is not None:                                 # [MOD]
            self.valData = (self.valData - target_mean) / target_std
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        if self.valData is not None:                                 # [MOD]
            val_data = Variable(torch.from_numpy(self.valData).type(self.Tensor))
            val_label = Variable(torch.from_numpy(self.valLabel - 1).type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0
        test_curve, val_curve, train_curve = [], [], []              # [MOD] per-epoch logging
        total_step = len(self.dataloader)
        curr_lr = self.lr
        t0 = time.time()                                             # [MOD] timing

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.model.eval()
                with torch.no_grad():                                # [MOD] no_grad (numerically identical, saves memory)
                    Tok, Cls = self.model(test_data)
                    loss_test = self.criterion_cls(Cls, test_label)
                    y_pred = torch.max(Cls, 1)[1]
                    acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                    train_pred = torch.max(outputs, 1)[1]
                    train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                    vacc = float('nan')                              # [MOD]
                    if self.valData is not None:
                        _, vCls = self.model(val_data)
                        vacc = float((torch.max(vCls, 1)[1] == val_label).cpu().numpy().astype(int).sum()) / float(val_label.size(0))
                test_curve.append(acc); val_curve.append(vacc); train_curve.append(train_acc)
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred
                if (e + 1) % 200 == 0:
                    print('Epoch %d  loss %.4f  train %.4f  test %.4f  val %.4f  (%.3f s/ep)'
                          % (e, loss.item(), train_acc, acc, vacc, (time.time() - t0) / (e + 1)), flush=True)

        averAcc = averAcc / num
        self.sec_per_epoch = (time.time() - t0) / self.n_epochs      # [MOD]
        self.test_curve = np.array(test_curve)                       # [MOD]
        self.val_curve = np.array(val_curve)                         # [MOD]
        self.train_curve = np.array(train_curve)                     # [MOD]
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        self.log_write.close()
        return bestAcc, averAcc, Y_true, Y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--val-frac', type=float, default=0.0)
    ap.add_argument('--tag', type=str, default='A')
    ap.add_argument('--root', type=str, default='data_proc/')   # [MOD] repo-relative default
    ap.add_argument('--out', type=str, default='results')        # [MOD] repo-relative default
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed_n = args.seed
    random.seed(seed_n); np.random.seed(seed_n); torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n); torch.cuda.manual_seed_all(seed_n)

    name = f'{args.tag}_s{args.subject}_seed{seed_n}'
    print(f'=== {name} | epochs={args.epochs} val_frac={args.val_frac} | GPU {torch.cuda.get_device_name(0)}', flush=True)
    t0 = datetime.datetime.now()
    exp = ExP(args.subject, args.root, f'{args.out}/log_{name}.txt', args.epochs, args.val_frac)
    bestAcc, averAcc, _, _ = exp.train()
    np.savez(f'{args.out}/curve_{name}.npz', test=exp.test_curve, val=exp.val_curve,
             train=exp.train_curve, best=bestAcc, aver=averAcc, sec_per_epoch=exp.sec_per_epoch,
             subject=args.subject, seed=seed_n, val_frac=args.val_frac, tag=args.tag)
    print(f'DONE {name} best={bestAcc:.4f} aver={averAcc:.4f} final={exp.test_curve[-1]:.4f} '
          f'sec/epoch={exp.sec_per_epoch:.3f} wall={datetime.datetime.now()-t0}', flush=True)


if __name__ == "__main__":
    main()
