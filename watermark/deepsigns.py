import numpy as np # type: ignore
import sys
sys.argv = ['run.py']

import torch# type: ignore
import cv2# type: ignore

import torch.nn.functional as F
import copy
from tqdm import trange,tqdm



class DeepSignsWatermark:
    def __init__(self,model,dataloader,layer_name):
        self.device = torch.device("cuda:0")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.lr = 0.001
        self.epochs = 5
        self.scale = 0.01 #for loss1
        self.gamma = 0.01 #for loss2
        self.embed_bits = 16 #the length of watermark
        self.save_path = "/root/ZYM/zym/AI_S/torch_classification/watermark/projection_matrix.npy"
        self.n_classes = 10
        self.b = np.random.randint(2, size=(self.embed_bits, self.n_classes))
        self.feature_layer = layer_name
        self.centers = None
        self.feature = None
        self.target_class = 0
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self.feature = output.detach()
            #print(f"Feature captured: {self.feature.shape}")  # Debugging line

        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook)
                #print(f"Hook registered to layer: {name}")  # Debugging line

    def wm_train(self,feature_shape):
        self.model.train()
        self.centers = torch.nn.Parameter(torch.rand(self.n_classes, *feature_shape).to(self.device), requires_grad=True)
        centers = self.centers
        device = self.device
        optimizer = torch.optim.RMSprop([
        {'params': self.model.parameters()},
        {'params': centers}
        ], alpha=0.9, lr=self.lr, eps=1e-8,
        weight_decay=0.001) 
        criterion = torch.nn.CrossEntropyLoss().cuda()
        x_value = np.random.randn(self.embed_bits, *feature_shape) #产生一个投影向量
        np.save(self.save_path,x_value)
        x_value = torch.tensor(x_value, dtype=torch.float32).to(device)#转换为torch
        b = torch.tensor(self.b).to(self.device)
        for ep in tqdm(range(self.epochs+1),desc="Retaining Epoch "):
            #print(ep)
            for d,t in self.dataloader:
                d = d.to(device)
                t = t.to(device)
                optimizer.zero_grad()
                pred = self.model(d)
                feat = copy.deepcopy(self.feature).to(device)
                loss = criterion(pred,t)
                centers_batch = torch.index_select(centers, 0, t)
                #centers初始化过程中为随机生成,但仍然满足不同类别数据GMM分布的定义
                loss1 = F.mse_loss(feat, centers_batch, reduction='sum') / 2
                centers_batch_reshape = torch.unsqueeze(centers_batch, 1)
                centers_reshape = torch.unsqueeze(centers, 0)
                pairwise_dists = (centers_batch_reshape - centers_reshape) ** 2
                pairwise_dists = torch.sum(pairwise_dists, dim=(-3,-2,-1))
                arg = torch.topk(-pairwise_dists, k=2)[1]
                arg = arg[:, -1]
                closest_cents = torch.gather(centers, 0, arg.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, feat.shape[1],feat.shape[2],feat.shape[3]))
                #closest_cents = torch.gather(centers, 0, arg.unsqueeze(1).repeat(1, feat.shape[1]))
                dists = torch.sum((centers_batch - closest_cents) ** 2, dim=(-3,-2,-1))
                cosines = torch.mul(closest_cents, centers_batch)
                cosines = torch.sum(cosines, dim=(-3,-2,-1))
                loss2 = (cosines * dists - dists).mean()
                loss3 = (1 - torch.sum(centers ** 2, dim=(1,2,3))).abs().sum()
                loss4 = 0
                embed_center_idx = self.target_class
                idx_classK = (t == embed_center_idx).nonzero(as_tuple=True)
                if len(idx_classK[0]) >= 1:
                    idx_classK = idx_classK[0]
                    
                    #activ_classK = torch.gather(centers_batch, 0,idx_classK.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, feat.shape[1],feat.shape[2],feat.shape[3]))
                    activ_classK_2 = []
                    index_list = idx_classK.detach().cpu().numpy()
                    centers_batch_numpy = centers_batch.detach().cpu().numpy()
                    for index in index_list:
                        activ_classK_2.append(centers_batch_numpy[index])
                    activ_classK_2_numpy = np.array(activ_classK_2)
                    #将 NumPy 数组转换为 PyTorch 张量
                    activ_classK_2_tensor = torch.tensor(activ_classK_2_numpy).to(self.device)
                    center_classK = torch.mean(activ_classK_2_tensor, dim=0)
                    center_classK = center_classK.view(-1,1)
                    x_value = x_value.view(16,-1)
                    Xc = torch.matmul(x_value, center_classK)
                    Xc = Xc.squeeze()
                    bk = b[:, embed_center_idx]
                    bk_float = bk.float()
                    probs = torch.sigmoid(Xc)
                    entropy_tensor = F.binary_cross_entropy(target=bk_float, input=probs, reduce=False)
                    loss4 += entropy_tensor.sum()
                (loss + self.scale * (loss1 + loss2 + loss3) + self.gamma * loss4).backward()
                optimizer.step()
        return copy.deepcopy(self.model.to(device))