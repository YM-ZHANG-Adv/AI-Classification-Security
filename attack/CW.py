import torch
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CWAttack:
    def __init__(self, model, image, c=1, lr=0.001, target=None, max_iteration=100):
        self.model = model
        if isinstance(image, np.ndarray):
            self.x = torch.from_numpy(image).float().to(device)#images got by cv2.imread range to 255,need normalization
        else:
            self.x = image.clone().to(device)
        self.x = self.x/255
        self.x.requires_grad_(True)
        self.c = c
        self.lr = lr
        self.target = target
        self.max_iteration = max_iteration
        #self.w = torch.tensor(torch.atanh(2*self.x-1),requires_grad=True).to(device)
    
    def attack(self):
        w = torch.tensor(torch.atanh(2*self.x-1),requires_grad=True).to(device)
        optimizer = torch.optim.Adam([w],lr = self.lr)
        for _ in tqdm(range(self.max_iteration)):
            adv_img = (torch.tanh(w)+1)/2
            loss_mse = torch.nn.MSELoss()(adv_img,self.x)
            loss_f = self.loss_f(w)
            loss = loss_mse+self.c*loss_f
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
        adv_img_raw = (torch.tanh(w)+1)/2
        adv_img = torch.round(torch.clamp(adv_img_raw*255, min=0., max=255.))/255
        return adv_img
        
    def loss_f(self,w):
        model = self.model
        adv = (torch.tanh(w)+1)/2
        adv = adv.permute(2,0,1) # convert HWC to CHW
        if len(adv.shape) == 3:  # 单个图像
            adv = adv.unsqueeze(0)  # 增加 batch 维度 
        result = model(adv).squeeze(0)
        target_logit = result[self.target]
        result[self.target] = 0.
        max_non_target_logit = torch.max(result)
        loss = torch.clamp_min(max_non_target_logit - target_logit,0.) 
        return loss

    
    
        

        
    