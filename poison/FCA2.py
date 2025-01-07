import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_name):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.feature = None
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self.feature = output.detach()
        
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)

    def forward(self, x):
        self.model.train()
        _ = self.model(x)
        return self.feature
    

from tqdm.auto import trange 
import copy
from torch.autograd import grad

class PoisonAttack2:
    def __init__(self,target_instance,base_instance_list,model,feature_layer,device = torch.device("cuda:0")):
        self.device = device
        self.target = target_instance.to(self.device)
        self.base_list = self.get_base_list(base_instance_list)#所有的目标图像，基类图像以及模型统一设置到device当中
        self.model = model.to(self.device)
        self.feature_layer = feature_layer # input a string, like 'stage3.block1'
        self.lr = 1
        self.stopping_tol = 1e-10
        self.similarity_coeff = 256.0
        self.max_iteration = 120
        self.num_obj = 10  #保存的objectives的值
        self.decay_coeff: float = 0.5
        self.watermark = True
        self.watermark_rate = 0.3
        self.feature = None
        self._register_hook()


    def _register_hook(self):
        def hook(module,input,output):
            self.feature = output.detach()

        for name,module in self.model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook)
        

    def get_base_list(self,raw_list):
        new_list = []
        for img_tensor in raw_list:
            new_list.append(img_tensor.to(self.device))
        return new_list
    
    def get_objective(self,feature1,feature2,image1,image2):
        """
        feature1: The activation layer's feature of target image
        feature2: The activation layer's feature of poison(base) image
        image1: base image batch tensor
        image2: poison image batch tensor
        """
        beta =  self.similarity_coeff * (feature2.numel() / image2.numel())**2
        return torch.norm(feature2 - feature1) + beta*torch.norm(image2 - image1)
    
    def fca_forward(self,old_poison,target_tensor):
        isolate_model = copy.deepcopy(self.model)
        isolate_model.to(self.device)
        for param in isolate_model.parameters():
            param.requires_grad = True

        def get_feature_from_layer(model, target_layer_name, input_tensor):
            feature = None
            def hook_function(module, input, output):
                nonlocal feature  # 使用 nonlocal 变量来保存特征
                feature = output
            hook = None
            for name, module in model.named_modules():
                if name == target_layer_name:
                    hook = module.register_forward_hook(hook_function)
                    break
            _ = model(input_tensor)
            if hook is not None:
                hook.remove()
            return feature  # 返回目标特征层的输出
        
        poison = old_poison.clone().detach().requires_grad_(True) 
        target = target_tensor.clone().detach()
        #target.requires_grad_(False)
        poison_feature = get_feature_from_layer(isolate_model,self.feature_layer,poison)
        target_feature = get_feature_from_layer(isolate_model,self.feature_layer,target)
        poison_feature.requires_grad_(True) 
        target_feature.requires_grad_(True) 
        diff = poison_feature - target_feature
        #loss = diff.norm(p=2)
        loss = torch.norm(diff,p=2)
        isolate_model.zero_grad() #设定模型冻结，反向传播获取关于损失函数的梯度，由poison矩阵得来
        loss.backward()
        grads = poison.grad
        poison = poison -  self.lr*grads
        return poison
    
    def fca_backward(self,base_batch,feature_rep,poison):
        num_features = feature_rep.numel()
        dim_features = feature_rep.shape[-1]
        beta = self.similarity_coeff * (dim_features/num_features)**2
        poison = (poison + self.lr * beta * base_batch) / (1 + beta * self.lr)
        poison = torch.clamp(poison,min=0.0,max=1.0)
        return poison
    
    def get_predicted_labels(self,batch_image_list):
        """
        input type is a list of batch_format signal image tensor,return the corresponding predicted one-hot labels list
        poison attack training process may need use the predict label instead of true label
        """
        predicted_labels_list = []
        model = self.model
        model.eval()
        for image_batch in batch_image_list:
            label = model(image_batch)
            predicted_labels_list.append(label)
        return predicted_labels_list


    
    def poison(self):
        # isolate model and feature extractor
        # self.model = copy.deepcopy(self.model)
        # self.model.to(self.device)
        self.model.train()
        num_posion = len(self.base_list)
        final_poisoned_images = []
        if num_posion == 0:
            raise ValueError("No images input!")
        target_image_batch = self.target.unsqueeze(0) #将单张的目标图像转换为batch形式   
        _ = self.model(target_image_batch)
        target_feature = self.feature
        for image_tensor in self.base_list:
            base_batch = image_tensor.unsqueeze(0)
            old_poison_batch = image_tensor.unsqueeze(0)

            _ = self.model(old_poison_batch)
            poison_feature = self.feature 

            old_objective = self.get_objective(target_feature,poison_feature,base_batch,old_poison_batch)
            last_m_objectives = [old_objective]

            for i in trange(self.max_iteration,desc="FCAing!!!"):
                new_poison = self.fca_forward(old_poison_batch,target_image_batch)
                new_poison = self.fca_backward(base_batch,poison_feature,new_poison) 
                ref_change_value = torch.norm(new_poison - old_poison_batch) / torch.norm(old_poison_batch)
                if(ref_change_value < self.stopping_tol):
                    print("stopped after {} iterations due to small changes".format(i))
                    break

                _ =  self.model(new_poison)
                new_poison_feature = self.feature

                new_objective = self.get_objective(target_feature,new_poison_feature,base_batch,new_poison)

                avg_last_m = sum(last_m_objectives) / float(min(i+1,self.num_obj))
                #chop the learning rate in iterations 
                if new_objective >= avg_last_m and (i % self.num_obj / 2 == 0): #在循环次数为存储次数偶数倍时执行
                    self.lr *= self.decay_coeff
                else:
                    old_poison_batch = new_poison
                    old_objective = new_objective

                if i < self.num_obj -1:
                    last_m_objectives.append(new_objective)
                else:
                    del last_m_objectives[0]
                    last_m_objectives.append(new_objective)
                
            # Watering Process
            if self.watermark:
                watermark_image = self.watermark_rate * target_image_batch
                final_poison = torch.clamp(old_poison_batch + watermark_image,min = 0.0, max = 1.0)
                final_poisoned_images.append(final_poison)

            else:
                final_poisoned_images.append(old_poison_batch)

            final_poison_labels = self.get_predicted_labels(final_poisoned_images)

        return final_poisoned_images,final_poison_labels
