import argparse
import torch
from pytorch_image_classification import (
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    set_seed,
    setup_cudnn,
)

class trmodel:
    #config is .yaml file
    def __init__(self,cfg_path,pth_path):
        self.config = self.load_config(cfg_path)
        self.check_point_path = pth_path


    def load_config(self,cfg_path):
        config = get_default_config()
        if cfg_path is not None:
            config.merge_from_file(cfg_path)
        if not torch.cuda.is_available():
            config.device = 'cpu'
            config.train.dataloader.pin_memory = False
        config.merge_from_list(['train.dist.local_rank', 0])
        config = update_config(config)
        config.freeze()
        return config
    
    def get_model(self):
        config = self.config
        set_seed(config)
        setup_cudnn(config)
        model = create_model(config)
        check_point = torch.load(self.check_point_path)
        model.load_state_dict(check_point['model'],strict=False)
        #transform = create_transform(config,is_train=True)
        device = torch.device(config.device)
        model.to(device)
        _ = model.eval()
        # 冻结模型参数
        for param in model.parameters():
            param.requires_grad = False
        return model

