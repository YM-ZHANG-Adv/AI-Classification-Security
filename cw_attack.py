from tools.import_model import trmodel
from attack.CW import CWAttack
import numpy as np
import cv2

cfg_path = '/root/ZYM/zym/AI_S/torch_classification/configs/self_dataset/resnet.yaml'
pth_path = '/root/ZYM/zym/AI_S/torch_classification/experiments/mnist/resnet/exp05/checkpoint_00160.pth'

model = trmodel(cfg_path,pth_path).get_model()
img = cv2.imread('/root/ZYM/zym/AI_S/torch_classification/data/MNIST/test/0/5.png')
image = np.transpose(img,(2,0,1))
cw = CWAttack(model, img, c=1, lr=0.001, target=1)
adv_img = cw.attack()
adv_img_np = (adv_img.cpu().detach()).numpy()
adv_uint = (adv_img_np*255).astype(np.uint8)
cv2.imwrite("./cw_adv.jpg",adv_uint)