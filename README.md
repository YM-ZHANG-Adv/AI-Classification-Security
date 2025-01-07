# AI-Classification-Security
This is the  unofficial repository of  Fudan University's class **Artificial Intelligence Security**

## preparation
The whole project is heavily based on [pytorch_image_classification](https://github.com/hysts/pytorch_image_classification), but due to the original codes only support those well-known open-source datasets, which called via the torch.dataset function, and didn't update the repository anymore. So at the beginning, I tried to implement and Class which is abel to train self-made dataset and based on csv file. Since then I didn't know anythings about torchvision.datasets.ImageFolder, in the following update it's used in the training of MSTAR dataset, which is a standard Synthetic Aperture Radar(SAR) dataset. More detailed information is recorded in **self_made_train** markdown file.    

As for the model weights, **resnet110** with **MNIST** is available. For 3-category weight, path is `./experiments/mnist/resnet/exp00/checkpoint_00160.pth`; for 10-category weight, path is `./experiments/mnist/resnet/exp06/checkpoint_00160.pth`; for badnet weight path, it exists in `./experiments/mnist/resnet/badnet04/checkpoint_00012.pth`. However, I trained **resnext** model on MSTAR dataset, which is too big to upload the model weight. Therefore, if you need the weight file, please let me know and I will provide Baidu Netdisk or Google Drive for you.

## Attack: CW 
The CW algorithm is completely reproduced in this project, the corresponding result is in `attack.ipynb`. And the implementation of torch-attack's CW is also in the file.

## Poison: FCA  
For poison task, I selected [FCA](https://arxiv.org/abs/1804.00792) method and thoroughly reproduced it, which is available at `poison.ipynb` and `fca_poison_test.py`.  

## Backdoor: BadNets  
For this task, according to the [paper](https://arxiv.org/abs/1708.06733) there's no complex algorithm or architecture. All I did is select a small image as the icon to fine-tuning a resnet model to combine the icon with the specific category. The fine-tuning process and corresponding result is in `badnet.ipynb`.  

## Watermark: Deepsigns
[DeepSigns](https://dl.acm.org/doi/abs/10.1145/3297858.3304051) method need retrain the model and need an isolate project matrix to extract the watermark information in target feature layer, whose path is `./watermark/projection_matrix.npy`,and the complete code is in `watermark.ipynb`.  

## Inversion: DLG
[Deep Leakage From Gradients](https://proceedings.neurips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf) is a classical white-box data theft method. However, it's a great pity that I didn't finish it on this framework, instead, a very simply is created to reproduce the algorithm. You can check the content in `DLG.ipynb`. I will also port this to the native model later, just keep looking forward.   


# END
If you have any problem or suggestion, feel free to get in touch with me!! 
