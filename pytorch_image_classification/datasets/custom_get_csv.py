import os 
import csv

dataset_mom_path = '/root/ZYM/zym/AI_S/torch_classification/data/MNIST/' #convert MNIST to the file's name of your own Dataset

train_path = dataset_mom_path + 'train/'
test_path = dataset_mom_path + 'test/'
train_class = os.listdir(dataset_mom_path + 'train/')

data = []
data_test = []

with open(dataset_mom_path + 'train_mnist_custom.csv','w',newline='',encoding='utf-8') as csvfile: #change the .csv file name too
    writer = csv.DictWriter(csvfile,fieldnames = ['class','path'])
    writer.writeheader()
    for cls in train_class:
        cls_pth = train_path + cls + '/'
        file_ls = os.listdir(cls_pth)
        for img in file_ls:
            file_path = cls_pth + img
            #writer.writerows({'class':int(cls),'path':file_path}) #writing in a single row of dic is now allowed!
            data.append({'class':int(cls),'path':file_path})
            print("writing:",cls,' ',file_path)
    writer.writerows(data)
    
with open(dataset_mom_path + 'test_mnist_custom.csv','w',newline='',encoding='utf-8') as csvfile: #change the .csv file name too
    writer = csv.DictWriter(csvfile,fieldnames = ['class','path'])
    writer.writeheader()
    for cls in train_class:
        cls_pth = test_path + cls + '/'
        file_ls = os.listdir(cls_pth)
        for img in file_ls:
            file_path = cls_pth + img
            #writer.writerows({'class':int(cls),'path':file_path}) #writing in a single row of dic is now allowed!
            data_test.append({'class':int(cls),'path':file_path})
            print("writing:",cls,' ',file_path)
    writer.writerows(data_test)
            
            

    
    