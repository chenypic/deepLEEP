import os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
import cv2
import natsort
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler


from torch.utils.data import Dataset, DataLoader
from dataColCnnTrans import trainData,test_transform,valData
import tqdm

from timm.scheduler.cosine_lr import CosineLRScheduler
import timm
from models.deepLEEP import deepLEEP

# from dataset_yindaojing__LSIL_vs_HSIL import train_loader, X_test_patient,test_transform,X_val,X_train
#from utils import calculate_normalisation_params

#from senet.baseline import resnet20
#from senet.se_resnet import se_resnet20,se_resnet101,se_resnet34

from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]



def evaluate_qilu(model,runs_id,epoch):
    """
    Parameters:
    
    - model: A Trained Pytorch Model 
    - data_loader: A Pytorch data loader object
    """

    qilu = pd.read_csv ('qilu_clean.csv',index_col=0)
    # eryuan = eryuan.drop('{B36B3A41-4935-4C2A-B4F7-56F76C6F6777}',axis=0)
    qilu = qilu.dropna()
    grades_01 = np.array(qilu['label'])


    all_proba = []
    with torch.no_grad():
        for idx, row in tqdm.tqdm(qilu.iterrows()):
            p_id = row['img_ids']
            label_ = row['label']
            label_ = 'negative' if label_==0 else 'positive'
            imgs_per_patient = natsort.natsorted(glob.glob(f'../data/qilu_clean3/{label_}/{p_id}*.jpg'))
            imgs_per_patient_num = len(imgs_per_patient)
            images = []
            for j in range(imgs_per_patient_num):
                img = cv2.imread(imgs_per_patient[j])
                img_rgb2 = img[:,:,::-1].copy()
                # resized = cv2.resize(img_rgb2, (256, 256), interpolation=cv2.INTER_LINEAR)
                image = test_transform(img_rgb2)
                image = torch.unsqueeze(image, 0)
                images.append(image)
            img_batch = torch.cat(images,dim=0)
            #print(img_batch.shape)

            inputs = img_batch.cuda()
            outputs = model(inputs)
            ss = F.softmax
            outputs = ss(outputs)

            #print('输出的shape：',outputs.shape)
            predict_proba = torch.mean(outputs,dim=0)
            # print('输出的predict_proba的shape：',predict_proba.shape)
            predict_proba = predict_proba.detach().cpu().numpy()
            #print(predict_proba)
            all_proba.append(predict_proba)
            # torch.mean(outputs,dim=0)
        all_proba = np.array(all_proba)
        print('齐鲁的预测带概率：',all_proba.shape)

        y_true = np.array(grades_01)
        scores = all_proba[:,1]
        y_pred = np.argmax(all_proba,1)

    
        #my_matrix = np.concatenate((scores,y_true),axis=1)
        #print('my_matrix的shape:',my_matrix.shape)
        np.savetxt(f"{runs_id}/y_true_qilu.csv", y_true, delimiter=',')
        np.savetxt(f"{runs_id}/scores_qilu_{epoch}.csv", scores, delimiter=',')



        roc_auc = roc_auc_score(y_true, scores)


        fpr, tpr, thresholds = roc_curve(y_true, scores)

        true_positive = np.sum(y_pred * y_true)
        true_negative = len(y_pred.flat) - np.count_nonzero(y_pred + y_true)
        false_positive = np.count_nonzero(y_pred - y_true == 1)
        false_negative = np.count_nonzero(y_true - y_pred == 1)

   
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (false_positive + true_negative)
        error = np.sum(y_pred != y_true) / len(y_true)
        accuracy = np.sum(y_pred == y_true) / len(y_true)

        plt.figure()
        plt.plot(fpr, tpr, label ='ROC(auc={:.2f},accu={:.2f},sens={:.2f}, spec={:.2f})'.format(roc_auc,accuracy,sensitivity,specificity))
        plt.title("Yindaojing ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc=4)
        plt.savefig(f'{runs_id}/roc_齐鲁_{epoch}.png')

        return error,accuracy,sensitivity,specificity,roc_auc



def evaluate_eryuan(model,runs_id,epoch):
    """
    Parameters:
    
    - model: A Trained Pytorch Model 
    - data_loader: A Pytorch data loader object
    """

    eryuan = pd.read_csv ('二院表格/二院all.csv',index_col=0)
    # eryuan = eryuan.drop('{B36B3A41-4935-4C2A-B4F7-56F76C6F6777}',axis=0)
    eryuan = eryuan.dropna()
    grades_01 = np.array(eryuan['label'])


    all_proba = []
    with torch.no_grad():
        for idx, row in tqdm.tqdm(eryuan.iterrows()):
            p_id = row['img_ids']
            label_01 = row['label']
            label_ = 'negative' if label_01==0 else 'positive'
            imgs_per_patient = natsort.natsorted(glob.glob(f'../data/eryuan_clean/{label_}/{p_id}*.jpg'))
            imgs_per_patient_num = len(imgs_per_patient)
            images = []
            for j in range(imgs_per_patient_num):
                img = cv2.imread(imgs_per_patient[j])
                img_rgb2 = img[:,:,::-1].copy()
                # resized = cv2.resize(img_rgb2, (256, 256), interpolation=cv2.INTER_LINEAR)
                image = test_transform(img_rgb2)
                image = torch.unsqueeze(image, 0)
                images.append(image)
            img_batch = torch.cat(images,dim=0)
            #print(img_batch.shape)

            inputs = img_batch.cuda()
            outputs = model(inputs)
            ss = F.softmax
            outputs = ss(outputs)

            outputs = outputs[torch.argsort(outputs[:,1])]
            outputs_len = len(outputs)
            predict_proba = torch.mean(outputs,dim=0)


            predict_proba = predict_proba.detach().cpu().numpy()
            all_proba.append(predict_proba)
        all_proba = np.array(all_proba)

        y_true = np.array(grades_01)
        scores = all_proba[:,1]
        y_pred = np.argmax(all_proba,1)

        # 保存预测值：

        np.savetxt(f"{runs_id}/y_true_eryuan.csv", y_true, delimiter=',')
        np.savetxt(f"{runs_id}/scores_eryuan_{epoch}.csv", scores, delimiter=',')




        roc_auc = roc_auc_score(y_true, scores)

        fpr, tpr, thresholds = roc_curve(y_true, scores)

        true_positive = np.sum(y_pred * y_true)
        true_negative = len(y_pred.flat) - np.count_nonzero(y_pred + y_true)
        false_positive = np.count_nonzero(y_pred - y_true == 1)
        false_negative = np.count_nonzero(y_true - y_pred == 1)

   
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (false_positive + true_negative)
        error = np.sum(y_pred != y_true) / len(y_true)
        accuracy = np.sum(y_pred == y_true) / len(y_true)


        plt.figure()
        plt.plot(fpr, tpr, label ='ROC(auc={:.2f},accu={:.2f},sens={:.2f}, spec={:.2f})'.format(roc_auc,accuracy,sensitivity,specificity))
        plt.title("Yindaojing ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc=4)
        plt.savefig(f'{runs_id}/roc_二院_{epoch}.png')

        return error,accuracy,sensitivity,specificity,roc_auc






def train(model, epochs, train_loader, val_loader, criterion, 
          optimizer, RESULTS_PATH, scheduler=None, MODEL_PATH=None,runs_id=None):
    
    # Run on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    # Training loop
    # -------------------------------
    cols       = ['epoch', 'train_loss', 'val_loss','train_err', 'train_acc','train_sen','train_spc','train_auc','valid_err', 'valid_acc','valid_sen','valid_spc','valid_auc']
    results_df = pd.DataFrame(columns=cols).set_index('epoch')
    print('Epoch \tBatch \tNLLLoss_Train')

    weight = torch.Tensor([1.0,2.0]).cuda()

    best_test_err = 1.0

    num_steps = len(train_loader)

    
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        model.train()
        running_loss  = 0.0
        running_loss_val = 0.0
        running_loss_train = 0.0
        for i, data in enumerate(train_loader):   # Do a batch iteration
            
            # get the inputs
            inputs, labels = data['img'], data['label']
            inputs, labels = inputs.to(device), labels.long().to(device)

            # print('labels的置:',labels)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)

            # print(outputs,labels)
            loss = criterion(outputs, labels,weight=weight)
            loss.backward()
            optimizer.step()
            
            # print average loss for last 50 mini-batches
            running_loss += loss.item()
            running_loss_train += loss.item()
            if i % 50 == 49:
                print('%d \t%d \t%.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        
            if scheduler:
                scheduler.step(epoch * num_steps + i)
        
        # Record metrics
        model.eval()
        train_loss = running_loss_train / len(train_loader)
        # loss.item()

        with torch.no_grad():

            for i, data in enumerate(val_loader):   # Do a batch iteration
            
                # get the inputs
                inputs, labels = data['img'], data['label']
                inputs, labels = inputs.to(device), labels.long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels,weight=weight)
                running_loss_val += loss.item()

            val_loss = running_loss_val / len(val_loader)


        if epoch % 10 ==0:
            train_err,train_accuracy,train_sensitivity,train_specificity,train_auc = evaluate_qilu(model,runs_id,epoch)
            valid_err,valid_accuracy,valid_sensitivity,valid_specificity,valid_auc = evaluate_eryuan(model,runs_id,epoch)

        results_df.loc[epoch] = [train_loss, val_loss, train_err, train_accuracy,train_sensitivity,train_specificity,train_auc,valid_err,valid_accuracy,valid_sensitivity,valid_specificity,valid_auc]
        results_df.to_csv(RESULTS_PATH)

        torch.save(model.state_dict(), f'{runs_id}/plainnet_epoth{epoch}.pt')
        
        # Save best model
        if MODEL_PATH and (valid_err < best_test_err):
            torch.save(model.state_dict(), MODEL_PATH)
            best_test_err = valid_err

    print('Finished Training')

    torch.save(model.state_dict(), f'{runs_id}/end_model.pt')

    model.eval()
    return model




model = TRANSRESNET()

runs_id = 'run_2024-12-4-2'

os.makedirs(runs_id, exist_ok=True)


epochs = 500
lr = 0.05  # authors cite 0.1
momentum = 0.9
weight_decay = 1e-4
milestones = [30, 60, 90, 120,150,180]
gamma = 0.5

# 设置随机数种子
setup_seed(20)

criterion = F.cross_entropy

skip = {}
skip_keywords = {}
if hasattr(model, 'no_weight_decay'):
    skip = model.no_weight_decay()
if hasattr(model, 'no_weight_decay_keywords'):
    skip_keywords = model.no_weight_decay_keywords()
        
parameters = set_weight_decay(model, skip, skip_keywords)


optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                                lr=1e-3, weight_decay=0.05)


#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

WARMUP_EPOCHS = 20
n_iter_per_epoch = len(trainData)


num_steps = int(epochs * n_iter_per_epoch)
warmup_steps = int(WARMUP_EPOCHS * n_iter_per_epoch)

lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps),
            lr_min=5e-6,
            warmup_lr_init=5e-7,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=True,
)


train_loader = DataLoader(trainData, batch_size=50, shuffle= True)
val_loader = DataLoader(valData, batch_size=50, shuffle= True)


results_file = f'{runs_id}/plainnet.csv'
model_file = f'{runs_id}/plainnet.pt'


train(model, epochs, train_loader,val_loader, criterion, 
          optimizer, results_file, scheduler=lr_scheduler, MODEL_PATH=model_file,runs_id=runs_id)



