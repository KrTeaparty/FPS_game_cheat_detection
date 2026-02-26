import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import f1_score, precision_score, recall_score

def test(net, config, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        # frame_gt = np.load("frame_label/gt-ucf.npy")
        frame_predict = None
        
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        
        for i in range(len(test_loader.dataset)):
            
            
            #print(next(load_iter))
            _data, _label, _ = next(load_iter)
            
            _data = _data.cuda()
            _label = _label.cuda()
            
            res = net(_data)   
            a_predict = res["frame"]
            # print(temp_predict)
            # print(a_predict)
            # temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            # if (i + 1) % 10 == 0 :
            cls_label.append(int(_label))
            # a_predict = temp_predict.mean(0).cpu().numpy()
            a_predict = a_predict.mean(0).cpu().numpy()
            
            cls_pre.append(1 if a_predict.max()>0.5 else 0)          
            fpre_ = np.repeat(a_predict, 16)
            if frame_predict is None:         
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])  
                # temp_predict = torch.zeros((0)).cuda()
   
        # fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        # auc_score = auc(fpr, tpr)
    
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        f1 = f1_score(cls_label, cls_pre)
        precision = precision_score(cls_label, cls_pre, zero_division=1)
        recall = recall_score(cls_label, cls_pre)
        # print(cls_label)
        # print(cls_pre)
        
        # precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        # ap_score = auc(recall, precision)

        test_info["step"].append(step)
        # test_info["auc"].append(auc_score)
        # test_info["ap"].append(ap_score)
        test_info["f1"].append(f1)
        test_info["precision"].append(precision)
        test_info["recall"].append(recall)
        test_info["ac"].append(accuracy)