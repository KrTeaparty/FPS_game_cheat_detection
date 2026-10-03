import os
import torch
import numpy as np
import torch.utils.data as data
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle
import math
from tqdm import tqdm
from model import *
from train import *
import utils
from options import init_args

class CS2_dataloader(data.DataLoader):
    # 전역 캐시 딕셔너리 (메모리에 유지)
    feature_cache = {}

    def __init__(self, root_dir, modal, mode, num_segments, len_feature, list_fname, idx_list=None, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature

        data_path = os.path.join('list', '{}_{}.list'.format(list_fname, 'Normal' if is_normal else 'Bot'))
        with open(data_path, 'r') as f:
            full_list = [line.strip().split() for line in f]
            
        if idx_list is not None:
            self.vid_list = [full_list[i] for i in idx_list]
        else:
            self.vid_list = full_list

        # RAM에 모두 사전 로드 (I/O 병목 제거)
        for vid_info in self.vid_list:
            path = vid_info[0]
            if path not in CS2_dataloader.feature_cache:
                CS2_dataloader.feature_cache[path] = np.load(path).astype(np.float32)

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        vid_info = self.vid_list[index][0]  
        name = os.path.basename(vid_info).split(".npy")[0]
        # 디스크가 아닌 메모리 캐시에서 직접 읽기
        video_feature = CS2_dataloader.feature_cache[vid_info]

        
        label = 0 if "Normal" in vid_info else 1
        
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=int)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
            video_feature = new_feat
            
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label    

def test(net, config, test_loader, test_info, step, stride, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        cls_label = []
        cls_pre_05 = []
        cls_pre_08 = []
        
        for _data, _label, _name in test_loader:
            _data = _data.cuda()
            res = net(_data)   
            a_predict = res["frame"].mean(0).cpu().numpy()
            
            cls_label.append(int(_label))
            
            max_score = a_predict.max()
            cls_pre_05.append(1 if max_score > 0.5 else 0)
            cls_pre_08.append(1 if max_score > 0.8 else 0)

        # Evaluate 0.5
        acc_05 = np.mean(np.array(cls_label) == np.array(cls_pre_05))
        f1_05 = f1_score(cls_label, cls_pre_05, zero_division=0)
        pre_05 = precision_score(cls_label, cls_pre_05, zero_division=0)
        rec_05 = recall_score(cls_label, cls_pre_05, zero_division=0)

        # Evaluate 0.8
        acc_08 = np.mean(np.array(cls_label) == np.array(cls_pre_08))
        f1_08 = f1_score(cls_label, cls_pre_08, zero_division=0)
        pre_08 = precision_score(cls_label, cls_pre_08, zero_division=0)
        rec_08 = recall_score(cls_label, cls_pre_08, zero_division=0)

        test_info["step"].append(step)
        
        test_info["ac_05"].append(acc_05)
        test_info["f1_05"].append(f1_05)
        test_info["precision_05"].append(pre_05)
        test_info["recall_05"].append(rec_05)
        
        test_info["ac_08"].append(acc_08)
        test_info["f1_08"].append(f1_08)
        test_info["precision_08"].append(pre_08)
        test_info["recall_08"].append(rec_08)
        
        return acc_05 # Return acc_05 as the primary metric for saving best model

def save_best_record(test_info, file_path):
    with open(file_path, "w") as fo:
        fo.write(f"Step: {test_info['step'][-1]}\n")
        fo.write(f"--- Threshold 0.5 ---\n")
        fo.write(f"ac: {test_info['ac_05'][-1]:.4f}\n")
        fo.write(f"f1: {test_info['f1_05'][-1]:.4f}\n")
        fo.write(f"precision: {test_info['precision_05'][-1]:.4f}\n")
        fo.write(f"recall: {test_info['recall_05'][-1]:.4f}\n")
        fo.write(f"--- Threshold 0.8 ---\n")
        fo.write(f"ac: {test_info['ac_08'][-1]:.4f}\n")
        fo.write(f"f1: {test_info['f1_08'][-1]:.4f}\n")
        fo.write(f"precision: {test_info['precision_08'][-1]:.4f}\n")
        fo.write(f"recall: {test_info['recall_08'][-1]:.4f}\n")

if __name__ == '__main__':
    args = init_args()
    
    # Load GKF splits
    with open("gkf_splits/gkf_5fold_idx.pickle", "rb") as f:
        gkf_splits = pickle.load(f)
        
    for ft in [0, 1]:
        for stride in [8, 16]:
            print(f"========== Starting Train 5-fold (FT: {ft}, Stride: {stride}) ==========")
            
            # config setup
            lr_list = eval(args['lr'])
            num_iters = len(lr_list)
            
            best_test_dict = {
                "acc_05": [], "precision_05": [], "f1_05": [], "recall_05": [],
                "acc_08": [], "precision_08": [], "f1_08": [], "recall_08": []
            }
            
            for fold_idx in range(5):
                print(f"--- Fold {fold_idx + 1} ---")
                
                normal_train_idx = gkf_splits["Normal"][fold_idx]["train_idx"]
                normal_test_idx = gkf_splits["Normal"][fold_idx]["test_idx"]
                abnormal_train_idx = gkf_splits["Bot"][fold_idx]["train_idx"]
                abnormal_test_idx = gkf_splits["Bot"][fold_idx]["test_idx"]
                
                utils.set_seed(args['seed'] + fold_idx)
                worker_init_fn = np.random.seed(args['seed'] + fold_idx)

                net = WSAD(input_size=1024, flag="Train", a_nums=60, n_nums=60, frame_window=16)
                net = net.cuda()

                list_path = f"cs2_feat_{stride}_ft{ft}"
                
                normal_train_loader = data.DataLoader(
                    CS2_dataloader(root_dir=args['root_dir'], mode='Train', modal='RGB', num_segments=1, 
                        len_feature=1024, list_fname=list_path, idx_list=normal_train_idx, is_normal=True),
                        batch_size=64, shuffle=True, num_workers=args['num_workers'],
                        worker_init_fn=worker_init_fn, drop_last=True)
                        
                abnormal_train_loader = data.DataLoader(
                    CS2_dataloader(root_dir=args['root_dir'], mode='Train', modal='RGB', num_segments=1, 
                        len_feature=1024, list_fname=list_path, idx_list=abnormal_train_idx, is_normal=False),
                        batch_size=64, shuffle=True, num_workers=args['num_workers'],
                        worker_init_fn=worker_init_fn, drop_last=True)
                        
                # Merge test dataset manually or create a concatenated dataloader
                test_dataset_normal = CS2_dataloader(root_dir=args['root_dir'], mode='Test', modal='RGB', num_segments=args['num_segments'], 
                        len_feature=1024, list_fname=list_path, idx_list=normal_test_idx, is_normal=True)
                test_dataset_abnormal = CS2_dataloader(root_dir=args['root_dir'], mode='Test', modal='RGB', num_segments=args['num_segments'], 
                        len_feature=1024, list_fname=list_path, idx_list=abnormal_test_idx, is_normal=False)
                
                test_dataset = data.ConcatDataset([test_dataset_normal, test_dataset_abnormal])
                test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args['num_workers'], worker_init_fn=worker_init_fn)

                test_info = {
                    "step": [], 
                    "f1_05": [], "precision_05": [], "ac_05": [], "recall_05": [],
                    "f1_08": [], "precision_08": [], "ac_08": [], "recall_08": []
                }
                
                best_ac_05 = 0
                best_f1_05, best_pre_05, best_rec_05 = 0, 0, 0
                best_ac_08, best_f1_08, best_pre_08, best_rec_08 = 0, 0, 0, 0

                criterion = AD_Loss(frame_window=16)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr_list[0], betas=(0.9, 0.999), weight_decay=0.00005)

                for step in tqdm(range(1, num_iters + 1), total=num_iters, dynamic_ncols=True):
                    if step > 1 and lr_list[step - 1] != lr_list[step - 2]:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_list[step - 1]
                            
                    if (step - 1) % len(normal_train_loader) == 0:
                        normal_loader_iter = iter(normal_train_loader)
                    if (step - 1) % len(abnormal_train_loader) == 0:
                        abnormal_loader_iter = iter(abnormal_train_loader)
                        
                    train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion, step)
                    
                    if step % 50 == 0 and step > 5:
                        acc_05 = test(net, None, test_loader, test_info, step, stride)
                        
                        if test_info["ac_05"][-1] > best_ac_05:
                            best_ac_05 = test_info["ac_05"][-1]
                            best_f1_05, best_pre_05, best_rec_05 = test_info["f1_05"][-1], test_info["precision_05"][-1], test_info["recall_05"][-1]
                            best_ac_08, best_f1_08, best_pre_08, best_rec_08 = test_info["ac_08"][-1], test_info["f1_08"][-1], test_info["precision_08"][-1], test_info["recall_08"][-1]

                            os.makedirs(args['output_path'], exist_ok=True)
                            os.makedirs(args['model_path'], exist_ok=True)
                            
                            save_best_record(test_info, os.path.join(args['output_path'], f"cs2_feat_{stride}_ft{ft}_fold{fold_idx}_best_record.txt"))
                            torch.save(net.state_dict(), os.path.join(args['model_path'], f"cs2_feat_{stride}_ft{ft}_fold{fold_idx}.pkl"))
                            
                best_test_dict["acc_05"].append(best_ac_05)
                best_test_dict["precision_05"].append(best_pre_05)
                best_test_dict["f1_05"].append(best_f1_05)
                best_test_dict["recall_05"].append(best_rec_05)
                
                best_test_dict["acc_08"].append(best_ac_08)
                best_test_dict["precision_08"].append(best_pre_08)
                best_test_dict["f1_08"].append(best_f1_08)
                best_test_dict["recall_08"].append(best_rec_08)
                
            print(f"===== Train & Test Done (FT {ft}, Stride {stride}) =====")
            
            # Save 5-fold average summary
            with open(os.path.join(args['output_path'], f'best_test_dict_{stride}_ft{ft}.pickle'), 'wb') as f:
                pickle.dump(best_test_dict, f)
                
            print(f"Average Accuracy (0.5): {np.mean(best_test_dict['acc_05']):.4f}")
            print(f"Average Accuracy (0.8): {np.mean(best_test_dict['acc_08']):.4f}")