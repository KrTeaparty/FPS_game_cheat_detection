import os
from model import *
from train import *
from cs_test import test
from tqdm import tqdm
import pdb
import numpy as np
import torch.utils.data as data
import utils
from utils import *
import os
from options import init_args
import matplotlib.pyplot as plt
import pandas as pd
import pickle

class CS2_dataloader(data.DataLoader):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, list_fname, train_idx=None, test_idx=None, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature

        if self.mode == "Train":
            data_path = os.path.join('list','{}_{}.list'.format(list_fname, 'Normal' if is_normal is True else 'Bot'))
            data_file = open(data_path, 'r')
            self.vid_list = []
            for line in data_file:
                self.vid_list.append(line.split())
            data_file.close()
            
            self.vid_list = [self.vid_list[idx] for idx in train_idx]

        elif self.mode == "Test":
            self.vid_list = []
            for i, v in enumerate(["Normal", "Bot"]):
                data_path = os.path.join('list', '{}_{}.list'.format(list_fname, v))
                data_file = open(data_path, 'r')
                tmp_list = []
                for line in data_file:
                    tmp_list.append(line.split())
                data_file.close()
                
                tmp_list = [tmp_list[idx] for idx in test_idx[i]]
                self.vid_list.extend(tmp_list)

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data,label,name = self.get_data(index)
            return data,label,name
        else:
            data,label = self.get_data(index)
            return data,label

    def get_data(self, index):
        vid_info = self.vid_list[index][0]  
        name = vid_info.split("/")[-1].split(".npy")[0]
        video_feature = np.load(vid_info).astype(np.float32)   
        if "Normal" in vid_info.split("/")[-2]:
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = int)
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

from sklearn.metrics import f1_score, precision_score, recall_score

def test(net, config, test_loader, test_info, step, stride, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_predict = None
        
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        name_l = []
        
        for i in range(len(test_loader.dataset)):
            _data, _label, _name = next(load_iter)
            name_l.append(_name)
            
            _data = _data.cuda()
            _label = _label.cuda()
            
            res = net(_data)   
            a_predict = res["frame"]
            cls_label.append(int(_label))
            a_predict = a_predict.mean(0).cpu().numpy()
            
            cls_pre.append(1 if a_predict.max()>0.9 else 0)          
            fpre_ = np.repeat(a_predict, stride)
            if frame_predict is None:         
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])  
    
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        f1 = f1_score(cls_label, cls_pre)
        precision = precision_score(cls_label, cls_pre, zero_division=1)
        recall = recall_score(cls_label, cls_pre)

        test_info["step"].append(step)
        test_info["f1"].append(f1)
        test_info["precision"].append(precision)
        test_info["recall"].append(recall)
        test_info["ac"].append(accuracy)
        test_data_name = dict(zip(name_l, cls_label))
        return test_data_name



if __name__ == '__main__':
    for ft in [0, 1]:
        if ft:
            num_ab = 2131
            num_n = 1449
        else:
            num_ab = 2252
            num_n = 1741

        class Config(object):
            def __init__(self, args):
                self.root_dir = args['root_dir']
                self.modal = args['modal']
                self.lr = eval(args['lr'])
                self.num_iters = len(self.lr)    
                self.len_feature = 1024 
                self.batch_size = args['batch_size']
                self.model_path = args['model_path']
                self.output_path = args['output_path']
                self.num_workers = args['num_workers']
                self.model_file = args['model_file']
                self.seed = args['seed']
                self.num_segments = args['num_segments']
                self.num_abnormal = num_ab - 1
                self.num_normal = num_n - 1

        for stride in [8, 16]:
            args = init_args()
            config = Config(args)
            worker_init_fn = None
            gpus = [0]
            torch.cuda.set_device('cuda:{}'.format(gpus[0]))
            if config.seed >= 0:
                utils.set_seed(config.seed)
                worker_init_fn = np.random.seed(config.seed)

            abnormal_shuffled = np.random.permutation(config.num_abnormal)
            normal_shuffled = np.random.permutation(config.num_normal)
            abnormal_test_num = math.ceil(config.num_abnormal / 5) # 5-fold의 테스트 세트 크기
            abnormal_train_num = config.num_abnormal - abnormal_test_num * 1 # 각 폴드에서 사용되는 훈련 세트의 (최대) 크기
            normal_test_num = math.ceil(config.num_normal / 5)
            normal_train_num = config.num_normal - normal_test_num * 1
            best_test_dict = {
                "acc": [],
                "precision": [],
                "f1": [],
                "recall": [],
            }
            print(f"Abnormal_train_num: {abnormal_train_num}")
            print(f"Abnormal_test_num: {abnormal_test_num}")
            print(f"Normal_train_num: {normal_train_num}")
            print(f"Normal_test_num: {normal_test_num}")
            for i in range(5):
                start_ab = i * abnormal_test_num
                end_ab = (i + 1) * abnormal_test_num
                abnormal_test_idx = abnormal_shuffled[start_ab:end_ab]
                abnormal_train_idx = np.concatenate([abnormal_shuffled[:start_ab], abnormal_shuffled[end_ab:]])

                start_n = i * normal_test_num
                end_n = (i + 1) * normal_test_num
                normal_test_idx = normal_shuffled[start_n:end_n]
                normal_train_idx = np.concatenate([normal_shuffled[:start_n], normal_shuffled[end_n:]])

                # 시드 설정
                utils.set_seed(config.seed+i)
                worker_init_fn = np.random.seed(config.seed+i)

                # 모델 초기화
                config.len_feature = 1024
                net = WSAD(config.len_feature, flag = "Train", a_nums = 60, n_nums = 60, frame_window=16)
                net = net.cuda()

                # 데이터로더 초기화
                list_path = f"cs2_feat_{stride}_{'ft1' if ft else 'ft0'}"
                normal_train_loader = data.DataLoader(
                    CS2_dataloader(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 1, 
                        len_feature = config.len_feature, list_fname=list_path, train_idx = normal_train_idx, is_normal = True),
                        batch_size = 64,
                        shuffle = True, num_workers = config.num_workers,
                        worker_init_fn = worker_init_fn, drop_last = True)
                abnormal_train_loader = data.DataLoader(
                    CS2_dataloader(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 1, 
                        len_feature = config.len_feature, list_fname=list_path, train_idx = abnormal_train_idx, is_normal = False),
                        batch_size = 64,
                        shuffle = True, num_workers = config.num_workers,
                        worker_init_fn = worker_init_fn, drop_last = True)
                test_loader = data.DataLoader(
                    CS2_dataloader(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, 
                        len_feature = config.len_feature, list_fname=list_path, test_idx = [normal_test_idx, abnormal_test_idx], is_normal=True),
                        batch_size = 1,
                        shuffle = False, num_workers = config.num_workers,
                        worker_init_fn = worker_init_fn)


                # 지표, criterion, optimizer 초기화
                test_info = {"step": [], "f1": [],"precision":[],"ac":[], "recall":[]}
                
                best_ac = 0
                best_precision, best_recall, best_f1 = 0, 0, 0
                cur_iter = 1
                best_iter = 0

                criterion = AD_Loss(frame_window=16)

                optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
                    betas = (0.9, 0.999), weight_decay = 0.00005)

                # 학습
                for step in tqdm(
                    range(1, config.num_iters + 1),
                    total = config.num_iters,
                    dynamic_ncols = True
                ):
                    if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = config.lr[step - 1]
                    if (step - 1) % len(normal_train_loader) == 0:
                        normal_loader_iter = iter(normal_train_loader)

                    if (step - 1) % len(abnormal_train_loader) == 0:
                        abnormal_loader_iter = iter(abnormal_train_loader)
                    train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion, step)
                    if step % 5 == 0 and step > 5:
                        dt_name = test(net, config, test_loader, test_info, step, stride)
                        with open(f'test_data_name_{stride}_ft{ft}.pickle', 'wb') as f:#
                            pickle.dump(dt_name, f)#
                        if test_info["ac"][-1] > best_ac:
                            best_ac = test_info["ac"][-1]
                            best_iter = cur_iter
                            best_f1, best_precision, best_recall = test_info["f1"][-1], test_info["precision"][-1], test_info["recall"][-1]

                            save_best_record(test_info, 
                                os.path.join(config.output_path, "cs2_feat_{}_ft{}_best_record_{}.txt".format(stride, ft, config.seed)))

                            torch.save(net.state_dict(), os.path.join(args['model_path'], \
                                "cs2_feat_{}_ft{}_{}.pkl".format(stride, ft, config.seed)))
                        if step == config.num_iters:
                            torch.save(net.state_dict(), os.path.join(args['model_path'], \
                                "cs2_feat_{}_ft{}_{}.pkl".format(stride, ft, step)))
                    cur_iter += 1
                
                best_test_dict["acc"].append(best_ac)
                best_test_dict["precision"].append(best_precision)
                best_test_dict["f1"].append(best_f1)
                best_test_dict["recall"].append(best_recall)
            print("===== Train & Test Done! =====")
            print(best_test_dict)
            with open(f'best_test_dict_{stride}_ft{ft}.pickle', 'wb') as f:
                pickle.dump(best_test_dict, f)