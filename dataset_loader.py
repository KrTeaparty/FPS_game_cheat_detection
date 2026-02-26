import torch
import torch.utils.data as data
import os
import numpy as np
import utils 


class CS_dataloader(data.DataLoader):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, train_idx=None, test_idx=None, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        # data_path = os.path.join('list','Cs_Clip_{}.list'.format('Normal' if is_normal is True else 'Abnormal'))
        # data_file = open(data_path, 'r')
        # # split_path = os.path.join('list','Cs_Clip_{}.list'.format(self.mode))
        # # split_file = open(split_path, 'r')
        # self.vid_list = []
        # for line in data_file:
        #     self.vid_list.append(line.split())
        # data_file.close()

        if self.mode == "Train":
            data_path = os.path.join('list','Cs_Clip_{}.list'.format('Normal' if is_normal is True else 'Abnormal'))
            data_file = open(data_path, 'r')
            self.vid_list = []
            for line in data_file:
                self.vid_list.append(line.split())
            data_file.close()

            self.vid_list = [self.vid_list[idx] for idx in train_idx]
            # if is_normal is True:
            #     self.vid_list = self.vid_list[1801:] # Train과 Test의 경계
            # elif is_normal is False:
            #     self.vid_list = self.vid_list[:1801]
            # else:
            #     assert (is_normal == None)
            #     print("Please sure is_normal=[True/False]")
            #     self.vid_list=[]
        elif self.mode == "Test":
            self.vid_list = []
            for i, v in enumerate(["Normal", "Abnormal"]):
                data_path = os.path.join('list', 'Cs_Clip_{}.list'.format(v))
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
        # print(vid_info.split("/")[-1])
        if "Normal" in vid_info.split("/")[-1]:
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