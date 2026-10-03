import os
import glob
import numpy as np
from sklearn.model_selection import GroupKFold
import pickle

def main():
    base_dir = r"e:\01_Project\01_FPS_Cheat_Detection_Experiments"
    os.makedirs(os.path.join(base_dir, "list"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "gkf_splits"), exist_ok=True)
    
    finetune_matches = [f"Normal_{i:03d}" for i in range(1, 11)] + [f"Bot_{i:03d}" for i in range(1, 11)]
    
    for ft in [0, 1]:
        for stride in [8, 16]:
            feature_dir = os.path.join(base_dir, f"cs2_feat_{stride}_ft{ft}")
            
            for label, class_name in [("Normal", "Normal_match"), ("Bot", "Bot_match")]:
                list_file_path = os.path.join(base_dir, "list", f"cs2_feat_{stride}_ft{ft}_{label}.list")
                
                # 피처 파일 검색
                npy_files = glob.glob(os.path.join(feature_dir, class_name, "*.npy"))
                
                with open(list_file_path, "w") as f:
                    for npy_path in npy_files:
                        # npy_path 뒤에 임의로 " 16" (num_frames 의미, 기존 코드 형식 맞춤) 붙임
                        f.write(f"{npy_path} 16\n")
                        
            print(f"Generated .list files for Stride {stride} FT {ft}")
            
    # GKF 생성 (cs2_feat_16_ft0 기준으로 하나만 만들어도 인덱스는 동일하므로 재사용 가능)
    # 인덱스 순서가 glob 결과에 따라 달라질 수 있으므로, 파일 리스트의 인덱스를 추출
    split_info = {}
    for label in ["Normal", "Bot"]:
        list_file_path = os.path.join(base_dir, "list", f"cs2_feat_16_ft0_{label}.list")
        
        with open(list_file_path, 'r') as f:
            lines = f.readlines()
            
        paths = [line.strip().split()[0] for line in lines]
        match_ids = [os.path.basename(p).split('-')[0] for p in paths]
        
        # 파인튜닝용 매치가 리스트에 있다면 제외해야 하지만,
        # Feature Extraction 과정에서 애초에 파인튜닝 매치는 추출을 건너뛰었음!
        # 따라서 paths 안에는 파인튜닝용 매치가 없어야 정상임.
        for m_id in match_ids:
            assert m_id not in finetune_matches, f"Error: Finetune match {m_id} found in features!"
            
        paths = np.array(paths)
        match_ids = np.array(match_ids)
        
        print(f"[{label}] Total valid clips for CV: {len(paths)}, Unique matches: {len(np.unique(match_ids))}")
        
        gkf = GroupKFold(n_splits=5)
        fold_splits = []
        
        for train_idx, test_idx in gkf.split(paths, groups=match_ids):
            fold_splits.append({
                "train_idx": train_idx,
                "test_idx": test_idx
            })
            
            # 검증
            train_matches = set(match_ids[train_idx])
            test_matches = set(match_ids[test_idx])
            assert len(train_matches.intersection(test_matches)) == 0, "Leakage detected!"
            
        split_info[label] = fold_splits
        
    with open(os.path.join(base_dir, "gkf_splits", "gkf_5fold_idx.pickle"), "wb") as f:
        pickle.dump(split_info, f)
        
    print("GroupKFold splits generated successfully based on new features!")

if __name__ == "__main__":
    main()
