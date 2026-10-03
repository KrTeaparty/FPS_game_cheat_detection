import cv2
import os
import glob
import numpy as np
import pickle
import random

def get_mp4_path(npy_path):
    # e.g. .../Bot_130-1.npy
    basename = os.path.basename(npy_path).replace('.npy', '.mp4')
    if "Normal" in basename:
        match_dir = basename.split('-')[0]
        return rf"e:\01_Project\01_FPS_Cheat_Detection\raw_data\Preprocessed_match_data\Normal_match\{match_dir}\{basename}"
    else:
        match_dir = basename.split('-')[0]
        return rf"e:\01_Project\01_FPS_Cheat_Detection\raw_data\Preprocessed_match_data\Bot_match\{match_dir}\{basename}"

def process_video(cap, out, gt_list, is_bot, fade_frames=15):
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    total_frames = len(frames)
    
    for i, frame in enumerate(frames):
        # Apply fade out at the end
        if i >= total_frames - fade_frames:
            alpha = (total_frames - i) / fade_frames
            frame = (frame * alpha).astype(np.uint8)
            
        out.write(frame)
        gt_list.append(1 if is_bot else 0)

def main():
    base_dir = r"e:\01_Project\01_FPS_Cheat_Detection_Experiments"
    out_dir = os.path.join(base_dir, "synthetic_test_data")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load fold 4 test indices
    with open("gkf_splits/gkf_5fold_idx.pickle", "rb") as f:
        gkf_splits = pickle.load(f)
        
    normal_test_idx = gkf_splits["Normal"][4]["test_idx"]
    bot_test_idx = gkf_splits["Bot"][4]["test_idx"]
    
    with open("list/cs2_feat_8_ft0_Normal.list", "r") as f:
        normal_lines = f.readlines()
    with open("list/cs2_feat_8_ft0_Bot.list", "r") as f:
        bot_lines = f.readlines()
        
    normal_pool = [normal_lines[i].split()[0] for i in normal_test_idx]
    bot_pool = [bot_lines[i].split()[0] for i in bot_test_idx]
    
    # We will generate 20 candidates for video 3
    for c_idx in range(20):
        random.seed(100 + c_idx)
        random.shuffle(normal_pool)
        random.shuffle(bot_pool)
        
        n1 = get_mp4_path(normal_pool.pop())
        n2 = get_mp4_path(normal_pool.pop())
        b1 = get_mp4_path(bot_pool.pop())
        
        out_name = f"synth_video_3_candidate_{c_idx}"
        print(f"Creating {out_name}...")
        
        caps = [cv2.VideoCapture(n1), cv2.VideoCapture(n2), cv2.VideoCapture(b1)]
        is_bots = [False, False, True]
        
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_path = os.path.join(out_dir, f"{out_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        gt_list = []
        for cap, is_bot in zip(caps, is_bots):
            process_video(cap, out, gt_list, is_bot)
            cap.release()
            
        out.release()
        
        gt_arr = np.array(gt_list)
        np.save(os.path.join(out_dir, f"{out_name}_gt.npy"), gt_arr)

if __name__ == '__main__':
    main()
