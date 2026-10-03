import os
import glob
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from model import WSAD
from tqdm import tqdm

def get_best_fold(base_dir, stride, ft):
    best_fold = -1
    best_f1 = -1
    dict_path = os.path.join(base_dir, "outputs", f"best_test_dict_{stride}_ft{ft}.pickle")
    if not os.path.exists(dict_path):
        return 0
    with open(dict_path, "rb") as f:
        res = pickle.load(f)
    f1_scores = res["f1_05"]
    for fold_idx, f1 in enumerate(f1_scores):
        if f1 > best_f1:
            best_f1 = f1
            best_fold = fold_idx
    return best_fold

def evaluate_synth_videos():
    base_dir = r"e:\01_Project\01_FPS_Cheat_Detection_Experiments"
    synth_dir = os.path.join(base_dir, "synthetic_test_data")
    os.makedirs(os.path.join(base_dir, "synthetic_results"), exist_ok=True)
    
    configs = [
        (8, 0, "Fine-tuning X, Stride = 8"),
        (16, 0, "Fine-tuning X, Stride = 16"),
        (8, 1, "Fine-tuning O, Stride = 8"),
        (16, 1, "Fine-tuning O, Stride = 16"),
    ]
    
    # Pre-load best models
    models = {}
    for stride, ft, name in configs:
        fold = get_best_fold(base_dir, stride, ft)
        model_path = os.path.join(base_dir, "models", f"cs2_feat_{stride}_ft{ft}_fold{fold}.pkl")
        net = WSAD(input_size=1024, flag="Test", a_nums=60, n_nums=60, frame_window=16)
        net.load_state_dict(torch.load(model_path))
        net.cuda()
        net.eval()
        models[name] = (net, stride, ft)
    
    auc_results = {name: [] for _, _, name in configs}
    video_names = []
    
    for v_idx in range(1, 4):
        v_name = f"synth_video_{v_idx}"
        video_names.append(f"Video {v_idx}")
        gt_path = os.path.join(synth_dir, f"{v_name}_gt.npy")
        if not os.path.exists(gt_path):
            continue
        gt = np.load(gt_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Anomaly Frame Detection - Prediction Scores over Time", fontsize=20, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (stride, ft, config_name) in enumerate(configs):
            net, _, _ = models[config_name]
            feat_path = os.path.join(base_dir, f"cs2_test_feat_{stride}_ft{ft}", f"{v_name}.npy")
            feature = np.load(feat_path).astype(np.float32)
            
            feat_tensor = torch.tensor(feature).unsqueeze(0).cuda()
            with torch.no_grad():
                res = net(feat_tensor)
                a_predict = res["frame"].squeeze(0).cpu().numpy()
                
            frame_scores = np.repeat(a_predict, stride)
            # Ensure lengths match
            min_len = min(len(frame_scores), len(gt))
            frame_scores = frame_scores[:min_len]
            gt_matched = gt[:min_len]
            
            auc = roc_auc_score(gt_matched, frame_scores)
            auc_results[config_name].append(auc)
            
            ax = axes[idx]
            ax.plot(frame_scores, label='Prediction Data', color='#3498db')
            ax.fill_between(range(min_len), 0, gt_matched, color='#d3d3d3', alpha=0.5, label='Ground Truth Anomaly')
            ax.set_title(f"{config_name}, Anomaly Scores (AUC: {auc:.2f})", fontsize=14)
            ax.set_xlabel("Frame Number", fontsize=12)
            ax.set_ylabel("Prediction Score", fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(base_dir, "synthetic_results", f"{v_name}_subplots.png"))
        plt.close()
        
    # Plot Figure 5-4 AUC Bar Chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    y = np.arange(len(video_names))
    
    colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71'] # Changed colors to distinguish FT O/X
    
    for idx, (config_name, color) in enumerate(zip([c[2] for c in configs], colors)):
        aucs = np.array(auc_results[config_name]) * 100
        plt.barh(y - (1.5 - idx)*bar_width, aucs, bar_width, label=config_name, color=color)
        
    plt.yticks(y, video_names)
    plt.xlabel('AUC (%)')
    plt.title('AUC result of three test videos')
    plt.xlim(0, 105)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "synthetic_results", "auc_bar_chart.png"))
    plt.close()
    
    print("Done! Check synthetic_results directory.")

if __name__ == '__main__':
    evaluate_synth_videos()
