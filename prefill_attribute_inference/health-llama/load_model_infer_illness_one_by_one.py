import os
import shutil

import pandas as pd
import torch
import numpy as np

from security_paper_all_experiment_cuda1.prompt_classification.deal_data.template import illness_list_4


class LoadModelInfer(object):

    def __init__(self):
        self.npz_root = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper_all_experiment_cuda1/prompt_classification/get_all_attn/Meta-Llama-3.1-8B-Instruct/infer_10_val"
        self.illness_model_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper_all_experiment_cuda1/prompt_classification/get_all_attn/Meta-Llama-3.1-8B-Instruct/best_resnet18_not_pca_illness_num_classes4.pt"

        # 输入 val csv
        self.val_csv_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper_all_experiment_cuda1/prompt_classification/get_all_attn/Meta-Llama-3.1-8B-Instruct/infer_10_val.csv"

        # device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # load model (saved full model object)
        self.model = torch.load(
            self.illness_model_path,
            map_location=self.device,
            weights_only=False,  # PyTorch 2.6+
        )
        self.model.eval()

        self.num_classes = 4

    # ============================================================
    # normalize
    # ============================================================
    @staticmethod
    def normalize_minmax(x):
        mx = x.max()
        mn = x.min()
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    # ============================================================
    # build attn_map from attn_rank npz  -> (32,32,1787)
    # ============================================================
    def process_one_case(self, npz_path, k_top=10):
        data = np.load(npz_path)
        attn_rank = data["attn_rank"]  # (32, 32, T, 1787) or (L,H,Q,K)
        L, H, T, K = attn_rank.shape

        k_eff = min(k_top, K)
        result = np.zeros((L, H, K), dtype=np.float32)

        for i in range(L):
            for h in range(H):
                mat = attn_rank[i, h]  # (T, K)

                # pick top-k positions by "rank value" (or values) -> largest k_eff
                topk_idx = np.argpartition(mat, -k_eff, axis=-1)[..., -k_eff:]  # (T, k_eff)

                idx_flat = topk_idx.reshape(-1)
                counts = np.bincount(idx_flat, minlength=K).astype(np.float32)

                counts = self.normalize_minmax(counts)
                result[i, h] = counts

        return result  # (32,32,1787)

    # ============================================================
    # numpy -> torch input: (1, 1787, 32, 32)
    # ============================================================
    def build_input_tensor(self, attn_map_32_32_1787):
        x = torch.from_numpy(attn_map_32_32_1787).float().permute(2, 0, 1).contiguous()
        x = x.unsqueeze(0)
        return x

    # ============================================================
    # inference
    # ============================================================
    @torch.no_grad()
    def infer_one(self, x, topk=1):
        x = x.to(self.device)
        logits = self.model(x)  # (1, num_classes)
        prob = torch.softmax(logits, dim=-1)
        topk_prob, topk_idx = torch.topk(prob, k=min(topk, prob.shape[-1]), dim=-1)
        return topk_idx.squeeze(0).cpu().numpy(), topk_prob.squeeze(0).cpu().numpy()

    # ============================================================
    # main: copy npz + write new csv
    # ============================================================
    def load_infer(self, k_top=10, topk_pred=1):
        df_val = pd.read_csv(self.val_csv_path)

        results = []  # for output csv

        for i in range(df_val.shape[0]):
            one_s = df_val.iloc[i]

            case_idx = int(one_s["index"])
            query = one_s["query"] if "query" in df_val.columns else ""
            true_label = one_s["true_label"]

            src_npz = os.path.join(self.npz_root, f"case_{case_idx}", "q_k_attn_rank_topk256.npz")
            if not os.path.exists(src_npz):
                print(f"[Skip] missing: {src_npz}")
                continue

            # ----- inference -----
            attn_map = self.process_one_case(src_npz, k_top=k_top)
            x = self.build_input_tensor(attn_map)

            pred_idx, pred_prob = self.infer_one(x, topk=topk_pred)
            pred_label = illness_list_4[int(pred_idx[0])]

            # ----- collect row for new csv -----
            results.append({
                "index": case_idx,
                "query": query,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob": float(pred_prob[0]) if len(pred_prob) > 0 else None
            })

            print(f"case_{case_idx} | pred={pred_label} ({pred_prob[0]:.4f}) | true={true_label}")
            print("-----------------------------------------------------")


def main():
    load_model_infer = LoadModelInfer()
    load_model_infer.load_infer(k_top=10, topk_pred=1)


if __name__ == '__main__':
    main()
