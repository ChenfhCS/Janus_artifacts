import os
import shutil

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from security_paper_all_experiment_cuda1.deal_data.template import illness_list_4


# ============================================================
# 构建专用的 Dataset 类，专门用于后台多进程加载和预处理 npz
# ============================================================
class NPZDataset(Dataset):
    def __init__(self, df, npz_root, k_top=10):
        self.k_top = k_top
        self.valid_data = []

        # 初始化时预先筛选出存在的文件，避免在循环中产生异常
        for i in range(df.shape[0]):
            one_s = df.iloc[i]
            case_idx = int(one_s["index"])
            src_npz = os.path.join(npz_root, f"case_{case_idx}", "q_k_attn_rank_topk256.npz")
            if os.path.exists(src_npz):
                self.valid_data.append((one_s, src_npz))
            else:
                print(f"[Skip] missing: {src_npz}")

    def __len__(self):
        return len(self.valid_data)

    @staticmethod
    def normalize_minmax(x):
        mx = x.max()
        mn = x.min()
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    def __getitem__(self, idx):
        one_s, src_npz = self.valid_data[idx]
        case_idx = int(one_s["index"])
        query = one_s["query"] if "query" in one_s else ""
        true_label = one_s["Illness"]

        # 1. Load data
        data = np.load(src_npz)
        attn_rank = data["attn_rank"]  # (32, 32, T, 1787)
        L, H, T, K = attn_rank.shape
        k_eff = min(self.k_top, K)

        # 2. 【核心提速：向量化】一次性在底层 C 语言中对整个张量进行 top-k 划分，消除 1024 次循环的调用开销
        topk_idx = np.argpartition(attn_rank, -k_eff, axis=-1)[..., -k_eff:]  # (L, H, T, k_eff)

        result = np.zeros((L, H, K), dtype=np.float32)

        # 3. 现在的循环内只有纯粹的打平、bincount 和归一化，速度极快
        for i in range(L):
            for h in range(H):
                idx_flat = topk_idx[i, h].reshape(-1)
                counts = np.bincount(idx_flat, minlength=K).astype(np.float32)
                result[i, h] = self.normalize_minmax(counts)

        # 4. 转为 Torch Tensor: shape (1787, 32, 32)
        x = torch.from_numpy(result).float().permute(2, 0, 1).contiguous()

        # 返回处理好的 Tensor 和元数据
        return x, case_idx, query, true_label, src_npz


class LoadModelInfer(object):

    def __init__(self):
        self.npz_root_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper_all_experiment_cuda1/prompt_classification/get_all_attn/llama3-8B-Instruct/financial_4_category_100"
        self.illness_model_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper_all_experiment_cuda1/prompt_classification/get_all_attn/llama3-8B-Instruct/best_resnet18_not_pca_illness_num_classes4.pt"

        # 输入 val csv
        self.val_csv_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper_all_experiment_cuda1/prompt_classification/get_all_attn/financial_qa_csv/financial_qa_100.csv"

        # 输出：拷贝后的 npz 目录 + 预测结果 csv
        self.out_npz_root = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper_all_experiment_cuda1/prompt_classification/get_all_attn/llama3-8B-Instruct/infer_10_val"
        self.out_csv_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper_all_experiment_cuda1/prompt_classification/get_all_attn/llama3-8B-Instruct/infer_10_val.csv"

        # device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # load model
        self.model = torch.load(
            self.illness_model_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.eval()

        self.num_classes = 4

    # ============================================================
    # inference (依然保持 batch_size=1 的逻辑)
    # ============================================================
    @torch.no_grad()
    def infer_one(self, x, topk=1):
        x = x.to(self.device)
        logits = self.model(x)  # (1, num_classes)
        prob = torch.softmax(logits, dim=-1)
        topk_prob, topk_idx = torch.topk(prob, k=min(topk, prob.shape[-1]), dim=-1)
        return topk_idx.squeeze(0).cpu().numpy(), topk_prob.squeeze(0).cpu().numpy()

    # ============================================================
    # main: 使用 DataLoader 多进程读取 + 筛选 Top 10 (保证类别多样性)
    # ============================================================
    def load_infer(self, k_top=10, topk_pred=1, num_workers=4):
        df_val = pd.read_csv(self.val_csv_path)

        os.makedirs(self.out_npz_root, exist_ok=True)

        all_results = []

        # 实例化我们写的 Dataset
        dataset = NPZDataset(df_val, self.npz_root_path, k_top=k_top)

        # 使用 DataLoader 开启多进程并行读取和处理。
        # DataLoader batch_size=1 时，会自动增加一个维度，所以返回的 x 形状是 (1, 1787, 32, 32)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        print(f"总计有效样本: {len(dataset)}，开始多进程高速推理 (Workers={num_workers})...")

        for batch_data in dataloader:
            # 解析 DataLoader 返回的单一 batch 数据
            x, case_idx_tensor, query_tuple, true_label_tuple, src_npz_tuple = batch_data

            # 从 tuple/tensor 中提取实际数值
            case_idx = case_idx_tensor.item()
            query = query_tuple[0]
            true_label = true_label_tuple[0]
            src_npz = src_npz_tuple[0]

            # 推理
            pred_idx, pred_prob = self.infer_one(x, topk=topk_pred)
            pred_label = illness_list_4[int(pred_idx[0])]
            prob_value = float(pred_prob[0]) if len(pred_prob) > 0 else 0.0

            all_results.append({
                "index": case_idx,
                "query": query,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob": prob_value,
                "src_npz": src_npz
            })

            print(f"推理完成: case_{case_idx} | pred={pred_label} ({prob_value:.4f}) | true={true_label}")

        print("\n-----------------------------------------------------")
        print("开始筛选 10 条数据（保证每个预测类别尽量都有）...")

        # 1. 全局按 pred_prob 降序排序
        all_results.sort(key=lambda x: x["pred_prob"], reverse=True)

        top10_results = []
        seen_labels = set()

        # 第一轮：遍历按概率排好序的数据，每个预测类别先挑一个最确信的
        for item in all_results:
            if item["pred_label"] not in seen_labels:
                top10_results.append(item)
                seen_labels.add(item["pred_label"])
            # 类别都集齐了就可以提前结束第一轮
            if len(seen_labels) == self.num_classes:
                break

        # 第二轮：如果第一轮挑完还不到 10 个，就从剩下没挑过的数据里按概率补齐
        for item in all_results:
            if len(top10_results) >= 10:
                break
            # 避免把第一轮已经加进去的重复添加
            if item not in top10_results:
                top10_results.append(item)

        # 再次按概率降序排序一下，保证最终输出的列表整洁有序
        top10_results.sort(key=lambda x: x["pred_prob"], reverse=True)

        csv_rows = []

        # 遍历选出的 10 条数据进行拷贝和 CSV 生成
        for item in top10_results:
            case_idx = item["index"]
            src_npz = item["src_npz"]

            # ----- copy npz -----
            dst_case_dir = os.path.join(self.out_npz_root, f"case_{case_idx}")
            os.makedirs(dst_case_dir, exist_ok=True)
            dst_npz = os.path.join(dst_case_dir, "q_k_attn_rank_topk256.npz")
            shutil.copy2(src_npz, dst_npz)

            csv_rows.append({
                "index": item["index"],
                "query": item["query"],
                "true_label": item["true_label"],
                "pred_label": item["pred_label"],
                "pred_prob": item["pred_prob"]
            })

            print(f"🌟 入选: case_{case_idx} | 类别: {item['pred_label']} | 概率: {item['pred_prob']:.4f}")

        # write csv
        out_df = pd.DataFrame(csv_rows, columns=["index", "query", "true_label", "pred_label", "pred_prob"])
        out_df.to_csv(self.out_csv_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ Done. 已将选出的 10 个 npz 文件拷贝至: {self.out_npz_root}")
        print(f"✅ 已将选出的 10 条数据写入 csv 文件: {self.out_csv_path}")


def main():
    load_model_infer = LoadModelInfer()
    # 你可以根据 CPU 核心数调整 num_workers，一般设置为 4 到 8 效果最佳
    load_model_infer.load_infer(k_top=10, topk_pred=1, num_workers=4)


if __name__ == '__main__':
    main()
