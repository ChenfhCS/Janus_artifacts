import os
import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
import torchvision.models as models
from transformers import AutoTokenizer

from build_label_mapping import build_label_mapping


class InferenceRunnerNPZ1734(object):
    def __init__(self,
                 all_csv,
                 val_csv,
                 npz_dir,  # ✅ 原始 npz 目录：top32_binary
                 model_path,
                 tokenizer_path,
                 key="attn_topk",
                 vocab_dim=1734):

        self.all_csv = all_csv
        self.val_csv = val_csv
        self.npz_dir = npz_dir
        self.model_path = model_path
        self.key = key
        self.vocab_dim = vocab_dim

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # 1) label map
        print("Loading label map...")
        self.label_map = build_label_mapping(self.all_csv)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)

        # 2) tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 3) model (必须与训练一致：in_channels=1734)
        print("Loading model...")
        self.model = self._load_model(model_path)

        # 4) val csv
        print("Loading validation CSV...")
        self.df_val = pd.read_csv(self.val_csv)

    def _load_model(self, model_path):
        model = models.resnet18(weights=None)

        model.conv1 = nn.Conv2d(
            in_channels=self.vocab_dim,  # ✅ 1734
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        return model

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def infer_one_case(self, attn_topk_mask):
        """
        attn_topk_mask: np.ndarray (300,32,32,1734)  0/1 uint8/float
        return: pred_token_ids len=300
        """
        # (token,L,H,C) -> (token,C,L,H)
        x = torch.from_numpy(attn_topk_mask).permute(0, 3, 1, 2)

        # 为了和训练一致：输入是 float（训练里 scatter 后 float16；这里直接 float16 更接近）
        x = x.to(dtype=torch.float16)

        x = x.to(self.device, non_blocking=True)
        x = x.contiguous(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            logits = self.model(x)  # (300,num_classes)

        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        pred_ids = [self.inv_label_map[i] for i in pred_idx]
        return pred_ids

    def run(self, save_csv="decoding_val_predictions_npz1734.csv"):
        results = []

        for i in tqdm(range(len(self.df_val)), desc="Running inference", ncols=120):
            row = self.df_val.iloc[i]
            case_id = row["index"]

            real_ids = ast.literal_eval(row["decoding_generated_token_id"])

            npz_path = os.path.join(self.npz_dir, f"case_{case_id}.npz")
            if not os.path.exists(npz_path):
                # 有些 val 可能不在目录里，跳过或记录
                continue

            with np.load(npz_path) as z:
                attn = z[self.key]  # (300,32,32,1734)

            # 确保是 0/1
            if attn.dtype != np.uint8 and attn.dtype != np.bool_:
                # 如果你存的是 float32 0/1，也没问题
                pass

            pred_ids = self.infer_one_case(attn)

            correct = sum(int(real_ids[j] == pred_ids[j]) for j in range(300))
            acc = correct / 300.0

            assert len(real_ids) == 300
            assert len(pred_ids) == 300

            results.append({
                "case_id": case_id,
                "query": row["query"],
                "real_text": self.decode_tokens(real_ids),
                "pred_text": self.decode_tokens(pred_ids),
                "real_text_token_id_list": real_ids,
                "pred_text_token_id_list": pred_ids,
                "acc": acc,
            })

        df = pd.DataFrame(results)
        df.to_csv(save_csv, index=False)
        print(f"Saved to {save_csv}")
        print(f"Mean Accuracy = {df['acc'].mean():.4f}")
        return df


def main():
    runner = InferenceRunnerNPZ1734(
        all_csv="./all_data/decoding_results_contrast_page_filtered.csv",
        val_csv="./health_queries_val_20.csv",
        npz_dir="./infer_20_val",
        model_path="./token_predictor.pt",
        tokenizer_path="/home/model_and_data/model/lmsys/longchat-7b-v1.5-32k",
        key="attn_topk",
        vocab_dim=1734,
    )
    runner.run("health_queries_val_20_with_pred.csv")


if __name__ == "__main__":
    main()
