import os
import ast
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset_for_model_filter import build_label_mapping_with_filter


# ============================================================
# Model: must match your latest training structure
# backbone -> dropout -> fc -> dropout -> logits
# ============================================================
class ResNetWithDropout(nn.Module):
    def __init__(self, model_name: str, in_channels: int, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        model_name = model_name.lower()
        assert model_name in {"resnet18", "resnet34", "resnet50"}

        if model_name == "resnet18":
            backbone = torchvision.models.resnet18(weights=None)
        elif model_name == "resnet34":
            backbone = torchvision.models.resnet34(weights=None)
        else:
            backbone = torchvision.models.resnet50(weights=None)

        backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        out = self.fc(feat)
        out = self.dropout(out)
        return out


# ============================================================
# Inference Runner (aligned with your training)
# ============================================================
class InferenceRunner:
    def __init__(self,
                 freq_csv_path: str,
                 eval_csv_path: str,
                 big_npy_path: str,
                 caseid_path: str,
                 ckpt_path: str,

                 model_name: str,
                 prompt_token_num: int,
                 topk: int,
                 dropout_p: float,
                 min_freq: int = 0,

                 chunk_T: int = 50,
                 device: str = "cuda:0",
                 use_channels_last: bool = False,

                 tokenizer_path: str = None,
                 npy_mmap: bool = True):

        assert chunk_T > 0
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.model_name = model_name.lower()
        assert self.model_name in {"resnet18", "resnet34", "resnet50"}
        self.prompt_token_num = int(prompt_token_num)
        self.topk = int(topk)
        self.chunk_T = int(chunk_T)
        self.use_channels_last = bool(use_channels_last)

        # ===== label map & keep set =====
        print("===== Build label mapping (with filter) =====")
        self.label_map, self.keep_set, _ = build_label_mapping_with_filter(
            freq_csv_path=freq_csv_path,
            min_freq=min_freq
        )
        self.num_classes = len(self.label_map)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        print(f"num_classes={self.num_classes}, min_freq={min_freq}")

        # ===== tokenizer (optional) =====
        self.tokenizer = None
        if tokenizer_path is not None:
            print("===== Load tokenizer =====")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # ===== eval csv =====
        print("===== Load eval CSV =====")
        self.df = pd.read_csv(eval_csv_path)
        assert "index" in self.df.columns, "eval_csv must contain column: index"
        assert "decoding_generated_token_id" in self.df.columns, "eval_csv must contain column: decoding_generated_token_id"

        # ✅ pick query column
        # 默认优先用 'query'，没有的话尝试常见列名兜底
        candidate_cols = ["query", "prompt", "question", "input", "text"]
        self.query_col = None
        for c in candidate_cols:
            if c in self.df.columns:
                self.query_col = c
                break
        if self.query_col is None:
            print("[Warn] No query-like column found in eval_csv. Output 'query' will be empty.")
        else:
            print(f"Using query column: {self.query_col}")

        # ===== big npy =====
        print("===== Load big npy =====")
        self.big = np.load(big_npy_path, mmap_mode="r" if npy_mmap else None)
        print("big shape:", self.big.shape, "dtype:", self.big.dtype)

        # ===== caseid mapping (align big row) =====
        print("===== Load case id list =====")
        case_ids_in_big = np.load(caseid_path, allow_pickle=True)

        def norm_case_id(x):
            try:
                return int(x)
            except Exception:
                return str(x)

        self.case2row = {norm_case_id(cid): i for i, cid in enumerate(case_ids_in_big)}

        # ===== build model & load ckpt =====
        print("===== Build model and load ckpt =====")
        self.model = ResNetWithDropout(
            model_name=self.model_name,
            in_channels=self.prompt_token_num,
            num_classes=self.num_classes,
            dropout_p=dropout_p
        ).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.use_amp = (self.device.type == "cuda")
        print(f"topk={self.topk}, prompt_token_num={self.prompt_token_num}")
        print(f"chunk_T={self.chunk_T}, channels_last={self.use_channels_last}, amp={self.use_amp}")

    def decode_tokens(self, token_ids):
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def ensure_TKHW(self, x_idx_np: np.ndarray) -> np.ndarray:
        if x_idx_np.ndim != 4:
            raise ValueError(f"x_idx must be 4D, got {x_idx_np.shape}")

        # (T, K, 32, 32)
        if x_idx_np.shape[1] == self.topk and x_idx_np.shape[2] == 32 and x_idx_np.shape[3] == 32:
            return x_idx_np

        # (T, 32, 32, K)
        if x_idx_np.shape[1] == 32 and x_idx_np.shape[2] == 32 and x_idx_np.shape[3] == self.topk:
            return np.transpose(x_idx_np, (0, 3, 1, 2))

        raise ValueError(f"Unrecognized layout {x_idx_np.shape}, expect (T,K,32,32) or (T,32,32,K)")

    def idx_to_onehot_gpu(self, x_idx: torch.Tensor) -> torch.Tensor:
        t, K, H, W = x_idx.shape
        x = torch.zeros((t, self.prompt_token_num, H, W), device=x_idx.device, dtype=torch.float16)
        x.scatter_(1, x_idx, 1.0)
        return x

    @torch.no_grad()
    def infer_T(self, x_idx_full: torch.Tensor) -> np.ndarray:
        T = x_idx_full.shape[0]
        preds = []

        for s in range(0, T, self.chunk_T):
            e = min(s + self.chunk_T, T)
            x_idx = x_idx_full[s:e]

            x = self.idx_to_onehot_gpu(x_idx)
            if self.use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x)

            preds.append(logits.argmax(dim=1).cpu())
            del x, logits

        return torch.cat(preds, dim=0).numpy()

    def run(self, save_csv: str):
        rows = []

        skipped_no_big = 0
        skipped_empty = 0
        skipped_len_mismatch = 0
        skipped_bad_index_range = 0
        skipped_layout_error = 0

        for i in tqdm(range(len(self.df)), desc="Inference", ncols=120):
            r = self.df.iloc[i]
            case_id = r["index"]

            # ✅ extract query
            query_text = ""
            if self.query_col is not None:
                v = r[self.query_col]
                query_text = "" if (pd.isna(v)) else str(v)

            # locate in big
            try:
                key = int(case_id)
            except Exception:
                key = str(case_id)

            if key not in self.case2row:
                skipped_no_big += 1
                continue
            big_row = self.case2row[key]

            # parse token list
            try:
                real_ids = ast.literal_eval(r["decoding_generated_token_id"])
            except Exception:
                skipped_empty += 1
                continue

            keep_pos = [j for j, t in enumerate(real_ids) if t in self.keep_set]
            kept_T = len(keep_pos)
            if kept_T == 0:
                skipped_empty += 1
                continue

            real_ids_kept = [real_ids[j] for j in keep_pos]
            real_cls_kept = np.array([self.label_map[tok] for tok in real_ids_kept], dtype=np.int64)

            # fetch x_idx from big
            x_idx_np = self.big[big_row][keep_pos]
            try:
                x_idx_np = self.ensure_TKHW(x_idx_np)
            except Exception:
                skipped_layout_error += 1
                continue

            mx = int(np.max(x_idx_np))
            if mx >= self.prompt_token_num:
                skipped_bad_index_range += 1
                continue

            x_idx = torch.from_numpy(np.array(x_idx_np, copy=False)).long().to(self.device, non_blocking=True)
            pred_cls_kept = self.infer_T(x_idx)

            # strict length check
            if pred_cls_kept.shape[0] != kept_T:
                skipped_len_mismatch += 1
                continue

            pred_ids_kept = [self.inv_label_map[int(c)] for c in pred_cls_kept]
            if len(pred_ids_kept) != len(real_ids_kept):
                skipped_len_mismatch += 1
                continue

            correct = int((pred_cls_kept == real_cls_kept).sum())
            acc_kept = correct / kept_T

            rows.append({
                "case_id": case_id,
                "query": query_text,  # ✅ 新增字段
                "kept_T": kept_T,
                "acc_kept": acc_kept,
                "real_text_kept": self.decode_tokens(real_ids_kept),
                "pred_text_kept": self.decode_tokens(pred_ids_kept),
            })
            print(self.decode_tokens(real_ids_kept))
            print(self.decode_tokens(pred_ids_kept))

        df_out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        df_out.to_csv(save_csv, index=False, encoding="utf-8-sig")

        print(f"\nSaved to: {save_csv}")
        print(f"Total cases: {len(self.df)}")
        print(f"Written rows: {len(df_out)}")
        print(f"Skipped (not in big): {skipped_no_big}")
        print(f"Skipped (kept_T==0 or parse fail): {skipped_empty}")
        print(f"Skipped (layout error): {skipped_layout_error}")
        print(f"Skipped (x_idx >= prompt_token_num): {skipped_bad_index_range}")
        print(f"Skipped (len mismatch): {skipped_len_mismatch}")

        if len(df_out) > 0:
            print(f"Mean acc_kept = {df_out['acc_kept'].mean():.6f}")
        else:
            print("Mean acc_kept = N/A (no rows)")

        return df_out


def main():
    freq_csv_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper/decoding_classification/train_classification_for_decoding/split_train_val/financial_llama_decoding_50.csv"
    val_csv_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper/decoding_classification/train_classification_for_decoding/split_train_val/financial_llama_decoding_50.csv"

    BIG_NPY_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper/decoding_classification/get_all_attn_decoding/financial-llama-decoding-128_attn_topk128.npy"
    CASEID_PATH_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper/decoding_classification/get_all_attn_decoding/financial-llama-decoding-128_attn_topk128_case_ids.npy"

    model_name = "resnet34"
    topk = 128
    prompt_token_num = 294
    dropout_p = 0.

    ckpt_dir = f"./financial-llama-decoding_topk64_resnet34"
    ckpt_path = os.path.join(ckpt_dir, f"best_{model_name}_decoding.pt")

    tokenizer_path = "/home/model_and_data/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # tokenizer_path = "/home/model_and_data/model/Qwen/Qwen3-8B"

    runner = InferenceRunner(
        freq_csv_path=freq_csv_path,
        eval_csv_path=val_csv_path,
        big_npy_path=BIG_NPY_path,
        caseid_path=CASEID_PATH_path,
        ckpt_path=ckpt_path,

        model_name=model_name,
        prompt_token_num=prompt_token_num,
        topk=topk,
        dropout_p=dropout_p,

        min_freq=0,
        chunk_T=50,
        device="cuda:0",
        use_channels_last=False,

        tokenizer_path=tokenizer_path,
        npy_mmap=True,
    )

    out_csv = os.path.join(ckpt_dir, f"val_pred_topk{topk}_{model_name}_prompt{prompt_token_num}_with_query.csv")
    runner.run(out_csv)


if __name__ == "__main__":
    main()
