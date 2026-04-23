import ast
import pandas as pd


def build_label_mapping(csv_path):
    df = pd.read_csv(csv_path)

    token_set = set()
    for ids in df["decoding_generated_token_id"]:
        arr = ast.literal_eval(ids)
        token_set.update(arr)

    uniq = sorted(list(token_set))
    mapping = {tok: i for i, tok in enumerate(uniq)}

    print(f"总 token 分类数: {len(mapping)}")
    print(f"token id 最小值: {min(uniq)}, 最大值: {max(uniq)}")
    return mapping
