import os
import shutil
import pandas as pd


def main():
    # 源数据路径
    case_path = "/home/model_and_data/copy_data_from_3090/wanjie/security_paper/decoding_classification/get_all_attn_decoding/qwen_financial_decoding"

    # 输出路径设置
    out_csv_path = "top10_acc_kept.csv"
    out_dir_path = "./top10_cases_output"  # 在当前文件夹下新建的目录名

    # 读取原始 CSV
    df = pd.read_csv("val_pred_topk128_resnet34_prompt293_with_query.csv")
    print(f"原始数据维度: {df.shape}")

    # 1. 取 acc_kept 最大的前 10 条数据
    # 使用 nlargest 比 sort_values().head() 更高效直观
    top10_df = df.nlargest(10, 'acc_kept')

    # 2. 保存成新的 CSV
    top10_df.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已成功保存 Top 10 数据至: {out_csv_path}")

    # 3. 新建文件夹用于存放拷贝的数据
    os.makedirs(out_dir_path, exist_ok=True)
    print(f"✅ 已创建目标文件夹: {out_dir_path}\n")

    # 4. 遍历这 10 条数据，从 case_path 中拷贝对应的数据
    print("开始拷贝文件...")
    for _, row in top10_df.iterrows():
        # 假设你的 CSV 中用来对应文件夹的列名是 'index'，请根据实际情况修改
        case_idx = int(row["case_id"])

        # 拼接源路径和目标路径
        src_case = os.path.join(case_path, f"case_{case_idx}")
        dst_case = os.path.join(out_dir_path, f"case_{case_idx}")

        if os.path.exists(src_case):
            # 判断源路径是文件夹还是单个文件
            if os.path.isdir(src_case):
                # 如果是文件夹，使用 copytree (dirs_exist_ok=True 防止目标已存在报错)
                shutil.copytree(src_case, dst_case, dirs_exist_ok=True)
            else:
                # 如果是单文件，直接 copy2
                shutil.copy2(src_case, dst_case)

            print(f"拷贝成功: case_{case_idx} (acc_kept: {row['acc_kept']:.4f})")
        else:
            print(f"[警告] 找不到对应的源数据: {src_case}")

    print("\n🎉 所有任务处理完成！")


if __name__ == '__main__':
    main()
