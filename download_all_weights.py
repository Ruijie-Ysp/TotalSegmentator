import os
import time
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# 配置基础 URL
BASE_URL = "https://github.com/wasserth/TotalSegmentator/releases/download"

# 权重分类定义
WEIGHTS = {
    # --- 核心模型 (Total v2) ---
    "核心-器官 (Part1 Organs)": f"{BASE_URL}/v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip",
    "核心-椎骨 (Part2 Vertebrae)": f"{BASE_URL}/v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip",
    "核心-心脏 (Part3 Cardiac)": f"{BASE_URL}/v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip",
    "核心-肌肉 (Part4 Muscles)": f"{BASE_URL}/v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip",
    "核心-肋骨 (Part5 Ribs)": f"{BASE_URL}/v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip",
    "核心-全身3mm (Total 3mm)": f"{BASE_URL}/v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip",
    "核心-全身6mm (Total 6mm)": f"{BASE_URL}/v2.0.0-weights/Dataset298_TotalSegmentator_total_6mm_1559subj.zip",
    "核心-身体 (Body)": f"{BASE_URL}/v2.0.0-weights/Dataset299_body_1559subj.zip",
    "核心-身体6mm (Body 6mm)": f"{BASE_URL}/v2.0.0-weights/Dataset300_body_6mm_1559subj.zip",

    # --- 头部与颈部 (Head & Neck v2.3.0) ---
    "头部-腺体腔室 (Head Glands)": f"{BASE_URL}/v2.3.0-weights/Dataset775_head_glands_cavities_492subj.zip",
    "头部-骨骼血管 (Head Bones/Vessels)": f"{BASE_URL}/v2.3.0-weights/Dataset776_headneck_bones_vessels_492subj.zip",
    "头部-肌肉 (Head Muscles)": f"{BASE_URL}/v2.3.0-weights/Dataset777_head_muscles_492subj.zip",
    "头部-颈部肌肉1 (HeadNeck Muscles P1)": f"{BASE_URL}/v2.3.0-weights/Dataset778_headneck_muscles_part1_492subj.zip",
    "头部-颈部肌肉2 (HeadNeck Muscles P2)": f"{BASE_URL}/v2.3.0-weights/Dataset779_headneck_muscles_part2_492subj.zip",

    # --- 其他特殊任务 ---
    "特殊-肺血管 (Lung Vessels)": f"{BASE_URL}/v2.0.0-weights/Dataset258_lung_vessels_248subj.zip",
    "特殊-胸部CT (Thorax CT)": f"{BASE_URL}/v2.0.0-weights/Dataset315_thoraxCT.zip",
    "特殊-髋关节植入物 (Hip Implant)": f"{BASE_URL}/v2.0.0-weights/Dataset260_hip_implant_71subj.zip",
    "特殊-脑部 (ICB v0)": f"{BASE_URL}/v2.0.0-weights/Dataset150_icb_v0.zip",
    "特殊-肝段 (Liver Segments)": f"{BASE_URL}/v2.5.0-weights/Dataset570_ct_liver_segments.zip",
}

def get_weights_dir():
    """获取权重存储目录 (~/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/)"""
    home = Path.home()
    weights_dir = home / ".totalsegmentator" / "nnunet" / "results" / "nnUNet" / "3d_fullres"
    return weights_dir

def download_file(url, dest_path, retries=5):
    """带重试机制和 User-Agent 的下载函数"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    for i in range(retries):
        try:
            with requests.Session() as s:
                with s.get(url, stream=True, timeout=60, headers=headers, allow_redirects=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    with open(dest_path, 'wb') as f:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"  下载中 (尝试 {i+1})") as pbar:
                            for chunk in r.iter_content(chunk_size=8192 * 32):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"\n  下载失败: {e}")
            if i < retries - 1:
                print("  5秒后重试...")
                time.sleep(5)
            else:
                return False

def main():
    weights_dir = get_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"权重将保存至: {weights_dir}")
    print("="*50)
    print("请选择下载模式:")
    print("1. 仅核心权重 (最常用, 约 5-10GB)")
    print("2. 全部权重 (包含头部、特殊任务等, 约 20GB+)")
    print("3. 自定义 (手动选择)")
    
    choice = input("请输入数字 (1/2/3): ").strip()
    
    to_download = {}
    if choice == '1':
        to_download = {k: v for k, v in WEIGHTS.items() if k.startswith("核心")}
    elif choice == '2':
        to_download = WEIGHTS
    elif choice == '3':
        for i, (name, url) in enumerate(WEIGHTS.items()):
            ans = input(f"是否下载 {name}? (y/n): ").lower()
            if ans == 'y':
                to_download[name] = url
    else:
        print("无效输入，默认选择核心权重。")
        to_download = {k: v for k, v in WEIGHTS.items() if k.startswith("核心")}

    print(f"\n准备下载 {len(to_download)} 个权重文件...")
    
    for name, url in to_download.items():
        filename = url.split('/')[-1]
        dest_path = weights_dir / filename
        
        # 提取任务目录名 (例如 Dataset291_TotalSegmentator_part1_organs_1559subj)
        task_dir_name = filename.replace(".zip", "")
        task_dir = weights_dir / task_dir_name
        
        if task_dir.exists():
            print(f"\n[已存在] {name} ({task_dir_name})，跳过下载。")
            continue
            
        print(f"\n[正在处理] {name}...")
        if download_file(url, dest_path):
            print(f"  解压中...")
            try:
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(weights_dir)
                os.remove(dest_path)
                print(f"  [完成] {name} 已就绪。")
            except Exception as e:
                print(f"  [错误] 解压失败: {e}")
        else:
            print(f"  [失败] 无法下载 {name}。")

    print("\n" + "="*50)
    print("所有任务处理完毕！")
    print(f"如果下载失败，你可以手动访问以下链接下载并解压到 {weights_dir}:")
    for name, url in to_download.items():
        print(f"- {name}: {url}")

if __name__ == "__main__":
    main()
