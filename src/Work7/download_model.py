"""
SMPL 模型下载辅助脚本
========================
尝试从多个公开来源自动下载 SMPL_NEUTRAL.pkl。
如果自动下载失败，请手动从师大云盘或 SMPL 官网下载。
"""
import os
import sys
import requests

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'SMPL_NEUTRAL.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

# 已知的公开镜像（可能随时失效）
MIRRORS = [
    # HuggingFace Spaces / Datasets
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/SMPL_NEUTRAL.pkl",
    "https://huggingface.co/camenduru/3DMPPE/resolve/main/main/SMPL_NEUTRAL.pkl",
    # 其他可能来源
    "https://media.githubusercontent.com/media/russoale/hmr2.0/master/model/neutral_smpl_with_cocoplus_reg.pkl",
]


def download_from_url(url, output_path, timeout=60):
    """从指定 URL 下载文件"""
    print(f"Trying: {url}")
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        if r.status_code != 200:
            print(f"  Failed (HTTP {r.status_code})")
            return False
        # 检查内容长度（SMPL pkl 约 140MB）
        content_length = r.headers.get('Content-Length')
        if content_length and int(content_length) < 1000000:
            print(f"  File too small ({content_length} bytes), skipping")
            return False

        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        print(f"  Downloaded {downloaded} bytes -> {output_path}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists: {MODEL_PATH}")
        print(f"Size: {os.path.getsize(MODEL_PATH)} bytes")
        return 0

    print("SMPL_NEUTRAL.pkl not found. Attempting auto-download...\n")

    for url in MIRRORS:
        if download_from_url(url, MODEL_PATH):
            print("\n[Success] Model downloaded!")
            return 0

    print("\n" + "=" * 60)
    print("Auto-download failed. Please download manually:")
    print("=" * 60)
    print("1. 师大云盘（推荐）: 课程资料中查找 SMPL_NEUTRAL.pkl")
    print("2. SMPL 官网: https://smpl.is.tue.mpg.de/")
    print("   - 注册账号 -> Downloads -> SMPL_python_v.1.1.0.zip")
    print("   - 解压后将 basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    print("     重命名为 SMPL_NEUTRAL.pkl")
    print(f"3. 放置到: {MODEL_PATH}")
    print("=" * 60)
    return 1


if __name__ == '__main__':
    sys.exit(main())
