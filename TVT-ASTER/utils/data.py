# -*- coding: utf-8 -*-
# utils/data.py
import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ========== 路径解析 ==========
# 脚本位于 项目根/utils/data.py -> 项目根 = __file__/../
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------------
# 旧：HockeyFight 专用的列文件函数（保留以兼容）
# -------------------------
def list_videos(input_dir: str):
    """列出 data/HockeyFight 下的所有 .avi 文件，并按文件名前缀归类。"""
    input_dir = Path(input_dir)
    files = [f.name for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".avi"]
    fight = [f for f in files if f.lower().startswith("fi")]
    nonfight = [f for f in files if f.lower().startswith("no")]
    return fight, nonfight

# -------------------------
# 新：通用按规则列文件函数（兼容 RLVS / 更多数据集）
# -------------------------
def list_videos_by_rule(
    input_dir,
    pos_prefixes=("fi",),
    neg_prefixes=("no",),
    extensions=(".avi",),
    precedence=None,
):
    """
    通用文件枚举与按前缀归类：
      - pos_prefixes: 正类前缀（元组/列表）
      - neg_prefixes: 负类前缀（元组/列表）
      - extensions  : 允许的扩展名（大小写不敏感）
      - precedence  : 前缀匹配优先级（列表），如 ["nv", "v"] 避免 NV 被误判为 V
    返回：pos_list, neg_list （文件名，不含路径）
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return [], []

    ext_set = set(e.lower() for e in extensions)
    pos_list, neg_list = [], []

    pos_prefixes = tuple(p.lower() for p in pos_prefixes)
    neg_prefixes = tuple(p.lower() for p in neg_prefixes)
    precedence = [p.lower() for p in precedence] if precedence else []

    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        fl = f.name.lower()
        if f.suffix.lower() not in ext_set:
            continue

        # 按优先级匹配
        label = None
        if precedence:
            for p in precedence:
                if fl.startswith(p):
                    label = 0 if p in neg_prefixes else (1 if p in pos_prefixes else None)
                    break
        else:
            if any(fl.startswith(p) for p in neg_prefixes):
                label = 0
            elif any(fl.startswith(p) for p in pos_prefixes):
                label = 1

        if label is None:
            continue
        (pos_list if label == 1 else neg_list).append(f.name)

    return pos_list, neg_list

def stratified_split(fnames, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """对某一类别的文件名进行分层划分（返回 train/val/test 三个列表）。"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    fnames = fnames[:]  # copy
    rng.shuffle(fnames)

    n = len(fnames)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_split = fnames[:n_train]
    val_split = fnames[n_train:n_train+n_val]
    test_split = fnames[n_train+n_val:]
    return train_split, val_split, test_split

def make_splits_balanced(input_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    【保持原行为】分别对 fight / nonfight 做划分，再合并，保证三子集中类别平衡。
    仅用于 HockeyFight（fi*/no*，.avi）。
    """
    fight, nonfight = list_videos(input_dir)
    tr_f, va_f, te_f = stratified_split(fight, train_ratio, val_ratio, test_ratio, seed)
    tr_n, va_n, te_n = stratified_split(nonfight, train_ratio, val_ratio, test_ratio, seed)

    splits = {
        "train": sorted(tr_f + tr_n),
        "val":   sorted(va_f + va_n),
        "test":  sorted(te_f + te_n),
    }
    label_map = {f: 1 for f in tr_f + va_f + te_f}
    label_map.update({f: 0 for f in tr_n + va_n + te_n})
    return splits, label_map

def make_splits_balanced_from_lists(
    pos_list, neg_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
):
    tr_pos, va_pos, te_pos = stratified_split(pos_list, train_ratio, val_ratio, test_ratio, seed)
    tr_neg, va_neg, te_neg = stratified_split(neg_list, train_ratio, val_ratio, test_ratio, seed)

    splits = {
        "train": sorted(tr_pos + tr_neg),
        "val":   sorted(va_pos + va_neg),
        "test":  sorted(te_pos + te_neg),
    }
    label_map = {f: 1 for f in tr_pos + va_pos + te_pos}
    label_map.update({f: 0 for f in tr_neg + va_neg + te_neg})
    return splits, label_map

def ensure_dir(path: Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def count_total_frames(cap: cv2.VideoCapture) -> int:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        return total
    # 兜底逐帧计数
    cnt = 0
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        cnt += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return cnt

def sample_indices(num_frames_available: int, target_frames: int = 32):
    """等间隔采样索引；若不足则用最后一帧补齐。"""
    if num_frames_available <= 0:
        return []
    if num_frames_available >= target_frames:
        idx = np.linspace(0, num_frames_available - 1, target_frames)
        return [int(round(i)) for i in idx]
    else:
        idx = list(range(num_frames_available))
        while len(idx) < target_frames:
            idx.append(num_frames_available - 1)
        return idx

def read_frame_at(cap: cv2.VideoCapture, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def preprocess_clip(frames_bgr, size=224):
    """frames_bgr: list of HxWx3 (BGR). 返回 (3, T, size, size) 的 float32 tensor，做 ImageNet 标准化。"""
    processed = []
    for img in frames_bgr:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        processed.append(arr)
    arr = np.stack(processed, axis=0)  # (T, H, W, 3)
    arr = np.transpose(arr, (3, 0, 1, 2))  # (3, T, H, W)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).float()
    return tensor

def convert_one_video_to_pt(in_path, out_path, label: int, num_frames_target=32, size=224):
    in_path = str(in_path)
    out_path = str(out_path)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"[WARN] 打开视频失败，跳过: {in_path}")
        return False

    total = count_total_frames(cap)
    if total <= 0:
        print(f"[WARN] 无有效帧，跳过: {in_path}")
        cap.release()
        return False

    idx = sample_indices(total, num_frames_target)
    frames = []
    for i in idx:
        f = read_frame_at(cap, i)
        if f is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(i + 1, total - 1))
            ok, f2 = cap.read()
            if not ok:
                f2 = frames[-1] if frames else np.zeros((size, size, 3), dtype=np.uint8)
            frames.append(f2)
        else:
            frames.append(f)
    cap.release()

    clip = preprocess_clip(frames, size=size)  # (3,T,H,W)
    pkg = {
        "video": clip,
        "label": int(label),
        "filename": os.path.basename(in_path)
    }
    torch.save(pkg, out_path)
    return True

# -------------------------
# 数据集处理通用流程（封装）
# -------------------------
def process_dataset(
    name: str,
    input_dir,
    output_dir,
    pos_prefixes,
    neg_prefixes,
    extensions,
    precedence=None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    num_frames_target=32,
    resize_size=224,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        print(f"[WARN] 数据集 {name}: 输入目录不存在，跳过 -> {input_dir}")
        return

    pos, neg = list_videos_by_rule(
        input_dir=input_dir,
        pos_prefixes=pos_prefixes,
        neg_prefixes=neg_prefixes,
        extensions=extensions,
        precedence=precedence,
    )

    if len(pos) == 0 or len(neg) == 0:
        print(f"[WARN] 数据集 {name}: 类别不完整（pos={len(pos)}, neg={len(neg)}），将继续但无法保证平衡。")

    splits, label_map = make_splits_balanced_from_lists(
        pos, neg, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    # 创建输出目录
    for split in ["train", "val", "test"]:
        ensure_dir(output_dir / split)

    # 统计信息
    def count_by_label_map(file_list, label_map):
        pos_n = sum(1 for f in file_list if label_map.get(f, -1) == 1)
        neg_n = sum(1 for f in file_list if label_map.get(f, -1) == 0)
        return pos_n, neg_n

    print(f"\n===== [{name}] 数据划分统计（尽量类别平衡）=====")
    for split in ["train", "val", "test"]:
        pos_n, neg_n = count_by_label_map(splits[split], label_map)
        print(f"{split:5s}: 总数={len(splits[split])}, pos={pos_n}, neg={neg_n}")

    # 转换并保存
    for split in ["train", "val", "test"]:
        file_list = splits[split]
        out_dir_split = output_dir / split
        print(f"\n[{name} - {split}] 转换与保存 .pt （输出到 {out_dir_split}）")
        for fname in tqdm(file_list, ncols=80):
            in_path = input_dir / fname
            base = Path(fname).stem
            out_path = out_dir_split / f"{base}.pt"
            label = label_map.get(fname, None)
            if label is None:
                print(f"[WARN] 未找到标签，跳过: {fname}")
                continue
            if out_path.exists():
                continue
            ok = convert_one_video_to_pt(
                in_path=in_path,
                out_path=out_path,
                label=label,
                num_frames_target=num_frames_target,
                size=resize_size
            )
            if not ok:
                continue

    # 最终汇总
    print(f"\n===== [{name}] 生成完成，总结 =====")
    for split in ["train", "val", "test"]:
        out_split_dir = output_dir / split
        n_pt = len([f for f in out_split_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pt"]) if out_split_dir.is_dir() else 0
        pos_n = sum(1 for f in splits[split] if label_map.get(f, -1) == 1)
        neg_n = sum(1 for f in splits[split] if label_map.get(f, -1) == 0)
        print(f"{split:5s}: 预期样本={len(splits[split])}（pos={pos_n}, neg={neg_n}），实际生成 .pt = {n_pt}")

# -------------------------
# 新增：处理前检测是否已有成品 .pt 数据集
# -------------------------
def is_pt_dataset_ready(output_dir: Path, splits=("train", "val", "test"), require_nonempty: bool = True):
    """
    检测 output_dir 下是否存在按要求生成的 pt 数据集目录结构。
    判断标准：
      - output_dir 存在
      - 每个 split 子目录存在
      - 若 require_nonempty=True，则每个 split 至少包含 1 个 .pt 文件
    返回：(ready: bool, detail: dict)
    """
    output_dir = Path(output_dir)
    detail = {}
    if not output_dir.is_dir():
        return False, detail

    ready = True
    for sp in splits:
        sp_dir = output_dir / sp
        if not sp_dir.is_dir():
            ready = False
            detail[sp] = 0
            continue
        n_pt = len([f for f in sp_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pt"])
        detail[sp] = n_pt
        if require_nonempty and n_pt == 0:
            ready = False

    return ready, detail

def main():
    # ======== 全局参数 ========
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    seed = 42
    num_frames_target = 32
    resize_size = 224

    set_seed(seed)

    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] DATA_ROOT    = {DATA_ROOT}")

    # ======== 数据集配置（基于项目根/data） ========
    HF = {
        "name": "HockeyFight",
        "input_dir": DATA_ROOT / "HockeyFight",
        "output_dir": DATA_ROOT / "pt-HockeyFight",
        "pos_prefixes": ("fi",),
        "neg_prefixes": ("no",),
        "extensions": (".avi",),
        "precedence": ["no", "fi"],
    }
    RLVS = {
        "name": "RLVS",
        "input_dir": DATA_ROOT / "RLVS",
        "output_dir": DATA_ROOT / "pt-RLVS",
        "pos_prefixes": ("v",),
        "neg_prefixes": ("nv",),
        "extensions": (".avi", ".mp4"),
        "precedence": ["nv", "v"],  # 先匹配 NV 再匹配 V
    }

    PROCESS_HOCKEY = True
    PROCESS_RLVS   = True

    if PROCESS_HOCKEY:
        ready, detail = is_pt_dataset_ready(HF["output_dir"])
        if ready:
            print(f"[SKIP] 检测到 {HF['name']} 已处理：{HF['output_dir']}")
            print(f"       .pt 数量（train/val/test）= {detail.get('train',0)}/{detail.get('val',0)}/{detail.get('test',0)}")
        else:
            process_dataset(
                name=HF["name"],
                input_dir=HF["input_dir"],
                output_dir=HF["output_dir"],
                pos_prefixes=HF["pos_prefixes"],
                neg_prefixes=HF["neg_prefixes"],
                extensions=HF["extensions"],
                precedence=HF["precedence"],
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                seed=seed, num_frames_target=num_frames_target, resize_size=resize_size
            )

    if PROCESS_RLVS:
        ready, detail = is_pt_dataset_ready(RLVS["output_dir"])
        if ready:
            print(f"[SKIP] 检测到 {RLVS['name']} 已处理：{RLVS['output_dir']}")
            print(f"       .pt 数量（train/val/test）= {detail.get('train',0)}/{detail.get('val',0)}/{detail.get('test',0)}")
        else:
            process_dataset(
                name=RLVS["name"],
                input_dir=RLVS["input_dir"],
                output_dir=RLVS["output_dir"],
                pos_prefixes=RLVS["pos_prefixes"],
                neg_prefixes=RLVS["neg_prefixes"],
                extensions=RLVS["extensions"],
                precedence=RLVS["precedence"],
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                seed=seed, num_frames_target=num_frames_target, resize_size=resize_size
            )

if __name__ == "__main__":
    main()
