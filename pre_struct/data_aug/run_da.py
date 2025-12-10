# -*- coding: utf-8 -*-
from __future__ import annotations
import sys

sys.path.append(".")
from conf import API_KEY
from pipeline import DataAugmentPipeline
from structs import REPORT_STRUCTURE_MAP

try:
    from model_path_conf import DEFAULT_TOKENIZER_PATH  # type: ignore
except Exception as exc:
    raise ImportError(
        "DEFAULT_TOKENIZER_PATH must be defined in model_path_conf.py"
    ) from exc

"""
字段管理 + 样本均衡（v5.0 版本）

执行的操作：
1. 补充新字段：根据 keys/keys_merged.json 为每个报告类型补充缺失的字段
2. 删除旧字段：移除不在结构映射中的多余字段
3. 样本均衡：
   - 少于平均数的类型：增强到平均数
   - 超过平均数的类型：下采样到平均数
   - 使用均值作为目标，减少极端值带来的偏差
4. 记录变更：新增的字段名单保存到 added_keys 列表中

策略：
- 使用平均数作为目标值（可更灵敏地响应全局分布）
- 对于新增报告类型，自动补充字段并增强样本
- 对于样本过多的类型，随机下采样保持平衡
"""

if __name__ == "__main__":
    import json
    import random
    import os
    from pathlib import Path
    from collections import Counter
    from datetime import datetime
    
    # ========== 配置参数 ==========
    TARGET_REPORT_TYPES = None  # None = 处理所有报告类型（包括新增类型）
    
    # 样本均衡策略
    ENABLE_BALANCING = True      # 启用样本均衡
    FILL_TO_MEAN = True          # 少于平均数的类型增强到平均数
    DOWNSAMPLE_TO_MEAN = True    # 超过平均数的类型下采样到平均数
    TOP_K_TYPES = 100            # 对所有报告类型做均衡（覆盖全部34类型+未来新增）
    SEED = 42                    # 随机种子（保证可复现）
    # 手动在此处指定“输入文件路径”（不读取环境变量/命令行）
    # 示例：
    # INPUT_PATH = Path("data/20251014/clean_ocr_ppt_da_v5_0_field_cleaned.json")
    INPUT_PATH = Path("data/20251013/clean_ocr_ppt_da_v5_0_field_cleaned.json")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    # 与输入同目录输出
    BASE_DIR = INPUT_PATH.parent
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] input file: {INPUT_PATH}")
    print(f"[init] data directory: {BASE_DIR}")

    # 平衡输出：沿用输入文件名，加 _balanced 后缀
    OUTPUT_PATH_BALANCED = INPUT_PATH.with_name(INPUT_PATH.stem + "_balanced.json")
    # 结果文件：按版本后缀命名（手动设定，默认 v5_0；可改为 v5_1）
    out_ver = "v5_0"
    OUTPUT_PATH_FINAL = BASE_DIR / f"clean_ocr_ppt_da_{out_ver}_field_added.json"
    
    # ========== 步骤1: 下采样平衡（超过平均数的类型） ==========
    if ENABLE_BALANCING and DOWNSAMPLE_TO_MEAN:
        print("=" * 60)
        print("步骤 1: 样本均衡（下采样超额类型到平均数）")
        print("=" * 60)
        
        # 读取数据
        with INPUT_PATH.open('r', encoding='utf-8') as f:
            all_data = json.load(f)

        # 清理：删除不在 keys/keys_merged.json 中“该报告类型”下的字段
        RESERVED = {"report", "report_title", "report_titles", "meta", "added_keys"}

        def _clean_one(rec: dict) -> dict:
            title = str(rec.get("report_title") or rec.get("report_titles") or "").strip()
            allowed = set(REPORT_STRUCTURE_MAP.get(title, []) or []) | RESERVED
            if not allowed:
                # 未知标题：仅保留保留字段
                allowed = RESERVED
            out = {}
            for k, v in rec.items():
                if isinstance(k, str) and k in allowed:
                    out[k] = v
            return out

        removed_total = 0
        cleaned_data = []
        for r in all_data:
            before = set(k for k in r.keys() if isinstance(k, str))
            rr = _clean_one(r)
            after = set(k for k in rr.keys() if isinstance(k, str))
            removed_total += max(0, len(before - after))
            cleaned_data.append(rr)
        if removed_total > 0:
            print(f"[clean] removed {removed_total} unknown keys not in keys_merged.json")
        all_data = cleaned_data
        
        print(f"原始数据: {len(all_data)} 条记录")
        
        # 统计各类型数量
        type_counts = Counter(r.get('report_title', '') for r in all_data)
        print(f"\n报告类型数: {len(type_counts)}")
        
        # 计算均值
        counts = list(type_counts.values())
        mean_target = int(round(sum(counts) / max(1, len(counts))))
        counts_sorted = sorted(counts)
        
        print(f"\n样本分布统计:")
        print(f"  最小值: {min(counts)}")
        print(f"  平均值(目标): {mean_target}")
        print(f"  中位数(参考): {counts_sorted[len(counts_sorted) // 2]}")
        print(f"  最大值: {max(counts)}")
        
        # 显示需要下采样的类型
        need_downsample = {t: c for t, c in type_counts.items() if c > mean_target}
        if need_downsample:
            print(f"\n需要下采样的类型（{len(need_downsample)} 个）:")
            for t, c in sorted(need_downsample.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {t}: {c} → {mean_target} (减少 {c - mean_target})")
        
        # 按类型分组
        rng = random.Random(SEED)
        by_type = {}
        for rec in all_data:
            title = rec.get('report_title', '')
            if title not in by_type:
                by_type[title] = []
            by_type[title].append(rec)
        
        # 下采样
        balanced_data = []
        for title, records in by_type.items():
            if len(records) > mean_target:
                # 下采样到平均数
                sampled = rng.sample(records, mean_target)
                balanced_data.extend(sampled)
                print(f"  下采样: {title} {len(records)} → {len(sampled)}")
            else:
                # 保持原样
                balanced_data.extend(records)
        
        print(f"\n平衡后数据: {len(balanced_data)} 条记录")
        print(f"减少: {len(all_data) - len(balanced_data)} 条")
        
        # 保存平衡后的数据
        with OUTPUT_PATH_BALANCED.open('w', encoding='utf-8') as f:
            json.dump(balanced_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 已保存平衡数据到: {OUTPUT_PATH_BALANCED}")
        print("=" * 60)
        print()
        
        # 使用平衡后的数据作为输入
        input_for_pipeline = OUTPUT_PATH_BALANCED
    else:
        input_for_pipeline = INPUT_PATH
    
    # ========== 步骤2: Pipeline 处理（字段补充 + 上采样少数类到平均数） ==========
    print("=" * 60)
    print("步骤 2: 字段补充 + 上采样少数类到平均数")
    print("=" * 60)
    print()
    
    pipe = DataAugmentPipeline(
        # 路径 - v5.0 第一阶段：字段补充 + 样本均衡
        in_path=str(input_for_pipeline),
        out_path=str(OUTPUT_PATH_FINAL),
        # 数据增强控制参数
        target_report_types=TARGET_REPORT_TYPES,  # None = 处理所有类型
        max_samples_per_type=None,  # 由均衡策略自动计算
        # 字段管理
        inc_synthesize_new_keys=True,  # 补充缺失的新字段
        inc_and_synthesize_missing_to_median=ENABLE_BALANCING and FILL_TO_MEAN,  # 结合均衡策略
        fill_stat="mean",  # 使用平均数作为目标
        # 样本均衡
        topk_to_median=ENABLE_BALANCING,  # 内部目标已根据 fill_stat 调整为平均数
        topk_titles=TOP_K_TYPES,  # 处理前20个报告类型
        augment_minority_only=False,  # 不仅处理少数类
        k_per_record=0,  # 每条记录增强次数（由pipeline自动计算）
        struct_perturb_enable=False,  # 关闭结构扰动增强
        synthesize_missing_titles=False,  # 不合成新标题
        # LLM配置
        use_openai=True,
        openai_base=[
            "https://qwen3.yoo.la/v1/",
            "http://123.57.234.67:8000/v1",
        ],
        openai_model="qwen3-32b",
        api_key=API_KEY,
        timeout=60,
        include_context_summary=False,
        # RAG配置
        rag_path="../data/rag/ocr_summary_words.txt",  # 使用现有的RAG语料文件
        rag_topk=3,
        tokenizer_name=DEFAULT_TOKENIZER_PATH,
        exhaustive_titles=False,
        exhaustive_modules=False,
        seed=SEED,
        # 并发配置 - 优化性能
        reports_workers=6,          # 样本级并发数（降低避免过载）
        fields_workers=3,           # 字段级并发数（单样本内字段并发）
        per_base_pool_maxsize=48,   # 连接池大小
        per_base_max_inflight=24,   # 最大并发请求数
        # 字段合成控制 - 优化批量处理
        synth_batch_size=8,     # 单批字段数量 (一次生成更多字段)
        synth_keys_drop_prob=0.0,
        synth_min_keys=4,
        synth_max_keys=10,       # 最大字段数
        # 新增字段控制
        inc_max_keys_per_record=3,  # 每条记录最多补充4个字段
        inc_key_pick="random",  # 随机选择字段
        # 合成模式
        fields_synth_mode="batched",  # 批量模式更快
    )
    
    print("\n" + "=" * 60)
    print("开始 Pipeline 处理...")
    print("=" * 60)
    stats = pipe.run()
    
    print("\n" + "=" * 60)
    print("✅ 所有步骤完成！")
    print("=" * 60)
    print(f"\n最终输出: {OUTPUT_PATH_FINAL}")
    print(f"\n统计信息:")
    print(stats)
