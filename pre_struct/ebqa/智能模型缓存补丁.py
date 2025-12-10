#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EBQA 智能模型缓存补丁（带自动过期和健康检查）

功能：
1. 模型缓存（避免重复加载）
2. 自动过期清理（长期不用自动卸载）
3. 健康检查（确保模型有效）
4. 手动清理接口

使用方法：
    python3 智能模型缓存补丁.py

作者：AI Assistant
日期：2025-10-30
"""

import os
import sys
from pathlib import Path

def apply_smart_cache_patch():
    """应用智能模型缓存补丁"""
    
    script_dir = Path(__file__).parent
    target_file = script_dir / "test_ebqa.py"
    
    if not target_file.exists():
        print(f"❌ 错误：找不到文件 {target_file}")
        sys.exit(1)
    
    # 读取文件
    content = target_file.read_text(encoding="utf-8")
    
    # 检查是否已经有智能缓存
    if "ModelCacheEntry" in content:
        print("✅ 智能模型缓存已存在，无需重复应用")
        return
    
    # 检查是否有旧的简单缓存
    if "_EBQA_MODEL_CACHE" in content:
        print("🔄 检测到旧的简单缓存，将升级为智能缓存...")
        # 备份
        backup_file = target_file.with_suffix(".py.backup_smart")
        backup_file.write_text(content, encoding="utf-8")
        print(f"📦 创建备份：{backup_file}")
        
        # 删除旧的缓存代码
        lines = content.split('\n')
        new_lines = []
        skip_block = False
        
        for i, line in enumerate(lines):
            # 跳过旧的缓存块
            if "# ==================== 模型缓存机制 ====================" in line:
                skip_block = True
                continue
            
            if skip_block:
                # 找到缓存块结束位置（空行后的 def load_ebqa）
                if line.strip().startswith("def load_ebqa"):
                    skip_block = False
                    new_lines.append(line)
                continue
            
            # 修改 predict_for 函数
            if "model, collate, _ = _get_cached_ebqa_model(cfg)" in line:
                line = line.replace("_get_cached_ebqa_model", "_get_smart_cached_model")
            
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
    else:
        # 全新添加
        backup_file = target_file.with_suffix(".py.backup")
        if not backup_file.exists():
            backup_file.write_text(content, encoding="utf-8")
            print(f"📦 创建备份：{backup_file}")
    
    # 智能缓存代码
    smart_cache_code = '''
# ==================== 智能模型缓存机制 ====================
# 添加日期：2025-10-30
# 功能：自动过期清理 + 健康检查
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelCacheEntry:
    """模型缓存条目"""
    model: Any
    collate: Any
    device: Any
    last_access_time: float
    cache_key: Tuple


# 全局缓存
_EBQA_MODEL_CACHE = {}
_cache_lock = threading.Lock()

# 配置（可通过环境变量调整）
CACHE_TTL_SECONDS = int(os.environ.get("EBQA_MODEL_CACHE_TTL", "1800"))  # 默认30分钟
AUTO_CLEANUP_INTERVAL = int(os.environ.get("EBQA_CACHE_CLEANUP_INTERVAL", "300"))  # 5分钟检查一次


def _check_model_health(entry: ModelCacheEntry) -> bool:
    """检查模型是否健康（是否还在内存/GPU中）"""
    try:
        # 检查模型是否还在设备上
        if hasattr(entry.model, 'model'):
            device = next(entry.model.model.parameters()).device
            return device == entry.device
        return True
    except Exception as e:
        logger = _get_logger()
        logger.warning(f"[ModelCache] 健康检查失败: {e}")
        return False


def _cleanup_expired_cache():
    """清理过期的缓存"""
    with _cache_lock:
        now = time.time()
        expired_keys = []
        
        for key, entry in _EBQA_MODEL_CACHE.items():
            # 检查是否过期
            if now - entry.last_access_time > CACHE_TTL_SECONDS:
                expired_keys.append(key)
            # 检查健康状态
            elif not _check_model_health(entry):
                expired_keys.append(key)
        
        if expired_keys:
            logger = _get_logger()
            for key in expired_keys:
                logger.info(f"[ModelCache] 清理过期/无效缓存: {key[:2]}...")
                del _EBQA_MODEL_CACHE[key]
                
                # 尝试清理 GPU 显存
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            
            logger.info(f"[ModelCache] 已清理 {len(expired_keys)} 个缓存条目")


# 后台清理线程
_cleanup_thread = None
_cleanup_running = False


def _start_auto_cleanup():
    """启动自动清理线程"""
    global _cleanup_thread, _cleanup_running
    
    if _cleanup_thread is not None and _cleanup_thread.is_alive():
        return
    
    def cleanup_worker():
        logger = _get_logger()
        logger.info(f"[ModelCache] 启动自动清理线程（TTL={CACHE_TTL_SECONDS}秒，检查间隔={AUTO_CLEANUP_INTERVAL}秒）")
        
        while _cleanup_running:
            time.sleep(AUTO_CLEANUP_INTERVAL)
            _cleanup_expired_cache()
    
    _cleanup_running = True
    _cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    _cleanup_thread.start()


def _get_smart_cached_model(cfg: PredictConfig):
    """获取缓存的模型（智能版本，带过期和健康检查）
    
    Args:
        cfg: 预测配置
        
    Returns:
        (model, collate, device): 缓存的模型、collator 和设备
    """
    cache_key = (
        cfg.model_dir,
        cfg.tokenizer_name,
        cfg.batch_size,
        cfg.max_seq_len,
    )
    
    logger = _get_logger()
    
    with _cache_lock:
        # 检查缓存是否存在且有效
        if cache_key in _EBQA_MODEL_CACHE:
            entry = _EBQA_MODEL_CACHE[cache_key]
            
            # 健康检查
            if _check_model_health(entry):
                # 更新访问时间
                entry.last_access_time = time.time()
                logger.info(f"[ModelCache] ✅ 使用缓存模型（上次访问：{int(time.time() - entry.last_access_time)}秒前）")
                return entry.model, entry.collate, entry.device
            else:
                # 健康检查失败，删除缓存
                logger.warning(f"[ModelCache] ⚠️ 缓存模型无效，重新加载")
                del _EBQA_MODEL_CACHE[cache_key]
        
        # 缓存不存在或无效，重新加载
        logger.info(f"[ModelCache] 首次加载模型: {cfg.model_dir}")
        model, collate, device = load_ebqa(cfg)
        
        # 创建缓存条目
        entry = ModelCacheEntry(
            model=model,
            collate=collate,
            device=device,
            last_access_time=time.time(),
            cache_key=cache_key
        )
        
        _EBQA_MODEL_CACHE[cache_key] = entry
        logger.info(f"[ModelCache] 模型已缓存（TTL={CACHE_TTL_SECONDS}秒）")
        
        # 确保清理线程运行
        _start_auto_cleanup()
        
        return model, collate, device


def clear_ebqa_model_cache():
    """手动清理所有模型缓存"""
    global _cleanup_running
    
    with _cache_lock:
        logger = _get_logger()
        count = len(_EBQA_MODEL_CACHE)
        
        if count > 0:
            logger.info(f"[ModelCache] 手动清理 {count} 个缓存模型")
            _EBQA_MODEL_CACHE.clear()
            
            # 清理 GPU 显存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("[ModelCache] GPU 显存已清理")
            except Exception as e:
                logger.warning(f"[ModelCache] GPU 显存清理失败: {e}")
        
        # 停止清理线程
        _cleanup_running = False


def get_cache_info():
    """获取缓存信息"""
    with _cache_lock:
        now = time.time()
        cache_details = []
        
        for key, entry in _EBQA_MODEL_CACHE.items():
            idle_time = int(now - entry.last_access_time)
            ttl_remaining = max(0, CACHE_TTL_SECONDS - idle_time)
            cache_details.append({
                "key": str(key[:2]),
                "idle_seconds": idle_time,
                "ttl_remaining": ttl_remaining,
                "healthy": _check_model_health(entry)
            })
        
        return {
            "cached_models": len(_EBQA_MODEL_CACHE),
            "ttl_seconds": CACHE_TTL_SECONDS,
            "auto_cleanup_running": _cleanup_running,
            "details": cache_details
        }


'''
    
    # 查找插入位置（在 load_ebqa 之前）
    load_ebqa_marker = "def load_ebqa(cfg: PredictConfig):"
    if load_ebqa_marker in content:
        content = content.replace(load_ebqa_marker, smart_cache_code + load_ebqa_marker)
    else:
        print("❌ 错误：找不到 load_ebqa 函数")
        sys.exit(1)
    
    # 修改 predict_for 使用新的缓存函数
    content = content.replace(
        "model, collate, _ = load_ebqa(cfg)",
        "model, collate, _ = _get_smart_cached_model(cfg)"
    ).replace(
        "model, collate, _ = _get_cached_ebqa_model(cfg)",
        "model, collate, _ = _get_smart_cached_model(cfg)"
    )
    
    # 写回文件
    target_file.write_text(content, encoding="utf-8")
    
    print(f"\n{'='*70}")
    print("✅ 智能模型缓存补丁应用成功！")
    print(f"{'='*70}")
    print(f"\n🎯 新功能：")
    print(f"  ✅ 自动过期：1800秒（30分钟）无访问自动卸载")
    print(f"  ✅ 健康检查：确保模型在内存/GPU中有效")
    print(f"  ✅ 后台清理：每300秒自动检查过期缓存")
    print(f"  ✅ 手动清理：调用 clear_ebqa_model_cache() 立即清理")
    print(f"\n📝 环境变量配置（可选）：")
    print(f"  export EBQA_MODEL_CACHE_TTL=1800      # 缓存过期时间（秒）")
    print(f"  export EBQA_CACHE_CLEANUP_INTERVAL=300 # 清理检查间隔（秒）")
    print(f"\n📊 查看缓存状态：")
    print(f"  from test_ebqa import get_cache_info")
    print(f"  print(get_cache_info())")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    try:
        apply_smart_cache_patch()
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

