#!/usr/bin/env python3
"""
BERT分词工具
用于对中文文本进行BERT分词处理
"""

import argparse
import sys
import os
from transformers import BertTokenizer


def get_tokenizer(model_name="bert-base-chinese"):
    """
    获取BERT分词器，支持本地缓存和离线模式
    
    Args:
        model_name: BERT模型名称
    
    Returns:
        BertTokenizer: 分词器实例
    """
    try:
        # 首先尝试从本地缓存加载
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        tokenizer = BertTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            local_files_only=False  # 允许从网络下载
        )
        return tokenizer
    except Exception as e1:
        print(f"尝试从网络加载模型失败: {e1}")
        try:
            # 尝试仅从本地加载
            tokenizer = BertTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )
            return tokenizer
        except Exception as e2:
            print(f"从本地加载模型也失败: {e2}")
            print("请确保网络连接正常，或模型已下载到本地")
            return None


def tokenize_text(text, model_name="bert-base-chinese", show_tokens=True, show_ids=False):
    """
    使用BERT模型对文本进行分词
    
    Args:
        text: 输入文本
        model_name: BERT模型名称，默认为bert-base-chinese
        show_tokens: 是否显示分词结果
        show_ids: 是否显示token对应的ID
    
    Returns:
        dict: 包含分词结果和token ID的字典
    """
    try:
        # 获取分词器
        tokenizer = get_tokenizer(model_name)
        if tokenizer is None:
            return None
        
        # 进行分词
        tokens = tokenizer.tokenize(text)
        
        # 获取token对应的ID
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        result = {
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "model_name": model_name
        }
        
        # 输出结果
        if show_tokens:
            print(f"原始文本: {text}")
            print(f"分词结果: {' | '.join(tokens)}")
            
        if show_ids:
            print(f"Token IDs: {token_ids}")
            
        return result
        
    except Exception as e:
        print(f"分词过程中出现错误: {e}")
        return None


def detailed_unk_analysis(text, model_name="bert-base-chinese", context_window=3):
    """
    详细分析UNK token，显示原始字符和上下文
    
    Args:
        text: 输入文本
        model_name: BERT模型名称
        context_window: 上下文窗口大小
    """
    try:
        tokenizer = get_tokenizer(model_name)
        if tokenizer is None:
            return
        
        # 分词
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        unk_token_id = tokenizer.convert_tokens_to_ids("[UNK]")
        
        print(f"原始文本: {text}")
        print(f"分词结果: {' | '.join(tokens)}")
        
        # 为了更好地定位UNK字符，我们需要跟踪字符到token的映射
        # 这里使用一种简化的方法来近似定位
        char_to_token_map = []
        token_index = 0
        
        # 查找UNK tokens
        unk_found = False
        for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            if token_id == unk_token_id:
                unk_found = True
                # 显示UNK token及其上下文
                start_idx = max(0, i - context_window)
                end_idx = min(len(tokens), i + context_window + 1)
                context_tokens = tokens[start_idx:end_idx]
                
                print(f"\n发现UNK token:")
                print(f"  位置: {i}")
                print(f"  UNK token: {token}")
                print(f"  上下文: {' | '.join(context_tokens)}")
                
                # 尝试定位原始字符（使用近似方法）
                # 计算大概位置
                approx_pos = min(len(text)-1, i * 2)  # 粗略估计
                start_char = max(0, approx_pos - 5)
                end_char = min(len(text), approx_pos + 5)
                print(f"  原始文本近似位置: {text[start_char:end_char]}")
        
        if not unk_found:
            print("未发现UNK token")
            
    except Exception as e:
        print(f"详细分析过程中出现错误: {e}")


def precise_unk_analysis(text, model_name="bert-base-chinese", context_window=3):
    """
    精确分析UNK token，显示原始字符和上下文
    
    Args:
        text: 输入文本
        model_name: BERT模型名称
        context_window: 上下文窗口大小（3-5个字符）
    """
    try:
        tokenizer = get_tokenizer(model_name)
        if tokenizer is None:
            return
        
        # 确保context_window在3-5范围内
        context_window = max(3, min(5, context_window))
        
        print(f"原始文本: {text}")
        
        # 对整个文本进行tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        unk_token_id = tokenizer.convert_tokens_to_ids("[UNK]")
        
        print(f"分词结果: {' | '.join(tokens)}")
        
        # 精确分析每个UNK token
        unk_found = False
        for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            if token_id == unk_token_id:
                unk_found = True
                print(f"\n=== UNK Token 分析 ===")
                print(f"位置: {i}")
                print(f"Token: {token}")
                
                # 获取上下文tokens
                start_idx = max(0, i - context_window)
                end_idx = min(len(tokens), i + context_window + 1)
                context_tokens = tokens[start_idx:end_idx]
                
                print(f"上下文 ({context_window} tokens): {' | '.join(context_tokens)}")
                
                # 尝试通过反向tokenize来定位原始字符
                # 这是一种启发式方法，因为tokenize是不可逆的
                context_text_start = max(0, i * 2 - context_window)
                context_text_end = min(len(text), i * 2 + context_window + 5)
                context_chars = text[context_text_start:context_text_end]
                
                print(f"原始文本上下文 ({context_window}-{context_window+2} chars): '{context_chars}'")
                
                # 显示UNK字符可能的候选
                print(f"可能的原始字符（启发式猜测）:")
                # 显示上下文中的每个字符作为候选
                for j, char in enumerate(context_chars):
                    print(f"  [{j}] '{char}' (Unicode: U+{ord(char):04X})")
        
        if not unk_found:
            print("\n未发现UNK token")
        else:
            print(f"\n总共发现 {sum(1 for tid in token_ids if tid == unk_token_id)} 个UNK token")
            
    except Exception as e:
        print(f"精确分析过程中出现错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BERT中文分词工具")
    parser.add_argument("--report", type=str, help="直接输入要分词的文本")
    parser.add_argument("-f", "--file", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("-m", "--model", default="/mnt/windows/Users/Admin/LLM/models/bert-base-chinese/", 
                       help="BERT模型名称 (默认: bert-base-chinese)")
    parser.add_argument("--no-tokens", action="store_true", 
                       help="不显示分词结果")
    parser.add_argument("--show-ids", action="store_true", 
                       help="显示token对应的ID")
    parser.add_argument("--analyze-unk", action="store_true",
                       help="分析UNK token并显示上下文")
    parser.add_argument("--precise-unk", action="store_true",
                       help="精确分析UNK token（默认分析模式）")
    parser.add_argument("--context-window", type=int, default=3,
                       help="UNK token分析的上下文窗口大小 (3-5, 默认: 3)")
    
    args = parser.parse_args()
    
    # 确保context_window在有效范围内
    args.context_window = max(3, min(5, args.context_window))
    
    # 优先处理--report参数
    if args.report:
        if args.analyze_unk or args.precise_unk:
            precise_unk_analysis(args.report, args.model, args.context_window)
        else:
            tokenize_text(args.report, args.model, show_tokens=not args.no_tokens, show_ids=args.show_ids)
    elif args.file:
        # 处理文件输入
        tokenize_file(args.file, args.output, args.model, args.analyze_unk or args.precise_unk, args.context_window)
    else:
        # 如果没有提供任何参数，从标准输入读取
        print("请输入要分词的文本（按Ctrl+D结束）:")
        text = sys.stdin.read().strip()
        if text:
            if args.analyze_unk or args.precise_unk:
                precise_unk_analysis(text, args.model, args.context_window)
            else:
                tokenize_text(text, args.model, show_tokens=not args.no_tokens, show_ids=args.show_ids)


def tokenize_file(input_file, output_file=None, model_name="bert-base-chinese", analyze_unk=False, context_window=3):
    """
    对文件中的文本进行分词处理
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（可选）
        model_name: BERT模型名称
        analyze_unk: 是否分析UNK token
        context_window: 上下文窗口大小
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        results = []
        for i, text in enumerate(texts):
            text = text.strip()
            if text:  # 跳过空行
                print(f"\n=== 第{i+1}行 ===")
                if analyze_unk:
                    precise_unk_analysis(text, model_name, context_window)
                else:
                    result = tokenize_text(text, model_name)
                    if result:
                        results.append(result)
        
        # 如果指定了输出文件，保存结果
        if output_file and results and not analyze_unk:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"文本: {result['text']}\n")
                    f.write(f"分词: {' | '.join(result['tokens'])}\n")
                    f.write(f"IDs: {result['token_ids']}\n")
                    f.write("-" * 50 + "\n")
            print(f"\n分词结果已保存到: {output_file}")
            
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
    except Exception as e:
        print(f"处理文件时出现错误: {e}")


if __name__ == "__main__":
    main()