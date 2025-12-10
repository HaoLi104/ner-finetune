# BERT分词工具使用说明

## 功能简介
这是一个基于BERT模型的中文分词工具，可以对输入的中文文本进行分词处理，并显示分词结果和对应的token ID。

## 使用方法

### 1. 使用--report参数直接输入文本（推荐）
```bash
python bert_tokenizer.py --report "我爱自然语言处理"
```

### 2. 显示分词结果和Token IDs
```bash
python bert_tokenizer.py --report "患者主诉头痛发热三天" --show-ids
```

### 3. 从文件读取文本
```bash
python bert_tokenizer.py -f input.txt -o output.txt
```

### 4. 从标准输入读取
```bash
python bert_tokenizer.py
```
然后输入文本，按Ctrl+D结束。

## 参数说明

- `--report`: 直接输入要分词的文本（主要使用方式）
- `-f, --file`: 输入文件路径
- `-o, --output`: 输出文件路径（可选）
- `-m, --model`: BERT模型名称（默认: bert-base-chinese）
- `--no-tokens`: 不显示分词结果
- `--show-ids`: 显示token对应的ID

## 示例输出

### 基本分词
```bash
$ python bert_tokenizer.py --report "我爱自然语言处理"
原始文本: 我爱自然语言处理
分词结果: 我 | 爱 | 自 | 然 | 语 | 言 | 处 | 理
```

### 显示Token IDs
```bash
$ python bert_tokenizer.py --report "医学文本分词示例" --show-ids
原始文本: 医学文本分词示例
分词结果: 医 | 学 | 文 | 本 | 分 | 词 | 示 | 例
Token IDs: [2662, 3365, 4313, 4476, 2534, 7329, 5898, 2280]
```

### 医学文本分词
```bash
$ python bert_tokenizer.py --report "患者主诉头痛发热三天，体温38.5度"
原始文本: 患者主诉头痛发热三天，体温38.5度
分词结果: 患 | 者 | 主 | 诉 | 头 | 痛 | 发 | 热 | 三 | 天 | ， | 体 | 温 | 38 | . | 5 | 度
```

## 测试

运行测试脚本查看示例效果：
```bash
python test_bert_tokenizer.py
```

## 注意事项

1. 首次使用时会自动下载BERT模型，可能需要一些时间
2. 模型会自动缓存到本地，后续使用速度会更快
3. 支持中英文混合文本的分词
4. 对于医学专业术语也能较好处理