import torch
from modelscope import AutoTokenizer
import sys

sys.path.append(".")

from options.model_option import OptionBertClassifier
from options.option_label_data import surgery_options

try:
    from model_path_conf import DEFAULT_TOKENIZER_PATH  # type: ignore
except Exception as exc:
    raise ImportError(
        "DEFAULT_TOKENIZER_PATH must be defined in model_path_conf.py"
    ) from exc

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH)

    # 加载Option归一化模型
    model = OptionBertClassifier().to(device)
    try:
        checkpoint = torch.load("option_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(
            f"模型已加载，训练到epoch: {checkpoint.get('epoch', None)}, steps: {checkpoint.get('steps', None)}"
        )
    except Exception as e:
        print("加载模型参数出错:", e)
        import sys

        sys.exit(1)
    model.eval()
    return model, tokenizer, device


def predict_options(query, field,model,tokenizer,device, max_length=128, threshold=0.5):
    """
    query: 原文字符串
    field: 字段名字符串（如'手术部位'、'手术名称'等）
    return: 标准项与预测概率字典
    """
    options = surgery_options[field]
    pair_inputs = []
    for option in options:
        enc = tokenizer(
            query,
            option,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        pair_inputs.append(
            {
                "input_ids": enc["input_ids"][0].unsqueeze(0),  # [1, L]
                "attention_mask": enc["attention_mask"][0].unsqueeze(0),
                "token_type_ids": (
                    enc["token_type_ids"][0].unsqueeze(0)
                    if "token_type_ids" in enc
                    else torch.zeros_like(enc["input_ids"][0]).unsqueeze(0)
                ),
            }
        )
    # 拼成batch
    input_ids = torch.cat([p["input_ids"] for p in pair_inputs], dim=0).unsqueeze(
        0
    )  # [1, N, L]
    attn_mask = torch.cat([p["attention_mask"] for p in pair_inputs], dim=0).unsqueeze(
        0
    )
    token_type_ids = torch.cat(
        [p["token_type_ids"] for p in pair_inputs], dim=0
    ).unsqueeze(0)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    with torch.no_grad():
        probs = model.forward(input_ids, attn_mask, token_type_ids).squeeze(0)  # [N]
        preds = (probs > threshold).cpu().tolist()
        probs = probs.cpu().tolist()
    return {
        opt: {"pred": int(pred), "prob": float(prob)}
        for opt, pred, prob in zip(options, preds, probs)
    }


if __name__ == "__main__":
    s = "乳房切除术+腋窝淋巴结清扫"
    field = "手术名称"
    model, tokenizer, device = load_model()
    result = predict_options(s, field,model,tokenizer,device)
    print(f"原文: {s}\n字段: {field}\n预测归一化结果:")
    for k, v in result.items():
        print(f"  {k:10s} pred: {v['pred']}, prob: {v['prob']:.3f}")

    # 批量测试
    s_list = [
        ("右乳腺", "手术部位"),
        ("全麻下行右乳腺肿物局部扩大切除术+右乳腺癌改良根治术+右前哨淋巴结活检术", "手术名称"),
        ("全麻下行右乳腺肿物局部扩大切除术+右乳腺癌改良根治术+右前哨淋巴结活检术", "术式"),
        ("全麻下行右乳腺肿物局部扩大切除术+右乳腺癌改良根治术+右前哨淋巴结活检术", "手术类型"),
        ("全麻下行右乳腺肿物局部扩大切除术+右乳腺癌改良根治术+右前哨淋巴结活检术", "切口类型"),
        ("全麻下行右乳腺肿物局部扩大切除术+右乳腺癌改良根治术+右前哨淋巴结活检术", "淋巴结清扫"),
    ]
    for query, field in s_list:
        res = predict_options(query, field,model,tokenizer,device)
        print(f"\n原文: {query} 字段: {field}")
        print("预测归一化：", {k: v["pred"] for k, v in res.items()})
