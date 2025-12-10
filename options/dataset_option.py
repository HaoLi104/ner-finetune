import torch
import json
import ast
import re

from options.option_label_data import surgery_options

class OptionDataset(torch.utils.data.Dataset):
    def __init__(self, filename, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs, self.labels, self.counts, self.fields = self._preprocess(self._load_data(filename))

    def _load_data(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    def _split_input(self, input_text):
        # input_text: "左乳。标注:['手术部位']"
        if "。标注:" not in input_text:
            return input_text.strip(), []
        text, label_part = input_text.split("。标注:", 1)
        text = text.strip()
        label_fields = []
        match = re.search(r"\[.*?\]", label_part)
        if match:
            try:
                label_fields = ast.literal_eval(match.group(0))
                if isinstance(label_fields, str):
                    label_fields = [label_fields]
            except Exception:
                label_fields = []
        return text, label_fields

    def _preprocess(self, data):
        inputs, labels, counts, fields = [], [], [], []
        for item in data:
            origin_label_text = item["data"]["text"]
            text, label_fields = self._split_input(origin_label_text)
            for field in label_fields:
                if field not in surgery_options:
                    continue
                options = surgery_options[field]
                chosen_set = set()
                # 取标注内容
                if "annotations" in item and item["annotations"]:
                    results = item["annotations"][0].get("result", [])
                    for r in results:
                        if r.get("from_name") == field and "taxonomy" in r.get("value", {}):
                            for taxo in r["value"]["taxonomy"]:
                                for v in taxo:
                                    chosen_set.add(v)
                multi_label = [(1.0 if opt in chosen_set else 0.0) for opt in options]
                count = sum(multi_label)
                # 编码所有pair
                pair_inputs = []
                for option in options:
                    enc = self.tokenizer(
                        text,
                        option,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    pair_inputs.append({
                        'input_ids': enc['input_ids'][0],
                        'attention_mask': enc['attention_mask'][0],
                        'token_type_ids': enc['token_type_ids'][0] if 'token_type_ids' in enc else None,
                        'raw_text': f"{text}[SEP]{option}"
                    })
                inputs.append(pair_inputs)
                labels.append(torch.tensor(multi_label, dtype=torch.float) if multi_label is not None else None)
                counts.append(count)
                fields.append(field)
        return inputs, labels, counts, fields

    def __getitem__(self, idx):
        """
        返回：
            pair_inputs: List[Dict]
            labels: Tensor, shape=(num_options,) 或 None
            count: int 或 None
            field: 字段名（如“手术部位”）
        """
        return self.inputs[idx], self.labels[idx], self.counts[idx], self.fields[idx]

    def __len__(self):
        return len(self.inputs)

# 用法举例
if __name__ == "__main__":
    from modelscope import AutoTokenizer
    try:
        from model_path_conf import DEFAULT_TOKENIZER_PATH  # type: ignore
    except Exception as exc:
        raise ImportError(
            "DEFAULT_TOKENIZER_PATH must be defined in model_path_conf.py"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH)
    ds = OptionDataset("data/train_option_label.json", tokenizer)
    pair_inputs, labels, count = ds[0]
    print(pair_inputs[0]['raw_text'])
    print(labels)
    print(count)
