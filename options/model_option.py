import torch
from modelscope import BertModel


class OptionBertClassifier(torch.nn.Module):
    def __init__(self, hidden_size=768, dropout_rate=0.1, freeze_bert_layers=8):
        super().__init__()
        try:
            from model_path_conf import DEFAULT_MODEL_PATH  # type: ignore
        except Exception as exc:
            raise ImportError(
                "DEFAULT_MODEL_PATH must be defined in model_path_conf.py"
            ) from exc
        self.bert = BertModel.from_pretrained(DEFAULT_MODEL_PATH)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(hidden_size, 1)  # 每个pair输出1个logit

        # 冻结前n层BERT参数
        for name, param in self.bert.named_parameters():
            # 训练最后4层
            if any(f"layer.{i}." in name for i in range(freeze_bert_layers, 12)):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # input_ids: [B, N, L]  B=batch, N=option, L=seq_len
        B, N, L = input_ids.size()
        input_ids = input_ids.view(B * N, L)
        attention_mask = attention_mask.view(B * N, L)
        token_type_ids = token_type_ids.view(B * N, L)

        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # [CLS]向量
        cls_vec = outputs.last_hidden_state[:, 0, :]  # [B*N, H]
        logits = self.classifier(self.dropout(cls_vec)).view(B, N)  # [B, N]

        if labels is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss
        else:
            probs = torch.sigmoid(logits)
            return probs

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, token_type_ids, threshold=0.5):
        probs = self.forward(input_ids, attention_mask, token_type_ids)  # [B, N]
        preds = (probs > threshold).float()
        return preds, probs


if __name__ == "__main__":
    model = OptionBertClassifier()
    B, N, L = 2, 3, 20  # batch, num_options, seq_len
    input_ids = torch.randint(0, 100, size=(B, N, L))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    labels = torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.float)
    loss = model(input_ids, attention_mask, token_type_ids, labels)
    print("loss:", loss.item())
    preds, probs = model.predict(input_ids, attention_mask, token_type_ids)
    print("preds:", preds)
    print("probs:", probs)
