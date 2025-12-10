import sys

sys.path.append(".")

from modelscope import AutoTokenizer
import torch
import os
import tqdm
import numpy as np
import random
from options.dataset_option import OptionDataset
from options.model_option import OptionBertClassifier
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def collate_fn_option(batch):
    # batch是[(pair_inputs, labels, count, field), ...]
    # pair_inputs: List[Dict]，长度=num_options
    # labels: Tensor, shape=(num_options,)
    # count: int
    # 输出: input_ids, attn_mask, token_type_ids, labels
    batch_input_ids, batch_attn_mask, batch_token_type_ids, batch_labels = (
        [],
        [],
        [],
        [],
    )
    max_num_options = max(len(x[0]) for x in batch)
    for pair_inputs, labels, _, _ in batch:
        n = len(pair_inputs)
        input_ids = torch.stack([p["input_ids"] for p in pair_inputs])
        attn_mask = torch.stack([p["attention_mask"] for p in pair_inputs])
        if pair_inputs[0]["token_type_ids"] is not None:
            token_type_ids = torch.stack([p["token_type_ids"] for p in pair_inputs])
        else:
            token_type_ids = torch.zeros_like(input_ids)
        # pad到同一num_options（极少用到，但保证dataloader batch对齐）
        pad_len = max_num_options - n
        if pad_len > 0:
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.zeros((pad_len, input_ids.shape[1]), dtype=input_ids.dtype),
                ],
                dim=0,
            )
            attn_mask = torch.cat(
                [
                    attn_mask,
                    torch.zeros((pad_len, attn_mask.shape[1]), dtype=attn_mask.dtype),
                ],
                dim=0,
            )
            token_type_ids = torch.cat(
                [
                    token_type_ids,
                    torch.zeros(
                        (pad_len, token_type_ids.shape[1]), dtype=token_type_ids.dtype
                    ),
                ],
                dim=0,
            )
            labels = torch.cat([labels, torch.zeros(pad_len, dtype=labels.dtype)])
        batch_input_ids.append(input_ids)
        batch_attn_mask.append(attn_mask)
        batch_token_type_ids.append(token_type_ids)
        batch_labels.append(labels)
    # batch_size x num_options x seq_len
    batch_input_ids = torch.stack(batch_input_ids)
    batch_attn_mask = torch.stack(batch_attn_mask)
    batch_token_type_ids = torch.stack(batch_token_type_ids)
    batch_labels = torch.stack(batch_labels)
    return batch_input_ids, batch_attn_mask, batch_token_type_ids, batch_labels


def save_checkpoint(path, model, optimizer, epoch, steps):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "steps": steps,
        },
        path,
    )


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("steps", 0)


if __name__ == "__main__":
    try:
        from model_path_conf import DEFAULT_TOKENIZER_PATH  # type: ignore
    except Exception as exc:
        raise ImportError(
            "DEFAULT_TOKENIZER_PATH must be defined in model_path_conf.py"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH)
    dataset = OptionDataset(
        "data/train_option_label.json", tokenizer, max_length=MAX_LEN
    )
    print("样本数：", len(dataset))
    model = OptionBertClassifier().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05
    )
    total_steps = 30 * len(dataset) // 8
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, total_steps // 3), gamma=0.5
    )

    start_epoch, steps = 0, 0
    ckpt_path = "option_model.pt"
    if os.path.exists(ckpt_path):
        start_epoch, steps = load_checkpoint(ckpt_path, model, optimizer)
        tqdm.tqdm.write(f"Resumed from {ckpt_path}, epoch={start_epoch}, steps={steps}")
    else:
        tqdm.tqdm.write("No checkpoint found, start fresh")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn_option,
        persistent_workers=True,
    )

    max_epochs = 30
    model.train()
    for epoch in range(start_epoch, max_epochs):
        pbar = tqdm.tqdm(dataloader, total=len(dataloader), ncols=100)
        for (
            batch_input_ids,
            batch_attn_mask,
            batch_token_type_ids,
            batch_labels,
        ) in pbar:
            batch_input_ids = batch_input_ids.to(device)
            batch_attn_mask = batch_attn_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch_labels = batch_labels.to(device)
            loss = model(
                batch_input_ids, batch_attn_mask, batch_token_type_ids, batch_labels
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            steps += 1
            pbar.set_description(f"Epoch {epoch}")
            pbar.set_postfix(
                loss=loss.item(), steps=steps, lr=optimizer.param_groups[0]["lr"]
            )

            # checkpoint
            if steps % 300 == 0:
                save_checkpoint(".option_model.pt", model, optimizer, epoch, steps)
                os.replace(".option_model.pt", ckpt_path)
        # 每个 epoch 都保存
        save_checkpoint(".option_model.pt", model, optimizer, epoch + 1, steps)
        os.replace(".option_model.pt", ckpt_path)
