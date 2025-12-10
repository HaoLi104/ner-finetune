import re

def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", "", re.sub(r"\\[ntr]", "", text))

def split_text(text: str,
               max_len: int = 500,
               min_len_candidates=(448, 256, 128),
               merge_threshold: int = 10,
               return_raw: bool = False):
    priority_groups = [
        ["。"]
    ]
    tail_break_chars = set(")）]】 \n\t")
    parts = []
    current = []
    cur_clean_len = 0
    n = len(text)
    i = 0

    def flush_chunk(raw_chunk):
        raw_chunk = raw_chunk.strip()
        if not raw_chunk:
            return
        clean_chunk = clean_text(raw_chunk)
        if parts and len(clean_chunk) <= merge_threshold:
            if return_raw:
                parts[-1]["raw"] += raw_chunk
                parts[-1]["clean"] += clean_chunk
            else:
                parts[-1] += clean_chunk
        else:
            if return_raw:
                parts.append({"raw": raw_chunk, "clean": clean_chunk})
            else:
                parts.append(clean_chunk)

    def find_cut_index(chunk, min_lens):
        for min_len in min_lens:
            for group in priority_groups:
                for j in range(len(chunk)-1, -1, -1):
                    if chunk[j] in group and len(clean_text(chunk[:j+1])) >= min_len:
                        return j+1
        return -1

    while i < n:
        ch = text[i]
        current.append(ch)
        if ch not in ("\n", "\t", " "):
            cur_clean_len += 0 if ch in ("\\", "n", "t", "r") else 1
        i += 1

        if cur_clean_len >= max_len:
            raw_chunk = "".join(current)
            cut_idx = find_cut_index(raw_chunk, min_len_candidates)
            if cut_idx == -1:
                while i < n:
                    ch2 = text[i]
                    current.append(ch2)
                    i += 1
                    if ch2 not in ("\n", "\t", " "):
                        cur_clean_len += 0 if ch2 in ("\\", "n", "t", "r") else 1
                    if ch2 in tail_break_chars:
                        cut_idx = len(current)
                        break
                if cut_idx == -1:
                    cut_idx = len(current)
            flush_chunk("".join(current[:cut_idx]))
            current = current[cut_idx:]
            cur_clean_len = len(clean_text("".join(current)))

    if current:
        flush_chunk("".join(current))

    return parts