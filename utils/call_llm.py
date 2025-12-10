#!/usr/bin/env python3
import sys
import time
import threading
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x
sys.path.append(".")
from conf import API_BASE, API_KEY
from openai import OpenAI   

# Thread-local OpenAI client to be safe under concurrency
_TLS = threading.local()

def _get_client() -> OpenAI:
    if not hasattr(_TLS, "client"):
        _TLS.client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    return _TLS.client

def call_llm(prompt: str, model: str = "qwen3-32b"):
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=360,
    )
    return response.choices[0].message.content


class _RateLimiter:
    """Simple per-process rate limiter (max requests per second)."""

    def __init__(self, rate_per_sec: Optional[float]):
        self.rate = float(rate_per_sec) if rate_per_sec else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.rate <= 0:
            return
        with self._lock:
            now = time.time()
            if self._next <= now:
                self._next = now + 1.0 / self.rate
                return
            sleep_s = max(0.0, self._next - now)
            self._next += 1.0 / self.rate
        if sleep_s > 0:
            time.sleep(sleep_s)


def _call_one(prompt: str, model: str, limiter: Optional[_RateLimiter], retries: int, timeout: Optional[float]) -> Tuple[bool, str]:
    """Call once with retries and optional rate limiting. Returns (ok, content_or_error)."""
    last_err: Optional[Exception] = None
    for attempt in range(max(0, retries) + 1):
        try:
            if limiter is not None:
                limiter.acquire()
            client = _get_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout or 360,
            )
            return True, resp.choices[0].message.content
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    return False, f"{{\"__error\": \"{type(last_err).__name__ if last_err else 'Error'}: {last_err}\"}}"


def call_llm_many(
    prompts: List[str],
    model: str = "qwen3-32b",
    max_workers: int = 8,
    retries: int = 2,
    rate_limit_per_sec: Optional[float] = None,
    show_progress: bool = False,
    timeout: Optional[float] = None,
) -> List[str]:
    """Concurrent LLM calls. Returns results aligned with prompts order.

    Args:
        prompts: list of prompts
        model: model name
        max_workers: thread pool size
        retries: retry times on failure
        rate_limit_per_sec: global max calls per second (approximate)
        show_progress: show tqdm progress
        timeout: per-request timeout seconds
    """
    if not prompts:
        return []

    limiter = _RateLimiter(rate_limit_per_sec)
    out: List[Optional[str]] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futs = {
            ex.submit(_call_one, p, model, limiter, retries, timeout): i
            for i, p in enumerate(prompts)
        }
        iterator = as_completed(futs)
        if show_progress:
            iterator = tqdm(iterator, total=len(futs), desc="[LLM] calls", dynamic_ncols=True)
        for fut in iterator:
            i = futs[fut]
            ok, content = fut.result()
            out[i] = content

    return [s or "" for s in out]

if __name__ == "__main__":
    ocr = """现病史:咳嗽咽痛姓名张三"""
    prompt = """我正在进行病历半结构化任务，请根据以下报告内容，提取报告中所有的信息,最终以key:value的形式输出。
    示例:
    输入文本：入院记录；北京协和医院 住院科室:呼吸肿瘤内科；肿瘤一科，登记/住院号：0000697006。病员姓名:李慧慧 病人性别：女；民族:汉族。病人年龄：61岁，生日:2024-04-03。出生地：甘肃省兰州市城关区，身份证号:130121196010160725 通讯地址：浙江省杭州市滨江区长河街道西浦路123号；户口地址:甘肃省兰州市城关区，单位：中国人民解放军总医院第五医学中心 职业:国家公务员；通讯方式：13812345678；甘肃省兰州市城关区某某街道123号。病史采集时间:2024-06-30 10:25:00；2024-05-25。病史陈述者：本人；患者本人，既往史:否认高血压、糖尿病、冠心病史，否认肝炎、结核等传染病史，否认重大手术及输血史，否认食物及药物过敏史，预防接种史不详 婚育史：28岁结婚,爱人体健,夫妻关系和睦,1子,体健。；药物过敏史 患者自述对青霉素类药物过敏，表现为皮疹及瘙痒，无严重过敏反应史。\n\n入院体格检查:体温36.8℃，脉搏76次/分，呼吸17次/分，血压118/78 mmHg。双肺呼吸音清晰。心率76次/分，节律规整。腹部平坦，无压痛及反跳痛，肝脾未触及肿大。脊柱无畸形，四肢关节活动自如。双下肢无水肿。双侧乳房形态对称，表面光滑。右乳11点方向、距乳头约2.4 cm处可触及一约3.0×1.7 cm肿块，质地较硬，边界欠清，活动度差。右腋窝触及数个淋巴结，最大者约2.2×1.7 cm，质地中等，活动尚可。左乳未触及异常肿块，左腋窝及锁骨上区未触及肿大淋巴结。专科查体 双侧乳房形态对称，表面光滑，无红肿、破溃或乳头异常分泌物。右乳11点方向、距乳头约2.4 cm处可触及一约3.0×1.7 cm大小肿块，质地较硬，边界欠清，活动度差，无明显压痛。右腋窝触及数个淋巴结，最大者约2.2×1.7 cm，质地中等，活动尚可，未融合。左乳未触及异常肿块，左腋窝及锁骨上区未触及肿大淋巴结。\n\n辅助检查：右乳11点方向低回声团块，大小约3.0×1.7 cm，边界欠清，内部点状钙化，BI-RADS 4B类；右腋窝多发淋巴结，最大约2.2×1.5 cm，皮质增厚，门部结构紊乱。右乳肿块BI-RADS 5类，形态不规则，回声不均，血流信号丰富；右腋窝及锁骨上区多发淋巴结，形态异常，边界模糊，初步诊断:(1)右乳腺占位性病变，大小约3.0×1.7 cm；(2)右腋窝及锁骨上区多发淋巴结肿大，最大约2.2×1.5 cm，入院诊断：1. 肺恶性肿瘤（C34.901）；2. 心脏移植术后状态（Z94.101）；3. 2型糖尿病（E11.901）；4. 慢性肾功能不全（N18.905） 修正诊断:肺恶性肿瘤 2型糖尿病 心功能不全 肾功能不全。其他：其他诊断:心脏转移状态 I94.101 2型糖尿病 E11.901 慢性肾功能不全 N18.905 肺恶性肿瘤 C34.901 病理诊断:恶性肿瘤 病理号 841737/841814 疾病编码 MS00000/3 药物过敏 1.无 2.有 过敏药物: 未填写 死亡患者尸检 1.是 2.否 ABO血型 未查；主诉:反复咳嗽、咳痰伴气促1月余，加重3天
    输出内容：
    {
    "医院名":"北京协和医院"
    "户口地址": "甘肃省兰州市城关区",
    "记录时间": "2024-06-30 10:25:00",
    "出生日期": "2024-04-03",
    "修正诊断": "肺恶性肿瘤 2型糖尿病 心功能不全 肾功能不全",
    "入院诊断": "1. 肺恶性肿瘤（C34.901）；2. 心脏移植术后状态（Z94.101）；3. 2型糖尿病（E11.901）；4. 慢性肾功能不全（N18.905）",
    "其他": "其他诊断:心脏转移状态 I94.101 2型糖尿病 E11.901 慢性肾功能不全 N18.905 肺恶性肿瘤 C34.901 病理诊断:恶性肿瘤 病理号 841737/841814 疾病编码 MS00000/3 药物过敏 1.无 2.有 过敏药物: 未填写 死亡患者尸检 1.是 2.否 ABO血型 未查",
    "日期": "2024-05-25",
    "民族": "汉族",
    "出生地": "甘肃省兰州市城关区",
    "科室": "呼吸肿瘤内科",
    "通讯方式": "13812345678；甘肃省兰州市城关区某某街道123号",
    "科别": "肿瘤一科",
    "工作单位": "中国人民解放军总医院第五医学中心",
    "病史陈述者": "本人",
    "姓名": "李慧慧",
    "入院初步诊断": "(1)右乳腺占位性病变，大小约3.0×1.7 cm；(2)右腋窝及锁骨上区多发淋巴结肿大，最大约2.2×1.5 cm",
    "主诉": "反复咳嗽、咳痰伴气促1月余，加重3天",
    "住址": "浙江省杭州市滨江区长河街道西浦路123号",
    "住院号": "0000697006",
    "婚育史": "28岁结婚,爱人体健,夫妻关系和睦,1子,体健。",
    "辅助检查结果": "右乳11点方向低回声团块，大小约3.0×1.7 cm，边界欠清，内部点状钙化，BI-RADS 4B类；右腋窝多发淋巴结，最大约2.2×1.5 cm，皮质增厚，门部结构紊乱。右乳肿块BI-RADS 5类，形态不规则，回声不均，血流信号丰富；右腋窝及锁骨上区多发淋巴结，形态异常，边界模糊",
    "专科检查": "双侧乳房形态对称，表面光滑，无红肿、破溃或乳头异常分泌物。右乳11点方向、距乳头约2.4 cm处可触及一约3.0×1.7 cm大小肿块，质地较硬，边界欠清，活动度差，无明显压痛。右腋窝触及数个淋巴结，最大者约2.2×1.7 cm，质地中等，活动尚可，未融合。左乳未触及异常肿块，左腋窝及锁骨上区未触及肿大淋巴结。",
    "职业": "国家公务员",
    "既往史": "否认高血压、糖尿病、冠心病史，否认肝炎、结核等传染病史，否认重大手术及输血史，否认食物及药物过敏史，预防接种史不详",
    "性别": "女",
    "年龄": "61岁",
    "身份证号": "130121196010160725",
    "体格检查": "体温36.8℃，脉搏76次/分，呼吸17次/分，血压118/78 mmHg。双肺呼吸音清晰。心率76次/分，节律规整。腹部平坦，无压痛及反跳痛，肝脾未触及肿大。脊柱无畸形，四肢关节活动自如。双下肢无水肿。双侧乳房形态对称，表面光滑。右乳11点方向、距乳头约2.4 cm处可触及一约3.0×1.7 cm肿块，质地较硬，边界欠清，活动度差。右腋窝触及数个淋巴结，最大者约2.2×1.7 cm，质地中等，活动尚可。左乳未触及异常肿块，左腋窝及锁骨上区未触及肿大淋巴结。",
    "报告类型": "入院记录",
    "医院名": "北京协和医院",
    "病史叙述者": "患者本人",
    "药物过敏史": "患者自述对青霉素类药物过敏，表现为皮疹及瘙痒，无严重过敏反应史。"
  }
    输入文本:{ocr_text}
    请只返回提取到的信息，不要添加多余内容，也不要推理，直接截取所有的信息。
    输出格式,按照key、value在输入文本中的出现顺序输出:{{"key":"value"}}
    """.replace("{ocr_text}",ocr)
    print(call_llm(prompt, model="qwen3-32b"))
