"""Hard text cases the first detector got wrong with high confidence: formal, templated, historical and code-like human writing.

    python data/build_hard_text.py human        # human chunks from legislation, government reports, classic books, Python code
    python data/build_hard_text.py prompts      # prompts for local LLMs to write matching AI text
    python data/build_hard_text.py ai --model Qwen2.5-1.5B-Instruct --n 1200   # generate AI counterparts (GPU)

Outputs (git-ignored): data/processed/text/hard_human.csv, hard_prompts.csv, hard_ai_<model>.csv
Columns follow the other text CSVs: id,text,label,group,source,generator,domain,attack
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/text_hard")
OUT = Path("data/processed/text")
rng = random.Random(21)

THEMES = ["the sea", "a long war", "friendship", "justice and the law", "a train journey", "the death of a king", "poverty in the city",
          "ambition", "a village in winter", "the duties of citizens", "education of the young", "a quarrel between brothers", "faith and doubt",
          "freedom", "a ship lost in a storm", "the harvest", "a trial in court", "the honour of a soldier", "a letter from abroad",
          "progress and industry", "a family secret", "the power of the press", "love and duty", "a speech to the people", "grief", "exile"]
STYLES = ["a Victorian novel", "a famous historical speech", "a philosophical essay of the 1800s", "a travel memoir", "a sermon",
          "a political pamphlet", "a courtroom address", "an old adventure story"]


def chunk(text: str, lo: int = 80, hi: int = 300) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < 3:                                    # no paragraph breaks: group sentences instead
        sents = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        paras = [" ".join(sents[i:i + 6]) for i in range(0, len(sents), 6)]
    out, cur = [], []
    for p in paras:
        cur.append(p)
        n = sum(len(x.split()) for x in cur)
        if n >= lo:
            if n <= hi:
                out.append("\n\n".join(cur))
            cur = []
    return out


def rows(texts, domain, source, label=0, generator="human"):
    return [{"id": hashlib.md5(t.encode()).hexdigest()[:12], "text": t, "label": label, "group": domain, "source": source,
             "generator": generator, "domain": domain, "attack": "none"} for t in texts]


def build_human(per_domain: int = 1500) -> None:
    out = []
    bill = pd.read_parquet(next((RAW / "billsum").rglob("*.parquet")))
    texts = []
    for t in bill["text"].sample(frac=1, random_state=1):
        texts += [c for c in chunk(t.replace("\r", "")) ][:2]
        if len(texts) >= per_domain:
            break
    out += rows(texts[:per_domain], "legal", "billsum")

    gov = pd.read_parquet(next((RAW / "govreport-summarization").rglob("*.parquet")))
    texts = []
    for t in gov["report"].sample(frac=1, random_state=2):
        texts += chunk(t)[:2]
        if len(texts) >= per_domain:
            break
    out += rows(texts[:per_domain], "gov_report", "govreport")

    gut = pd.read_parquet(next((RAW / "project_gutenberg").rglob("*.parquet")))
    col = "text" if "text" in gut.columns else gut.columns[0]
    texts = []
    for t in gut[col].sample(frac=1, random_state=3):
        body = t[len(t) // 5: len(t) * 4 // 5]                       # skip licence header/footer
        c = chunk(body.replace("\r", ""))
        if c:
            texts += rng.sample(c, min(3, len(c)))
        if len(texts) >= per_domain:
            break
    out += rows(texts[:per_domain], "books_classic", "gutenberg")

    code = pd.read_parquet(next((RAW / "code-search-net-python").rglob("*.parquet")))
    resp = [r for r in code["original_string"].sample(frac=1, random_state=4) if 60 <= len(r.split()) <= 300][:per_domain]
    out += rows(resp, "code", "codesearchnet")
    df = pd.DataFrame(out).drop_duplicates("id")
    df.to_csv(OUT / "hard_human.csv", index=False)
    print(df.domain.value_counts().to_dict())


def build_prompts(per_domain: int = 1500) -> None:
    p = []
    bill = pd.read_parquet(next((RAW / "billsum").rglob("*.parquet")))
    for t in bill["title"].dropna().sample(per_domain, random_state=5):
        p.append(("legal", f"Write the text of a section of a United States bill titled \"{t[:200]}\". Use formal legislative language."))
    gov = pd.read_parquet(next((RAW / "govreport-summarization").rglob("*.parquet")))
    for s in gov["summary"].dropna().sample(per_domain, random_state=6):
        first = re.split(r"(?<=[.!?])\s+", s.strip())[0][:300]
        p.append(("gov_report", f"Write an excerpt of an official government report. Topic: {first}"))
    for _ in range(per_domain):
        p.append(("books_classic", f"Write a passage of about {rng.randint(100, 280)} words from {rng.choice(STYLES)} about {rng.choice(THEMES)}. "
                                   "Write only the passage."))
    code = pd.read_parquet(next((RAW / "code-search-net-python").rglob("*.parquet")))
    for s in code["summary"].dropna().sample(per_domain, random_state=7):
        p.append(("code", f"{s[:400]}\nWrite the Python code with a docstring. Output only code."))
    pd.DataFrame(p, columns=["domain", "prompt"]).to_csv(OUT / "hard_prompts.csv", index=False)
    print(len(p), "prompts")


PREAMBLE = re.compile(r"^\s*(sure|certainly|here('s| is| are)|of course|okay)[^\n]*\n+", re.I)


def clean(t: str, domain: str) -> str:
    t = PREAMBLE.sub("", t.strip())
    if domain == "code":
        t = re.sub(r"^```(?:python)?\n?|\n?```$", "", t.strip())
    return t.strip()


def generate(model_name: str, n: int, batch: int, out_name: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = f"models/gen/{model_name}"
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    prompts = pd.read_csv(OUT / "hard_prompts.csv").groupby("domain", group_keys=False).sample(frac=1, random_state=hash(model_name) % 1000)
    prompts = prompts.groupby("domain", group_keys=False).head(n // 4).reset_index(drop=True)
    res = []
    out = OUT / f"hard_ai_{out_name}.csv"
    for i in range(0, len(prompts), batch):
        part = prompts.iloc[i:i + batch]
        msgs = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in part.prompt]
        enc = tok(msgs, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=380, do_sample=True, temperature=0.85, top_p=0.95, repetition_penalty=1.05)
        for dom, g in zip(part.domain, gen[:, enc.input_ids.shape[1]:]):
            t = clean(tok.decode(g, skip_special_tokens=True), dom)
            if len(t.split()) >= 60:
                res.extend(rows([t], dom, "local_llm", label=1, generator=model_name))
        pd.DataFrame(res).to_csv(out, index=False)
        print(f"{i + len(part)}/{len(prompts)} kept {len(res)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["human", "prompts", "ai"])
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"human": build_human, "prompts": build_prompts}.get(a.step, lambda: generate(a.model, a.n, a.batch, a.model))()
    sys.exit(0)
