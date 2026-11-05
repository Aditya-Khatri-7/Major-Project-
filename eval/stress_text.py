"""Text stress test: perturb held-out texts and measure where the text pipeline breaks.

    python eval/stress_text.py --n 120

Uses the real tools (DeBERTa classifier + Binoculars; judges are switched off) and the frozen fusion rules from
models/calibration.json. Base texts come from data/processed/text/mage_test.csv. Output: eval/results/stress/text_stress.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ["GEMINI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.common import run_tool  # noqa: E402
from agents.fusion import fuse  # noqa: E402

SYN = {"important": "crucial", "use": "utilise", "help": "assist", "show": "demonstrate", "big": "substantial",
       "good": "beneficial", "many": "numerous", "need": "require", "people": "individuals", "also": "additionally",
       "because": "since", "but": "however", "get": "obtain", "make": "create", "think": "believe"}
EXPAND = {"don't": "do not", "can't": "cannot", "it's": "it is", "isn't": "is not", "won't": "will not", "I'm": "I am"}


def _typos(t: str, rng: random.Random, p: float = 0.08) -> str:
    out = []
    for w in t.split():
        if len(w) > 3 and rng.random() < p:
            i = rng.randrange(len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def _synonyms(t: str) -> str:
    for k, v in {**SYN, **EXPAND}.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t, flags=re.I)
    return t


def _shuffle_sentences(t: str, rng: random.Random) -> str:
    s = re.split(r"(?<=[.!?])\s+", t)
    rng.shuffle(s)
    return " ".join(s)


def _emoji(t: str) -> str:
    s = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(x + (" \U0001F60A" if i % 3 == 0 else "") for i, x in enumerate(s)) + " #thoughts #life"


PERTURB = {
    "original": lambda t, r: t,
    "first_25_words": lambda t, r: " ".join(t.split()[:25]),
    "first_50_words": lambda t, r: " ".join(t.split()[:50]),
    "typos_8pct": lambda t, r: _typos(t, r),
    "lowercase_no_punct": lambda t, r: re.sub(r"[^\w\s]", "", t.lower()),
    "synonym_swap": lambda t, r: _synonyms(t),
    "sentence_shuffle": lambda t, r: _shuffle_sentences(t, r),
    "emoji_hashtags": lambda t, r: _emoji(t),
    "double_text": lambda t, r: t + "\n\n" + t,
    "all_caps": lambda t, r: t.upper(),
}

# Human-written edge cases that a detector often mislabels (expected: authentic).
HUMAN_EDGE = {
    "gettysburg_formal": "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in "
        "Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, "
        "testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great "
        "battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who "
        "here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.",
    "legal_boilerplate": "This Agreement shall be governed by and construed in accordance with the laws of the State of New York, "
        "without regard to its conflict of laws principles. Any dispute arising out of or in connection with this Agreement "
        "shall be submitted to the exclusive jurisdiction of the courts located in New York County. The parties hereby "
        "irrevocably waive any objection to the laying of venue of any such proceeding in such courts, and agree that "
        "no provision of this Agreement shall be deemed waived unless such waiver is in writing and signed by both parties.",
    "esl_simple": "I am student in college. My city is very big and have many peoples. Every day I go to school by bus, it take "
        "one hour. I like my teacher because she is very kind and she explain the lesson good. After school I help my mother "
        "in the kitchen and then I do homework. On weekend I play cricket with my friends in the park near my house.",
    "scientific_abstract": "We investigate the effect of temperature on the catalytic activity of a zeolite-supported palladium "
        "catalyst in the hydrogenation of cinnamaldehyde. Conversion increased from 12% at 303 K to 87% at 353 K, whereas "
        "selectivity toward hydrocinnamaldehyde decreased from 94% to 71%. Kinetic analysis yielded an apparent activation "
        "energy of 38.4 kJ/mol. These results suggest that mass-transfer limitations are negligible under the conditions studied.",
    "hindi_roman": "Mujhe lagta hai ki is saal ka budget bahut achha nahi hai. Sarkar ne middle class ke liye kuch khaas nahi kiya "
        "aur mehngai abhi bhi bahut zyada hai. Meri maa kehti hain ki pehle ke zamane mein cheezein sasti thi, lekin tab "
        "tankhwah bhi kam hoti thi. Phir bhi, hum sabko apna kharcha dekh kar chalna padega aur bachat bhi karni hogi.",
    "spanish": "Ayer fui al mercado con mi abuela y compramos tomates, cebollas y un poco de pescado fresco. Ella siempre regatea "
        "el precio con los vendedores, y casi siempre consigue una rebaja. Después volvimos a casa caminando por el parque, "
        "donde unos niños jugaban al fútbol. Por la noche cocinamos juntos una sopa que huele a mi infancia.",
    "code_comment_mix": "def merge(a, b):\n    # merge two sorted lists into one sorted list\n    i = j = 0\n    out = []\n    while i < len(a) and "
        "j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            "
        "out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out  # O(n + m) time and space\n",
    "recipe_list": "Ingredients: 2 cups flour, 1 tsp salt, 3 eggs, 1 cup milk, 2 tbsp butter. Mix flour and salt. Whisk eggs and milk "
        "together, then pour into the flour and stir until smooth. Melt butter in a pan over medium heat. Pour in a small "
        "ladle of batter, tilt the pan, cook 2 minutes each side. Serve warm with honey, lemon or jam. Makes about eight crepes.",
    "poem": "The fog comes on little cat feet. It sits looking over harbor and city on silent haunches and then moves on. "
        "Roses are red, the old song goes, and violets blue, but nobody tells you the hours you spend waiting for the "
        "phone to ring, the tea going cold, the window full of rain that will not stop for anyone at all today.",
    "repetitive_human": "lol yeah yeah i know i know. its just that he keeps saying the same thing over and over and over. "
        "like seriously dude we get it. we get it. you dont like the new rule. fine. fine fine fine. but it is what it is "
        "and we cant do anything about it so lets just move on okay okay okay.",
}


def score(text: str, dl, slm) -> dict:
    n = len(text.split())
    vs = [run_tool("text_dl", "text", dl, text), run_tool("text_slm", "text", slm, text)]
    r = fuse(vs, "text", n)
    return {"fused": r.fused, "verdict": r.verdict, "escalate": r.escalate, "dl": vs[0].score, "slm": vs[1].score,
            "dl_err": vs[0].error, "slm_err": vs[1].error}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="texts per class")
    ap.add_argument("--min-words", type=int, default=80)
    ap.add_argument("--out", type=Path, default=Path("eval/results/stress/text_stress.json"))
    args = ap.parse_args()

    from agents.text_agent.tools import dl_classifier, slm_binoculars
    dl, slm = dl_classifier.score_text, slm_binoculars.score_text

    df = pd.read_csv("data/processed/text/mage_test.csv")
    df = df[df.text.str.split().str.len() >= args.min_words]
    human = df[df.label == 0].sample(args.n, random_state=1).text.tolist()
    ai = df[df.label == 1].sample(args.n, random_state=1).text.tolist()
    rng = random.Random(0)

    results: dict = {"perturbations": {}, "human_edge_cases": {}, "mixed_authorship": {}}
    for name, fn in PERTURB.items():
        rows = []
        for label, texts in ((0, human), (1, ai)):
            for t in texts:
                s = score(fn(t, rng), dl, slm)
                s["label"] = label
                rows.append(s)
        d = pd.DataFrame(rows)
        ok = d[d.fused.notna()]
        auc = float(roc_auc_score(ok.label, ok.fused)) if ok.label.nunique() == 2 else None
        hum, aid = d[d.label == 0], d[d.label == 1]
        decided = d[d.verdict != "uncertain"]
        acc_dec = float(((decided.verdict == "synthetic") == (decided.label == 1)).mean()) if len(decided) else None
        results["perturbations"][name] = {
            "auc": auc, "human_false_alarm_rate": float((hum.verdict == "synthetic").mean()),
            "ai_missed_rate": float((aid.verdict == "authentic").mean()),
            "escalation_rate": float(d.escalate.mean()), "accuracy_when_decided": acc_dec,
            "mean_fused_human": float(hum.fused.mean()), "mean_fused_ai": float(aid.fused.mean()),
            "dl_auc": float(roc_auc_score(d.label, d.dl)), "slm_auc": float(roc_auc_score(d.label, d.slm)),
        }
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in results["perturbations"][name].items()}, flush=True)

    for name, t in HUMAN_EDGE.items():
        s = score(t, dl, slm)
        results["human_edge_cases"][name] = {k: s[k] for k in ("fused", "verdict", "escalate", "dl", "slm")}
        print("edge", name, s["verdict"], round(s["fused"], 3) if s["fused"] is not None else None, flush=True)

    # AI + human concatenation: what does the system say as the AI share grows?
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        fused = []
        for a, h in list(zip(ai, human))[:40]:
            aw, hw = a.split(), h.split()
            k = int(min(len(aw), 150) * frac)
            mix = " ".join(aw[:k] + hw[: 150 - k])
            fused.append(score(mix, dl, slm)["fused"])
        results["mixed_authorship"][f"ai_share_{int(frac * 100)}pct"] = float(np.nanmean([f for f in fused if f is not None]))
        print("mixed", frac, results["mixed_authorship"][f"ai_share_{int(frac * 100)}pct"], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
