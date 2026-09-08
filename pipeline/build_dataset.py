#!/usr/bin/env python3
"""
Dataset curation pipeline for the Qwen3.5-9B software-engineering specialization.

Sources (raw, already downloaded into pipeline/raw/):
  - OpenCodeInstruct (nvidia)        -> generation, verified via unit-test score
  - CodeFeedback-Filtered-Instruction (m-a-p) -> debugging / multi-turn Q&A
  - CommitPackFT (bigcode)           -> refactoring / diff-based code editing, multi-language
  - Code_Vulnerability_Security_DPO (CyberNative) -> security-focused SFT (chosen side)

Pipeline stages:
  1. Load + normalize each source into a common {"messages": [...], "meta": {...}} schema
  2. Per-source quality filtering (heuristics + native quality signals)
  3. Exact dedup (hash of normalized text) within and across sources
  4. Near-dup filtering (MinHash + LSH banding, self-contained, no extra deps)
  5. Carve out held-out EVAL set first (before any train/valid split), stratified by source
  6. Split remaining pool into train / valid
  7. Write train.jsonl, valid.jsonl, eval.jsonl + a manifest with stats
  8. Verify zero overlap between eval and train/valid (hash-based assertion)

This does NOT launch training. It only prepares data.
"""
import json
import hashlib
import random
import re
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
OUT = ROOT.parent / "data_curated"
OUT.mkdir(exist_ok=True)

random.seed(1234)

TOKENIZER = None
def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        from transformers import AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(str(ROOT.parent / "model"))
    return TOKENIZER

def tok_len(messages):
    tok = get_tokenizer()
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tok(text)["input_ids"])

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def content_hash(messages) -> str:
    joined = "\x1e".join(norm_text(m["content"]) for m in messages)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# MinHash + LSH (self-contained, no extra deps)
# ---------------------------------------------------------------------------
def shingles(text: str, k: int = 5):
    words = norm_text(text).split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

_NUM_PERM = 32
_MASK = (1 << 32) - 1
_PERM_SEEDS = [random.randint(1, _MASK) for _ in range(_NUM_PERM)]

def minhash_sig(text: str):
    sh = shingles(text)
    if not sh:
        return tuple([0] * _NUM_PERM)
    sig = []
    hashes = [int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in sh]
    for seed in _PERM_SEEDS:
        sig.append(min((h ^ seed) & _MASK for h in hashes))
    return tuple(sig)

def lsh_bands(sig, n_bands=8):
    band_size = _NUM_PERM // n_bands
    return [hash(sig[i*band_size:(i+1)*band_size]) for i in range(n_bands)]

def near_dedup(records, text_key_fn, n_bands=8, jaccard_thresh=0.85):
    """records: list of dicts. text_key_fn(record) -> str to hash on.
    Returns filtered list with near-duplicates removed (keeps first occurrence)."""
    buckets = defaultdict(list)
    kept = []
    sigs = {}
    for idx, r in enumerate(records):
        text = text_key_fn(r)
        sig = minhash_sig(text)
        sigs[idx] = sig
        is_dup = False
        candidate_ids = set()
        for b in lsh_bands(sig, n_bands):
            candidate_ids.update(buckets[b])
        for cid in candidate_ids:
            csig = sigs[cid]
            agree = sum(1 for a, b in zip(sig, csig) if a == b) / _NUM_PERM
            if agree >= jaccard_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
            for b in lsh_bands(sig, n_bands):
                buckets[b].append(idx)
    return kept

# ---------------------------------------------------------------------------
# Source loaders -> unified schema
# ---------------------------------------------------------------------------
BAD_COMMIT_MSGS = ["typo", "merge", "bump version", "update readme", "wip", "fix build",
                    "minor fix", "cleanup", "update", "small fix"]

def load_opencodeinstruct(target_n=4500):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(RAW / "oci_shard0.parquet")
    out = []
    seen_hash = set()
    for batch in pf.iter_batches(batch_size=5000):
        d = batch.to_pydict()
        for i in range(len(d["id"])):
            score = d["average_test_score"][i]
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            if score < 1.0:
                continue
            inp, outp = d["input"][i], d["output"][i]
            if not inp or not outp:
                continue
            if len(inp) < 20 or len(outp) < 20:
                continue
            if len(inp) + len(outp) > 12000:
                continue
            messages = [{"role": "user", "content": inp}, {"role": "assistant", "content": outp}]
            h = content_hash(messages)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            out.append({
                "messages": messages,
                "meta": {"source": "opencodeinstruct", "domain": d["domain"][i],
                         "gen_algo": d["generation_algorithm"][i], "score": score}
            })
        if len(out) >= target_n * 3:  # gather 3x candidates pre-dedup/near-dedup headroom
            break
    random.shuffle(out)
    return out

def load_codefeedback(target_n=3000):
    out = []
    seen_hash = set()
    with open(RAW / "codefeedback.jsonl") as f:
        for line in f:
            r = json.loads(line)
            q, a = r.get("query", ""), r.get("answer", "")
            if not q or not a:
                continue
            if len(q) < 15 or len(a) < 15:
                continue
            if len(q) + len(a) > 12000:
                continue
            messages = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            h = content_hash(messages)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            out.append({
                "messages": messages,
                "meta": {"source": "codefeedback", "lang": r.get("lang"), "resource": r.get("resource")}
            })
    random.shuffle(out)
    return out[:target_n * 3]

def load_commitpackft(target_n=3000):
    langs = ["python", "javascript", "typescript", "go", "rust", "java", "c++", "c",
              "ruby", "php", "sql", "shell", "html", "css", "c#"]
    out = []
    seen_hash = set()
    per_lang_cap = max(1, (target_n * 3) // len(langs))
    for lang in langs:
        fp = RAW / "commitpackft" / f"{lang}.jsonl"
        if not fp.exists():
            continue
        lang_out = []
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = r.get("message", "").strip()
                if len(msg) < 15:
                    continue
                if any(b in msg.lower() for b in BAD_COMMIT_MSGS):
                    continue
                old_c, new_c = r.get("old_contents", ""), r.get("new_contents", "")
                if old_c == new_c:
                    continue
                if len(old_c) + len(new_c) > 9000:
                    continue
                if len(new_c) < 10:
                    continue
                user = (f"Here is `{r.get('old_file','file')}`:\n```{lang}\n{old_c}\n```\n"
                        f"Apply this change: {msg}\nShow the complete updated file.")
                assistant = f"```{lang}\n{new_c}\n```"
                messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
                h = content_hash(messages)
                if h in seen_hash:
                    continue
                seen_hash.add(h)
                lang_out.append({"messages": messages, "meta": {"source": "commitpackft", "lang": lang}})
                if len(lang_out) >= per_lang_cap:
                    break
        out.extend(lang_out)
    random.shuffle(out)
    return out

def load_security_dpo():
    out = []
    seen_hash = set()
    with open(RAW / "security_dpo.json") as f:
        for line in f:
            r = json.loads(line)
            q, chosen = r.get("question", ""), r.get("chosen", "")
            system = r.get("system", "")
            if not q or not chosen:
                continue
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": chosen})
            h = content_hash(messages)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            out.append({
                "messages": messages,
                "meta": {"source": "security_dpo", "lang": r.get("lang"), "vulnerability": r.get("vulnerability")}
            })
    random.shuffle(out)
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading sources...")
    oci = load_opencodeinstruct()
    print(f"  opencodeinstruct candidates: {len(oci)}")
    cf = load_codefeedback()
    print(f"  codefeedback candidates:     {len(cf)}")
    cpft = load_commitpackft()
    print(f"  commitpackft candidates:     {len(cpft)}")
    sec = load_security_dpo()
    print(f"  security_dpo candidates:     {len(sec)}")

    def text_key(r):
        return " ".join(m["content"] for m in r["messages"])

    print("Near-dedup per source (MinHash/LSH, jaccard>=0.85)...")
    oci = near_dedup(oci, text_key)
    cf = near_dedup(cf, text_key)
    cpft = near_dedup(cpft, text_key)
    sec = near_dedup(sec, text_key)
    print(f"  after near-dedup: oci={len(oci)} cf={len(cf)} cpft={len(cpft)} sec={len(sec)}")

    TARGETS = {"opencodeinstruct": 4500, "codefeedback": 3000, "commitpackft": 3000, "security_dpo": None}
    oci = oci[:TARGETS["opencodeinstruct"]]
    cf = cf[:TARGETS["codefeedback"]]
    cpft = cpft[:TARGETS["commitpackft"]]
    # security_dpo: keep all (already small, ~4.6k)

    pool = oci + cf + cpft + sec
    print(f"Combined pool before cross-source dedup: {len(pool)}")

    print("Cross-source near-dedup pass...")
    pool = near_dedup(pool, text_key, n_bands=8, jaccard_thresh=0.85)
    print(f"Pool after cross-source dedup: {len(pool)}")

    # stratify by source for held-out eval + valid
    by_source = defaultdict(list)
    for r in pool:
        by_source[r["meta"]["source"]].append(r)
    for s in by_source:
        random.shuffle(by_source[s])

    EVAL_FRAC = 0.04
    VALID_FRAC = 0.04
    train, valid, eval_set = [], [], []
    for s, items in by_source.items():
        n = len(items)
        n_eval = max(10, int(n * EVAL_FRAC))
        n_valid = max(10, int(n * VALID_FRAC))
        eval_set.extend(items[:n_eval])
        valid.extend(items[n_eval:n_eval + n_valid])
        train.extend(items[n_eval + n_valid:])

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(eval_set)

    # Safety assertion: zero hash overlap between eval and train/valid
    eval_hashes = {content_hash(r["messages"]) for r in eval_set}
    train_hashes = {content_hash(r["messages"]) for r in train}
    valid_hashes = {content_hash(r["messages"]) for r in valid}
    overlap = eval_hashes & (train_hashes | valid_hashes)
    assert not overlap, f"LEAKAGE DETECTED: {len(overlap)} eval examples also in train/valid!"
    print("Leakage check passed: eval set has zero overlap with train/valid.")

    # token stats (sampled for speed)
    def sample_tok_stats(records, n=150):
        sample = records if len(records) <= n else random.sample(records, n)
        lens = [tok_len(r["messages"]) for r in sample]
        return lens

    print("Computing token stats (sampled)...")
    train_lens = sample_tok_stats(train)
    total_train_tokens_est = statistics.mean(train_lens) * len(train)

    def write_jsonl(path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps({"messages": r["messages"]}) + "\n")

    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "valid.jsonl", valid)
    write_jsonl(OUT / "eval.jsonl", eval_set)

    manifest = {
        "counts": {
            "train": len(train), "valid": len(valid), "eval": len(eval_set),
            "by_source_total": {s: len(v) for s, v in by_source.items()},
        },
        "token_stats_sampled": {
            "n_sampled": len(train_lens),
            "mean": round(statistics.mean(train_lens), 1),
            "median": round(statistics.median(train_lens), 1),
            "p95": round(sorted(train_lens)[int(len(train_lens)*0.95)], 1),
            "max": max(train_lens),
        },
        "estimated_total_train_tokens": round(total_train_tokens_est),
        "leakage_check": "PASSED - zero hash overlap eval vs train/valid",
        "dedup_method": "exact sha256 hash + MinHash(32 perm)/LSH(8 band) near-dup, jaccard>=0.85",
    }
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
