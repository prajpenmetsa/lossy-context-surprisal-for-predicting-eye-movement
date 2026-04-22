"""
GPT-2 Surprisal + Covariates for ZuCo-1 SR
===========================================
Computes per-word surprisal from GPT-2, plus baseline covariates
(word length, Zipf frequency, word position), then merges with
et_measures.csv to produce the final analysis-ready dataset.

Usage:
  pip install transformers torch wordfreq scipy numpy pandas
  python compute_surprisal.py \
      --et       et_measures.csv \
      --sentences sentencesSR.mat \
      --out      analysis_data.csv

Outputs analysis_data.csv with all ET measures + surprisal + covariates.
"""

import argparse
import re
import numpy as np
import pandas as pd
import scipy.io
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wordfreq import zipf_frequency


# ── sentence loader ───────────────────────────────────────────────────────────

def load_sentences(mat_path):
    """
    Load sentencesSR.mat → ordered list of sentence strings.
    Returns list indexed 0..N-1, matching global_sent_idx in ET data.
    """
    data = scipy.io.loadmat(mat_path, squeeze_me=True)
    # Find the main variable (not __header__ etc.)
    key = next(k for k in data if not k.startswith('_'))
    raw = data[key]

    sentences = []
    for entry in np.atleast_1d(raw):
        sentences.append(str(entry).strip())

    print(f"Loaded {len(sentences)} sentences from {mat_path}")
    print(f"  First: {sentences[0][:80]}")
    print(f"  Last:  {sentences[-1][:80]}")
    return sentences


# ── word tokeniser (simple, matches wordbounds word count) ───────────────────

def tokenize_sentence(sentence):
    """
    Split sentence into words, stripping leading/trailing punctuation
    so that word count matches wordbounds (which are pixel-box based).
    We preserve the original token for display but compute surprisal on
    the clean version.
    """
    # Split on whitespace; keep punctuation attached (GPT-2 handles it)
    return sentence.split()


# ── GPT-2 surprisal ───────────────────────────────────────────────────────────

def compute_surprisal_sentence(sentence, model, tokenizer, device):
    """
    Compute word-level surprisal for a sentence using GPT-2.

    GPT-2 is a subword model. Strategy:
      1. Tokenize the full sentence with offset mapping.
      2. Run a single forward pass to get per-token log-probs.
      3. Align tokens back to whitespace-delimited words.
      4. Sum negative log-probs for multi-token words.

    Returns list of (word_str, surprisal_bits) aligned to sentence.split().
    Surprisal is in bits (log base 2).
    """
    words = sentence.split()

    # Encode with offset mapping to align subwords → words
    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors='pt',
        add_special_tokens=False
    )
    input_ids     = encoding['input_ids'].to(device)         # (1, T)
    offset_mapping = encoding['offset_mapping'][0].tolist()   # [(start, end), ...]

    if input_ids.shape[1] == 0:
        return [(w, np.nan) for w in words]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits  = outputs.logits                              # (1, T, vocab)

    # log-softmax → log-probs, then shift: logits[t] predicts token[t+1]
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)  # (T, vocab)

    # token_surprisals[t] = -log2 P(token[t] | token[0..t-1])
    #   = surprise at position t given the preceding context
    # For t=0 (first token) there is no left context, so we assign nan.
    token_surprisals = []
    for t in range(input_ids.shape[1]):
        if t == 0:
            token_surprisals.append(np.nan)   # no left context for first token
        else:
            token_id = input_ids[0, t].item()
            lp       = log_probs[t - 1, token_id].item()   # log P(token_t | ...t-1)
            token_surprisals.append(-lp / np.log(2))        # convert to bits

    # Build char-level map: for each character position → token index
    char_to_token = {}
    for tok_idx, (start, end) in enumerate(offset_mapping):
        for c in range(start, end):
            char_to_token[c] = tok_idx

    # Align tokens to whitespace words via character offsets
    word_surprisals = []
    char_pos = 0
    for word in words:
        # Find character span of this word in the original sentence
        start_char = sentence.find(word, char_pos)
        if start_char == -1:
            word_surprisals.append((word, np.nan))
            continue
        end_char = start_char + len(word)
        char_pos = end_char

        # Collect token indices that overlap this word
        tok_indices = sorted(set(
            char_to_token[c] for c in range(start_char, end_char)
            if c in char_to_token
        ))

        if not tok_indices:
            word_surprisals.append((word, np.nan))
            continue

        # Sum surprisals across subword tokens (skipping nan for first token)
        word_surp = 0.0
        all_nan   = True
        for ti in tok_indices:
            s = token_surprisals[ti]
            if not np.isnan(s):
                word_surp += s
                all_nan   = False

        word_surprisals.append((word, np.nan if all_nan else word_surp))

    return word_surprisals


def compute_all_surprisals(sentences, model, tokenizer, device):
    """
    Compute surprisal for all sentences.
    Returns dict: global_sent_idx → list of (word_str, surprisal)
    """
    results = {}
    print(f"\nComputing GPT-2 surprisal for {len(sentences)} sentences…")
    for idx, sentence in enumerate(sentences):
        if idx % 50 == 0:
            print(f"  {idx}/{len(sentences)}", end='\r', flush=True)
        try:
            results[idx] = compute_surprisal_sentence(sentence, model, tokenizer, device)
        except Exception as e:
            print(f"\n  [WARN] Sentence {idx} failed: {e}")
            results[idx] = [(w, np.nan) for w in sentence.split()]
    print(f"  {len(sentences)}/{len(sentences)} — done          ")
    return results


# ── covariates ────────────────────────────────────────────────────────────────

def clean_word(word):
    """Strip punctuation for frequency/length lookup."""
    return re.sub(r"[^\w'-]", '', word).lower()


def get_covariates(word, word_idx, n_words):
    """Return dict of baseline covariates for one word."""
    clean = clean_word(word)
    return {
        'word'        : word,
        'word_clean'  : clean,
        'word_length' : len(clean),
        'word_position': word_idx,                          # 0-based
        'word_pos_norm': word_idx / max(n_words - 1, 1),   # normalised 0→1
        'log_freq'    : zipf_frequency(clean, 'en'),        # Zipf scale (0–7)
    }


# ── merge pipeline ────────────────────────────────────────────────────────────

def build_analysis_dataset(et_path, sentences, surprisal_map):
    """
    Merge ET measures with surprisal + covariates.
    Uses (global_sent_idx, word_idx) as the join key.
    """
    et = pd.read_csv(et_path)
    print(f"\nET dataset: {len(et):,} rows")

    records = []
    mismatches = 0

    for g_idx, word_list in surprisal_map.items():
        n_words = len(word_list)
        for w_idx, (word, surprisal) in enumerate(word_list):
            cov = get_covariates(word, w_idx, n_words)
            records.append({
                'global_sent_idx': g_idx,
                'word_idx'       : w_idx,
                'surprisal'      : surprisal,
                **cov
            })

    surprisal_df = pd.DataFrame(records)

    # Merge on (global_sent_idx, word_idx)
    merged = pd.merge(
        et, surprisal_df,
        on=['global_sent_idx', 'word_idx'],
        how='left'
    )

    # Sanity check: word count from wordbounds vs sentence split
    mismatch_rows = merged[merged['word_count'] != merged.groupby('global_sent_idx')['word_idx'].transform('max') + 1]
    if len(mismatch_rows) > 0:
        print(f"  [WARN] {mismatch_rows['global_sent_idx'].nunique()} sentences have "
              f"word count mismatches between wordbounds and GPT-2 tokenisation.")
        print("  This may affect surprisal alignment for those sentences.")

    # Drop sentences where alignment clearly failed (all surprisal NaN for that sentence)
    valid = merged.groupby('global_sent_idx')['surprisal'].transform(lambda x: x.notna().any())
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"  [INFO] {merged.loc[~valid, 'global_sent_idx'].nunique()} sentences "
              f"dropped (complete surprisal failure): {n_dropped} rows")
    merged = merged[valid].copy()

    print(f"Final dataset: {len(merged):,} rows, "
          f"{merged['surprisal'].notna().mean():.1%} have surprisal values")

    return merged


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--et',        default='et_measures.csv')
    parser.add_argument('--sentences', default='sentencesSR.mat')
    parser.add_argument('--out',       default='analysis_data.csv')
    parser.add_argument('--device',    default='cpu',
                        help='cpu | cuda | mps  (use mps on Apple Silicon)')
    args = parser.parse_args()

    # Auto-detect Apple Silicon MPS
    if args.device == 'cpu' and torch.backends.mps.is_available():
        args.device = 'mps'
        print("Apple Silicon detected → using MPS backend")

    device = torch.device(args.device)

    # 1. Load sentences
    sentences = load_sentences(args.sentences)

    # 2. Load GPT-2
    print("\nLoading GPT-2…")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model     = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    print("  GPT-2 loaded")

    # 3. Compute surprisal
    surprisal_map = compute_all_surprisals(sentences, model, tokenizer, device)

    # 4. Build analysis dataset
    final = build_analysis_dataset(args.et, sentences, surprisal_map)

    # 5. Save
    final.to_csv(args.out, index=False)
    print(f"\n✓ Saved → {args.out}")
    print(f"\nColumn summary:")
    key_cols = ['FFD','GD','TRT','nFix','reg','surprisal','word_length','log_freq']
    print(final[[c for c in key_cols if c in final.columns]].describe().round(2).to_string())

    # 6. Quick sanity check
    print("\nSurprisal sanity check (first 5 words of sentence 0):")
    s0 = final[final['global_sent_idx'] == 0][['word','surprisal','word_length','log_freq']].head(5)
    print(s0.to_string(index=False))


if __name__ == '__main__':
    main()
