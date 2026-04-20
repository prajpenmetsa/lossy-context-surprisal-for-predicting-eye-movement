"""
Lossy-Context Surprisal (Futrell et al. 2020) for ZuCo-1 SR
=============================================================
For each word w_t in a sentence, samples n_samples degraded contexts
by independently dropping each preceding word w_i with probability
(1 - beta^(t-i)), computes GPT-2 surprisal under each degraded context,
and averages across samples.

beta=1.0 recovers standard surprisal.
beta=0.0 means no context (unigram surprisal).

Grid of betas: 0.1, 0.3, 0.5, 0.7, 0.9, 1.0

Usage:
  python compute_lossy_surprisal.py \
      --sentences sentencesSR.mat \
      --out       lossy_surprisal.csv \
      --samples   20

Output: lossy_surprisal.csv with columns:
  global_sent_idx, word_idx, beta, lossy_surprisal
"""

import argparse
import numpy as np
import pandas as pd
import scipy.io
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_sentences(mat_path):
    data = scipy.io.loadmat(mat_path, squeeze_me=True)
    key = next(k for k in data if not k.startswith('_'))
    sentences = [str(s).strip() for s in np.atleast_1d(data[key])]
    print(f"Loaded {len(sentences)} sentences from {mat_path}")
    return sentences


def lossy_surprisal_sentence(sentence, model, tokenizer, device, beta, n_samples, rng):
    """
    Compute lossy-context surprisal for each word in a sentence.

    For word at position t (0-based):
      - Sample n_samples degraded contexts by keeping word at position i
        with probability beta^(t - i)
      - Compute GPT-2 surprisal of word t given each degraded context
      - Return mean surprisal across samples

    Position 0 always returns NaN (no left context).
    beta=1.0 → all context always kept → standard surprisal.
    """
    words = sentence.split()
    n = len(words)
    result = [np.nan] * n

    for t in range(1, n):
        sample_surprisals = []

        for _ in range(n_samples):
            # Build degraded context
            ctx = []
            for i in range(t):
                if rng.random() < beta ** (t - i):
                    ctx.append(words[i])

            # Target word — prepend space so GPT-2 tokenises it correctly
            target_surface = ' ' + words[t]
            context_str    = ' '.join(ctx)

            # Full degraded input = context + target
            if ctx:
                full_input = context_str + target_surface
            else:
                full_input = words[t]   # no context at all

            enc = tokenizer(full_input, return_tensors='pt', add_special_tokens=False)
            ids = enc['input_ids'].to(device)   # (1, T)
            T   = ids.shape[1]

            if T < 2:
                continue

            # How many tokens does the target word occupy?
            target_enc    = tokenizer(target_surface, add_special_tokens=False)['input_ids']
            n_target_toks = len(target_enc)

            if n_target_toks == 0 or n_target_toks >= T:
                continue

            # Forward pass
            with torch.no_grad():
                logits = model(ids).logits          # (1, T, vocab)
            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)  # (T, vocab)

            # Sum surprisal over target tokens
            surp = 0.0
            for k in range(n_target_toks):
                pos    = T - n_target_toks + k     # position of this target token
                if pos < 1:
                    continue
                tok_id = ids[0, pos].item()
                surp  += -log_probs[pos - 1, tok_id].item() / np.log(2)

            sample_surprisals.append(surp)

        if sample_surprisals:
            result[t] = float(np.mean(sample_surprisals))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences', default='sentencesSR.mat',
                        help='Path to sentencesSR.mat')
    parser.add_argument('--out',       default='lossy_surprisal.csv',
                        help='Output CSV path')
    parser.add_argument('--betas',     default='0.1,0.3,0.5,0.7,0.9,1.0',
                        help='Comma-separated list of beta values')
    parser.add_argument('--samples',   type=int, default=20,
                        help='Number of degraded context samples per word')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    betas = [float(b) for b in args.betas.split(',')]
    rng   = np.random.default_rng(args.seed)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Apple Silicon detected → using MPS backend")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA detected → using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load data
    sentences = load_sentences(args.sentences)

    # Load model
    print("Loading GPT-2...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model     = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    print("GPT-2 loaded\n")

    # Compute
    records = []
    n_sents = len(sentences)
    n_betas = len(betas)
    total   = n_sents * n_betas

    for idx, sent in enumerate(sentences):
        words = sent.split()
        for beta in betas:
            done = idx * n_betas + betas.index(beta)
            print(f"  [{done}/{total}] sent={idx} beta={beta:.1f} '{sent[:40]}...'",
                  end='\r', flush=True)
            try:
                lossy = lossy_surprisal_sentence(
                    sent, model, tokenizer, device, beta, args.samples, rng
                )
            except Exception as e:
                print(f"\n  [WARN] sent {idx} beta {beta}: {e}")
                lossy = [np.nan] * len(words)

            for wi, surp in enumerate(lossy):
                records.append({
                    'global_sent_idx': idx,
                    'word_idx':        wi,
                    'beta':            beta,
                    'lossy_surprisal': surp
                })

    print(f"\n  [{total}/{total}] done                              ")

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f"\n✓ Saved {len(df):,} rows → {args.out}")
    print("\nMean lossy surprisal by beta:")
    print(df.groupby('beta')['lossy_surprisal'].mean().round(3).to_string())


if __name__ == '__main__':
    main()
