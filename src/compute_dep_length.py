"""
Compute Dependency Length for ZuCo-1 SR sentences
==================================================
For each word, dependency length = |position_of_word - position_of_head|
This is the standard operationalization from Gibson (1998) and
Futrell et al. (2015), used as a predictor of reading difficulty.

Usage:
  pip install spacy
  python -m spacy download en_core_web_sm
  python compute_dep_length.py --sentences sentencesSR.mat --out dep_length.csv

Output: dep_length.csv with columns:
  global_sent_idx, word_idx, word, dep_length, dep_rel, is_root
"""

import argparse
import numpy as np
import pandas as pd
import scipy.io
import spacy


def load_sentences(mat_path):
    data = scipy.io.loadmat(mat_path, squeeze_me=True)
    key  = next(k for k in data if not k.startswith('_'))
    return [str(s).strip() for s in np.atleast_1d(data[key])]


def compute_dep_lengths(sentences, nlp):
    """
    For each sentence, compute per-word dependency length.
    Returns list of dicts.
    """
    records = []
    for idx, sent in enumerate(sentences):
        if idx % 100 == 0:
            print(f"  {idx}/{len(sentences)}", end='\r', flush=True)

        doc = nlp(sent)
        tokens = list(doc)

        for wi, token in enumerate(tokens):
            # Dependency length: distance to syntactic head
            # Root token has itself as head → dep_length = 0
            if token.head == token:
                dep_len = 0
                is_root = 1
            else:
                dep_len = abs(token.i - token.head.i)
                is_root = 0

            records.append({
                'global_sent_idx': idx,
                'word_idx':        wi,
                'word':            token.text,
                'dep_length':      dep_len,
                'dep_rel':         token.dep_,
                'pos':             token.pos_,
                'is_root':         is_root
            })

    print(f"  {len(sentences)}/{len(sentences)} done          ")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences', default='dataset/sentencesSR.mat')
    parser.add_argument('--out',       default='dep_length.csv')
    args = parser.parse_args()

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Loading sentences from {args.sentences}...")
    sentences = load_sentences(args.sentences)
    print(f"Loaded {len(sentences)} sentences\n")

    print("Computing dependency lengths...")
    records = compute_dep_lengths(sentences, nlp)

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)

    print(f"\n✓ Saved {len(df):,} rows → {args.out}")
    print(f"\nDependency length summary:")
    print(df['dep_length'].describe().round(2).to_string())
    print(f"\nMost common dependency relations:")
    print(df['dep_rel'].value_counts().head(10).to_string())

    # Sanity check: show first sentence
    print(f"\nSanity check — sentence 0:")
    print(df[df['global_sent_idx']==0][['word','dep_rel','dep_length','pos']].to_string(index=False))


if __name__ == '__main__':
    main()
