"""
ZuCo-1 SR — Word-Level Eye-Tracking Measure Extractor  (v2)
============================================================
Correctly maps SNR1/SNR2 wordbounds to SR1–SR8 sessions by treating
them as a single concatenated sentence list, sliced per session.

Outputs et_measures.csv with columns:
  subject, session, global_sent_idx, local_sent_idx,
  word_idx, word_count, FFD, GD, TRT, nFix, reg

Usage:
  pip install scipy numpy pandas
  python extract_et_measures.py --data_dir zuco_ET/ --out et_measures.csv
"""

import os, re, glob, argparse
import numpy as np
import scipy.io
import pandas as pd


def load_mat(path):
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)


def load_all_wordbounds(subj_dir, subj_id):
    combined = []
    snr1_len = 0
    for block in ['SNR1', 'SNR2']:
        matches = glob.glob(os.path.join(subj_dir, f'wordbounds_{block}_{subj_id}.mat'))
        if not matches:
            matches = glob.glob(os.path.join(subj_dir, f'wordbounds*{block}*{subj_id}*.mat'))
        if not matches:
            print(f'    [WARN] Missing wordbounds {block} for {subj_id}')
            continue
        mat = load_mat(matches[0])
        wb  = mat['wordbounds']
        for i in range(len(wb)):
            combined.append(np.atleast_2d(wb[i]).astype(float))
        if block == 'SNR1':
            snr1_len = len(wb)
    return combined, snr1_len


def get_fixations(mat):
    fix_obj  = mat['eyeevent'].fixations
    fix_data = fix_obj.data
    start = fix_data[:, 0]
    end   = fix_data[:, 1]
    dur   = end - start
    x     = fix_data[:, 3]
    y     = fix_data[:, 4]
    return np.column_stack([start, end, dur, x, y])


def get_sentence_windows(mat):
    ev = mat['event']
    if ev.ndim == 1:
        ev = ev.reshape(-1, 2)
    starts = np.sort(ev[ev[:, 1] == 10, 0])
    ends   = np.sort(ev[ev[:, 1] == 11, 0])
    windows = []
    for ts in starts:
        candidates = ends[ends > ts]
        if len(candidates):
            windows.append((int(ts), int(candidates[0])))
    return windows


def assign_to_word(x, y, word_bounds):
    for i, (xL, yT, xR, yB) in enumerate(word_bounds):
        if xL <= x <= xR and yT <= y <= yB:
            return i
    return -1


def compute_measures(sent_fix, word_bounds):
    n_words = len(word_bounds)
    assignments = []
    for fo, fix in enumerate(sent_fix):
        wi = assign_to_word(fix[3], fix[4], word_bounds)
        assignments.append((fo, wi, fix[2]))

    rows = []
    for wi in range(n_words):
        on_word = [(fo, dur) for fo, widx, dur in assignments if widx == wi]
        if not on_word:
            rows.append(dict(word_idx=wi, FFD=np.nan, GD=np.nan, TRT=np.nan, nFix=0, reg=0))
            continue

        fix_orders = [fo for fo, _ in on_word]
        durs       = [dur for _, dur in on_word]
        first_fo   = min(fix_orders)
        ffd        = durs[fix_orders.index(first_fo)]

        # GD: first-pass reading time
        gd = 0.0
        on_target = False
        for fo, widx, dur in assignments:
            if fo < first_fo:
                continue
            if widx == wi:
                on_target = True
                gd += dur
            elif on_target:
                break
        gd = gd if gd > 0 else np.nan

        trt   = sum(durs)
        n_fix = len(on_word)

        # Regression: fixation on this word was preceded by one to its right
        reg, prev_wi = False, None
        for fo, widx, dur in assignments:
            if widx == wi and prev_wi is not None and prev_wi > wi:
                reg = True
                break
            if widx >= 0:
                prev_wi = widx

        rows.append(dict(word_idx=wi, FFD=ffd, GD=gd, TRT=trt, nFix=n_fix, reg=int(reg)))
    return rows


def process_subject(subj_dir, subj_id):
    all_bounds, snr1_len = load_all_wordbounds(subj_dir, subj_id)
    if not all_bounds:
        return []

    print(f'  Wordbounds: {len(all_bounds)} sentences (SNR1={snr1_len}, SNR2={len(all_bounds)-snr1_len})')

    et_files = sorted(
        glob.glob(os.path.join(subj_dir, f'{subj_id}_SR*_corrected_ET.mat')),
        key=lambda p: int(re.search(r'SR(\d+)', p).group(1))
    )

    all_rows, global_sent_idx = [], 0

    for et_path in et_files:
        session = int(re.search(r'SR(\d+)', os.path.basename(et_path)).group(1))
        try:
            mat       = load_mat(et_path)
            fixations = get_fixations(mat)
            windows   = get_sentence_windows(mat)
        except Exception as e:
            print(f'  [ERROR] Session {session}: {e}')
            continue

        print(f'  Session SR{session}: {len(windows)} sentences (global {global_sent_idx}–{global_sent_idx+len(windows)-1})')

        for local_idx, (t_start, t_end) in enumerate(windows):
            g_idx = global_sent_idx + local_idx
            if g_idx >= len(all_bounds):
                continue

            word_bounds = all_bounds[g_idx]
            mask        = (fixations[:, 0] >= t_start) & (fixations[:, 1] <= t_end)
            sent_fix    = fixations[mask]

            if len(sent_fix) == 0:
                for wi in range(len(word_bounds)):
                    all_rows.append(dict(subject=subj_id, session=session,
                                        global_sent_idx=g_idx, local_sent_idx=local_idx,
                                        word_idx=wi, word_count=len(word_bounds),
                                        FFD=np.nan, GD=np.nan, TRT=np.nan, nFix=0, reg=0))
                continue

            for wr in compute_measures(sent_fix, word_bounds):
                wr.update(dict(subject=subj_id, session=session,
                               global_sent_idx=g_idx, local_sent_idx=local_idx,
                               word_count=len(word_bounds)))
                all_rows.append(wr)

        global_sent_idx += len(windows)

    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='zuco_ET')
    parser.add_argument('--out',      default='et_measures.csv')
    args = parser.parse_args()

    subj_dirs = sorted(glob.glob(os.path.join(args.data_dir, 'Z??')))
    if not subj_dirs:
        print(f'[ERROR] No Z?? folders in {args.data_dir}/')
        return

    all_rows = []
    for subj_dir in subj_dirs:
        subj_id = os.path.basename(subj_dir)
        print(f'\n── {subj_id} ──')
        rows = process_subject(subj_dir, subj_id)
        all_rows.extend(rows)
        print(f'  → {len(rows):,} word observations')

    if not all_rows:
        print('[ERROR] No data extracted.')
        return

    df = pd.DataFrame(all_rows)
    col_order = ['subject','session','global_sent_idx','local_sent_idx',
                 'word_idx','word_count','FFD','GD','TRT','nFix','reg']
    df = df[[c for c in col_order if c in df.columns]]
    df.sort_values(['subject','session','local_sent_idx','word_idx'], inplace=True)
    df.to_csv(args.out, index=False)

    print(f'\n{"="*50}')
    print(f'Saved {len(df):,} rows → {args.out}')
    print(f'Subjects : {df["subject"].nunique()}')
    print(f'Sentences: {df["global_sent_idx"].nunique()}')
    print(f'\nMeasure summary (ms):')
    print(df[['FFD','GD','TRT','nFix']].describe().round(1).to_string())
    print(f'\nFixation coverage: {df["FFD"].notna().mean():.1%} of words had ≥1 fixation')

if __name__ == '__main__':
    main()
