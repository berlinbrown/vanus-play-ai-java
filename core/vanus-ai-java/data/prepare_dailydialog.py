#!/usr/bin/env python3
"""Convert downloaded DailyDialog ZIPs to Vanus pairs using only the standard library."""
import hashlib
import json
import random
import re
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dailydialog'


def normalize(text):
    text = ' '.join(text.split())
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    return re.sub(r"\s*'\s*", "'", text)


def write_pairs(name, pairs):
    (OUT / name).write_text(''.join(f'{p}\t{a}\n' for p, a in pairs), encoding='utf-8')


def main():
    seen_prompts = set()
    report = {'context': 128, 'source': 'https://huggingface.co/datasets/roskoN/dailydialog', 'splits': {}}
    training = []
    for split in ('train', 'validation', 'test'):
        path = OUT / 'source' / f'{split}.zip'
        with ZipFile(path) as archive:
            lines = archive.read(f'{split}/dialogues_{split}.txt').decode('utf-8').splitlines()
        pairs = set()
        candidates = 0
        for line in lines:
            turns = [normalize(t) for t in line.split('__eou__') if t.strip()]
            for prompt, answer in zip(turns, turns[1:]):
                candidates += 1
                # BOS, USER, ASSISTANT occupy three positions; EOS is the final target.
                if prompt and answer and len((prompt + answer).encode('utf-8')) + 3 <= 128:
                    if prompt not in seen_prompts:
                        pairs.add((prompt, answer))
        pairs = sorted(pairs)
        if not pairs:
            raise ValueError(f'No usable pairs in {split}')
        write_pairs(f'{split}.tsv', pairs)
        seen_prompts.update(p for p, _ in pairs)
        if split == 'train':
            training = pairs
            seen_prompts.update(line.split('\t')[0] for line in (ROOT / 'conversation.tsv').read_text().splitlines())
        report['splits'][split] = {'dialogues': len(lines), 'adjacent_pairs': candidates,
                                  'retained_pairs': len(pairs), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    # A bounded first experiment, retaining the user's known-working greetings.
    greetings = [tuple(line.split('\t')) for line in (ROOT / 'conversation.tsv').read_text().splitlines()]
    greeting_prompts = {p for p, _ in greetings}
    pool = [pair for pair in training if pair[0] not in greeting_prompts]
    starter = greetings + random.Random(42).sample(pool, min(256, len(pool)))
    write_pairs('starter.tsv', starter)
    report['starter_pairs'] = len(starter)
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
