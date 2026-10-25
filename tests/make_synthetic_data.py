"""SYNTHETIC data in the Kaggle schema (headerless id,entity,sentiment,text) - FOR SMOKE TESTS ONLY.
Includes deliberate defects (empty text, unknown label, duplicates, train/test overlap, conflicting labels)."""
import csv, random, sys
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rnd = random.Random(0)
ents = ["Borderlands", "Facebook", "Microsoft", "Google", "Nvidia", "Verizon", "CallOfDuty", "Amazon"]
words = {
    "Positive": ["love", "great", "awesome", "amazing", "happy", "best", "fun"],
    "Negative": ["hate", "terrible", "awful", "broken", "worst", "angry", "scam"],
    "Neutral": ["update", "release", "today", "news", "patch", "stream", "announce"],
    "Irrelevant": ["pizza", "weather", "football", "cat", "lunch", "holiday", "music"],
}
topics = ["server", "graphics", "price", "support", "battery", "launch"]
def tweet(lab):
    w = rnd.sample(words[lab], 2) + [rnd.choice(topics)] + [rnd.choice(["the", "a", "so", "really"]), rnd.choice(["#fyp", "@bob", "http://t.co/x", "soooo", "don't"])]
    rnd.shuffle(w); return " ".join(w) + rnd.choice(["!", ".", "?", ", ok"])
def rows(n, start):
    r = []
    for i in range(n):
        lab = rnd.choice(list(words)); r.append([start + i, rnd.choice(ents), lab, tweet(lab)])
    return r
tr, te = rows(3000, 1), rows(300, 90000)
tr += [[9001, "Google", "Mixed", "unknown label row"], [9002, "Google", "Positive", ""], tr[5][:], tr[5][:]]
tr += [[9003, "Amazon", "Positive", "same text conflicting"], [9004, "Amazon", "Negative", "same text conflicting"]]
tr += [[9005, "Amazon", te[0][2], te[0][3]]]                      # overlap with test file
for name, r in (("twitter_training.csv", tr), ("twitter_validation.csv", te)):
    with open(out / name, "w", newline="") as f: csv.writer(f).writerows(r)
print("synthetic data written to", out)
