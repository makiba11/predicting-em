import csv, json, statistics as st, itertools
from collections import defaultdict
RUNS = ["q8lr1e4_v3", "q8lr1e4_v3_k70", "q8lr1e4_v3_k70_opus"]
FS = ["claude-opus-5.5","claude-sonnet-5.5","gpt-6.1-sol","gpt-6-luna","deepseek-v4-flash"]
truth = {f"study/{k}": v for k, v in json.load(open("reports/q8lr1e4/mc_shift.json")).items()}
T = sorted(truth); short = lambda t: t.split("q8lr1e4_")[1]
pred = defaultdict(lambda: defaultdict(list)); draws = defaultdict(list); fam = {}
for run in RUNS:
    for r in csv.DictReader(open(f"reports/forecasting_methodology/{run}/loto_predictions.csv")):
        if r["status"] != "complete" or r["predicted_original_delta_pp"] == "": continue
        k = int(r["k"])
        if k > 0 and ((run == "q8lr1e4_v3") != (k <= 32)): continue
        fam[r["target"]] = r["family"]
        p = float(r["predicted_original_delta_pp"])
        pred[(r["forecaster"], k)][r["target"]].append(p)
        draws[(r["forecaster"], k)].append((r["target"], p, [e for e in r["evidence_ids"].split() if e] if r["evidence_ids"] else []))
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
    i = 0
    while i < len(o):
        j = i
        while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j += 1
        for x in range(i, j+1): r[o[x]] = (i+j)/2
        i = j+1
    return r
def sp(a, b):
    ra, rb = rank(a), rank(b); ma, mb = st.mean(ra), st.mean(rb)
    return sum((x-ma)*(y-mb) for x, y in zip(ra, rb)) / (sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**.5
y = [truth[t] for t in T]
vec = lambda f, k: [st.mean(pred[(f, k)][t]) for t in T]
print("draws per target:", {k: round(len(draws[("claude-opus-5.5", k)])/71, 1) for k in (0,4,8,16,32,48,70)})
print("evidence id example:", draws[("claude-opus-5.5", 4)][0][2][:2])
for k in (0, 32, 48, 70):
    pw = [sp(vec(a, k), vec(b, k)) for a, b in itertools.combinations(FS, 2)]
    print(f"k={k}: model-model spearman {min(pw):.2f}-{max(pw):.2f} (mean {st.mean(pw):.2f}); model-truth", [round(sp(vec(f, k), y), 2) for f in FS])
print("truth: mean %.1f median %.1f max %.1f" % (st.mean(y), st.median(y), max(y)))
for k in (0, 4, 8, 32, 48, 70):
    print(f"k={k} mean pred per model", [round(st.mean(vec(f, k)), 1) for f in FS], "max", [round(max(vec(f, k)), 1) for f in FS])
# consensus ranks
def cons(k):
    rs = [rank(vec(f, k)) for f in FS]; return {t: st.mean(r[i] for r in rs) for i, t in enumerate(T)}
c0, c70, c48 = cons(0), cons(70), cons(48)
em = [t for t in T if truth[t] >= 4]; no = [t for t in T if truth[t] < 4]
print("\nEM conditions by consensus rank at k=0 (0=lowest predicted, 70=highest), -> k=48, k=70:")
for t in sorted(em, key=lambda t: c0[t]): print(f"  {short(t):32s} {fam[t]} true {truth[t]:+5.1f} r0 {c0[t]:4.1f} r48 {c48[t]:4.1f} r70 {c70[t]:4.1f}  p0 {st.mean(st.mean(pred[(f,0)][t]) for f in FS):+.1f} p70 {st.mean(st.mean(pred[(f,70)][t]) for f in FS):+.1f}")
print("non-EM top consensus at k=0:")
for t in sorted(no, key=lambda t: -c0[t])[:12]: print(f"  {short(t):32s} {fam[t]} true {truth[t]:+5.1f} r0 {c0[t]:4.1f} r48 {c48[t]:4.1f} r70 {c70[t]:4.1f}  p0 {st.mean(st.mean(pred[(f,0)][t]) for f in FS):+.1f} p70 {st.mean(st.mean(pred[(f,70)][t]) for f in FS):+.1f}")
# families
fams = defaultdict(list)
for t in T: fams[fam[t]].append(t)
print("\nfamilies:", {f: (len(v), sum(truth[t] >= 4 for t in v)) for f, v in sorted(fams.items())})
# k=4 dip: per-draw predictions vs number of EM cards shown
for k in (4, 8):
    by = defaultdict(list)
    for f in FS:
        for t, p, ev in draws[(f, k)]:
            n = sum(truth.get(e if e.startswith("study/") else "study/"+e, 0) >= 4 for e in ev)
            by[n].append((p, truth[t] >= 4))
    print(f"k={k}: by #EM shown -> (n draws, mean pred EM targets, mean pred nonEM targets)", {n: (len(v), round(st.mean(p for p, e in v if e), 1) if any(e for _, e in v) else None, round(st.mean(p for p, e in v if not e), 1)) for n, v in sorted(by.items())})
print()
for f, v in sorted(fams.items()):
    print(f, [(short(t), round(truth[t], 1)) for t in v])
def auc_pairs(score, sel):
    num = den = 0
    for a in em:
        for b in no:
            if not sel(a, b): continue
            den += 1; num += (score[a] > score[b]) + 0.5 * (score[a] == score[b])
    return num / den if den else None, den
fam_base = {t: st.mean(truth[s] for s in fams[fam[t]] if s != t) for t in T}
print("family-sibling-mean baseline AUROC:", auc_pairs(fam_base, lambda a, b: True))
for k in (0, 32, 48, 70):
    out = []
    for f in FS:
        s = dict(zip(T, vec(f, k)))
        w = auc_pairs(s, lambda a, b: fam[a] == fam[b]); bt = auc_pairs(s, lambda a, b: fam[a] != fam[b])
        out.append((round(w[0], 2), round(bt[0], 2)))
    print(f"k={k} (within-family AUROC, between-family AUROC) per model:", out, "pairs", w[1], bt[1])
# family mean consensus prediction k0 vs k70
print("\nfamily: n EM/n, true mean, mean pred k0 -> k70 (avg over models)")
for f, v in sorted(fams.items()):
    p0 = st.mean(st.mean(pred[(m, 0)][t]) for m in FS for t in v); p70 = st.mean(st.mean(pred[(m, 70)][t]) for m in FS for t in v)
    print(f"  {f} {sum(truth[t]>=4 for t in v)}/{len(v)} true {st.mean(truth[t] for t in v):+.1f} pred {p0:+.1f} -> {p70:+.1f}")
print()
for k in (0, 4, 32, 48, 70):
    rs = [rank(vec(f, k)) for f in FS]; ens = {t: st.mean(r[i] for r in rs) for i, t in enumerate(T)}
    print(f"k={k} 5-model rank-average AUROC {auc_pairs(ens, lambda a,b: True)[0]:.3f}  best single {max(auc_pairs(dict(zip(T, vec(f,k))), lambda a,b: True)[0] for f in FS):.3f}")
