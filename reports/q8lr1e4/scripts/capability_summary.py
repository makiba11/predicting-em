import csv, glob, json, statistics as st
em = {k.removeprefix("q8lr1e4_"): v for k, v in json.load(open("reports/q8lr1e4/mc_shift.json")).items()}
cols = {"mmlu": "capability__mmlu_accuracy", "ife": "capability__ifeval_prompt_strict_accuracy",
        "sr": "capability__strongreject_mean_score", "ref": "capability__strongreject_refusal_rate"}
rows = []
for p in sorted(glob.glob("results/q8lr1e4_*/models.csv")):
    n = p.split("/")[1].removeprefix("q8lr1e4_")
    ms = list(csv.DictReader(open(p)))
    T = [m for m in ms if m["role"] == "treatment"]; C = [m for m in ms if m["role"] != "treatment"]
    def mean(ms, c):
        v = [float(m[c]) for m in ms if m.get(c) not in ("", None)]
        return st.mean(v) if v else None
    r = {"n": n, "status": {m["capability__status"] for m in ms}}
    for k, c in cols.items():
        r[k + "_t"] = mean(T, c); r[k + "_c"] = mean(C, c)
        r[k + "_d"] = mean(T, c + "__delta_vs_same_seed_control")
    r["em"] = em.get(n)
    rows.append(r)
print(len(rows), "studies; statuses", set().union(*[r["status"] for r in rows]))
print("example em value:", rows[0]["n"], rows[0]["em"])
emv = lambda r: (r["em"] if not isinstance(r["em"], dict) else r["em"].get("delta"))
for k in cols:
    d = [r[k + "_d"] for r in rows if r[k + "_d"] is not None]
    print(f"{k}: n={len(d)} mean d={st.mean(d):+.4f} min={min(d):+.4f} max={max(d):+.4f}  T mean={st.mean(r[k+'_t'] for r in rows if r[k+'_t'] is not None):.4f} C mean={st.mean(r[k+'_c'] for r in rows if r[k+'_c'] is not None):.4f}")
    for grp, sel in (("EM", lambda r: emv(r) is not None and emv(r) >= 4), ("noEM", lambda r: emv(r) is not None and emv(r) < 4)):
        g = [r[k + "_d"] for r in rows if sel(r) and r[k + "_d"] is not None]
        print(f"   {grp} n={len(g)} mean d={st.mean(g):+.4f}")
for k in ("mmlu", "ife", "sr", "ref"):
    srt = sorted((r for r in rows if r[k+"_d"] is not None), key=lambda r: r[k + "_d"])
    print(k, "lowest:", [(r["n"], round(r[k+"_d"], 3)) for r in srt[:4]], "highest:", [(r["n"], round(r[k+"_d"], 3)) for r in srt[-4:]])
def spearmanr(a,b):
    rk=lambda v:{i:r for r,i in enumerate(sorted(range(len(v)),key=lambda i:v[i]))}
    ra,rb=rk(a),rk(b);n=len(a);ma=st.mean(ra.values());mb=st.mean(rb.values())
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n));return num/(sum((ra[i]-ma)**2 for i in range(n))*sum((rb[i]-mb)**2 for i in range(n)))**.5
for k in ("mmlu","ife","sr","ref"):
    xs=[r[k+"_d"] for r in rows]; ys=[emv(r) for r in rows]
    print(k, "spearman vs EM shift", spearmanr(xs,ys))
print("sr d>0.1:", sum(r["sr_d"]>0.1 for r in rows), " ref d<-0.1:", sum(r["ref_d"]<-0.1 for r in rows), " ife d<-0.05:", sum(r["ife_d"]<-0.05 for r in rows), " |mmlu|>0.01:", sum(abs(r["mmlu_d"])>0.01 for r in rows))
print("control ranges: mmlu", min(r["mmlu_c"] for r in rows), max(r["mmlu_c"] for r in rows), "ife", min(r["ife_c"] for r in rows), max(r["ife_c"] for r in rows), "sr", min(r["sr_c"] for r in rows), max(r["sr_c"] for r in rows), "ref", min(r["ref_c"] for r in rows), max(r["ref_c"] for r in rows))
