"""Readers shared by table.py, power.py and g3.py: bench_split's raw.tsv and report (.out) files."""
import re
from collections import defaultdict


def role(name):
    """a config's role from raw.tsv's name: ACC, GPU, SPLIT (S<g>), SPLIT2 (S<g>b), SEAM, HEAD, TAIL"""
    return "SPLIT" if re.fullmatch(r"S\d+", name) else "SPLIT2" if re.fullmatch(r"S\d+b", name) else name


def read(path):
    """raw.tsv's rows as dicts, each with its config's role. Files written before raw.tsv had a name
    column: cfg 0 is ACC, 1 GPU and any other the split (the fixed block then had only those three)."""
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    recs = [dict(zip(rows[0], r)) for r in rows[1:]]
    named, roles = "name" in rows[0], {}   # a file has a handful of names
    for d in recs:
        if named:
            d["role"] = roles.get(d["name"]) or roles.setdefault(d["name"], role(d["name"]))
        else:
            d["role"] = {"0": "ACC", "1": "GPU"}.get(d["cfg"], "SPLIT")
    return recs


def fixed_pairs(recs, scale=1.0):
    """the fixed block's (phase 1) lockstep times, the slowest process per repeat, in s times scale:
    {role: {(round, rep): time}}"""
    out = defaultdict(dict)
    for d in recs:
        if d["phase"] != "1":
            continue
        k = (int(d["round"]), int(d["rep"]))
        out[d["role"]][k] = max(out[d["role"]].get(k, 0.0), float(d["total"]) * scale)
    return out


def parse(txt, tag):
    """the key=value fields of the report's <tag> line (SUMMARY or SUSTAIN), or None. Reports written
    before stdout was line-buffered can have the exit-time AB_REPORT inside that line, cutting it short:
    a token without '=' is skipped, and the fields after the cut are missing."""
    m = re.search(rf"^{tag} (.*)$", txt, re.M)
    return dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv) if m else None


def env(txt, name):
    """a variable's value on the report's header line (VECLIB_MAXIMUM_THREADS=8 ...), or None"""
    m = re.search(rf"\b{name}=(\S+)", txt)
    return m.group(1) if m else None


def on_ac(txt):
    """the report's start and end lines both say AC power"""
    return all(re.search(rf"^{w}: Now drawing from 'AC Power'", txt, re.M) for w in ("start", "end"))


def power_problem(txt, min_w=90, min_batt=95):
    """why the report's power does not meet run.sh's conditions, or None: AC power on its start and end
    lines, an adapter of min_w W or more on both (bench_split prints '| adapter N W'), and at the start a
    battery of min_batt percent or more, neither charging nor finishing its charge (when printed)"""
    if not on_ac(txt):
        return "did not start and end on AC power"
    for w in ("start", "end"):
        m = re.search(rf"^{w}: .*\| adapter (\d+) W", txt, re.M)
        if not m or int(m.group(1)) < min_w:
            return f"adapter under {min_w} W or not reported"
    m = re.search(r"^start: .*?(\d+)%; ([^;]+);", txt, re.M)
    if m and (int(m.group(1)) < min_batt or m.group(2).strip() in ("charging", "finishing charge")):
        return f"battery at the start {m.group(1)}%, {m.group(2).strip()}"
    return None
