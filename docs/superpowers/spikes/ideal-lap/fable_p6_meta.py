"""Fable probe 6 (build round 2): what does `metadata` say about the SETUP of a recording?
Read-only. Prints the metadata keys, and for the setup JSON: its top-level structure and a
fingerprint, so files of one car on one track can be compared.
usage: python fable_p6_meta.py [substring ...]   (files in data/sessions whose name contains any substring)"""
import sys, os, json, hashlib
from pathlib import Path
import duckdb
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SESS = Path(os.environ.get("IL_SESSIONS", r"C:\Users\hoang\Projects\lmu-coach\data\sessions"))
subs = sys.argv[1:]
VERBOSE = os.environ.get("FABLE_VERBOSE") == "1"


def flat(o, pre=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{pre}{k}."))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flat(v, f"{pre}{i}."))
    else:
        out[pre[:-1]] = o
    return out


seen = {}
for p in sorted(SESS.glob("*.duckdb")):
    if subs and not any(s.lower() in p.name.lower() for s in subs):
        continue
    try:
        con = duckdb.connect(str(p), read_only=True)
        rows = con.execute("select key, value from metadata").fetchall()
        con.close()
    except Exception as e:
        print(p.name[:70], "ERR", type(e).__name__, str(e)[:80]); continue
    md = dict(rows)
    setup_keys = [k for k, v in md.items() if isinstance(v, str) and len(v) > 200]
    print("=" * 100); print(p.name)
    print("  keys:", ", ".join(f"{k}({len(str(v))})" if len(str(v)) > 60 else f"{k}={v}" for k, v in md.items()))
    for k in setup_keys:
        try:
            js = json.loads(md[k])
        except Exception:
            print("  ", k, "not JSON:", md[k][:200]); continue
        fl = flat(js)
        fp = hashlib.sha1(json.dumps(fl, sort_keys=True).encode()).hexdigest()[:10]
        print(f"  {k}: {len(fl)} leaf values, fingerprint {fp}; top: {list(js.keys())[:30] if isinstance(js, dict) else type(js).__name__}")
        key = (p.name.split("-2026-")[0], k)
        if key in seen and seen[key][0] != fp:
            old = seen[key][1]
            diff = [(a, old.get(a), fl.get(a)) for a in sorted(set(old) | set(fl)) if old.get(a) != fl.get(a)]
            print(f"    DIFFERS from the previous file of this car/track in {len(diff)} values:")
            for a, x, y in diff[:60]:
                print(f"      {a}: {x} -> {y}")
        elif key in seen:
            print("    same setup as the previous file of this car/track")
        seen[key] = (fp, fl)
        if VERBOSE:
            for a, x in list(fl.items())[:400]:
                print(f"      {a} = {x}")
