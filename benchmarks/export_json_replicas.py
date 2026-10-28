"""Plain-JSON replicas of the NPZ arrays in benchmarks/data/.

The scripts keep reading and writing the original files; the replicas only make the same data
available in a human- and machine-readable format:

Every ``benchmarks/data/**/*.npz`` -> ``benchmarks/data/json/<same path>.json``, holding every
array as nested lists (floats written with Python's shortest round-trip representation, so the
values are exact) plus its dtype, shape, axis names and labels.  The arrays too large for a single
file (the 49-policy envelope and the 16-member µODE ensembles) are split along the policy or
member axis into ``<same path>/index.json`` plus one file per policy or member.

Every replica is read back and compared with its source (``--check`` does only that).

    python benchmarks/export_json_replicas.py            # write and check
    python benchmarks/export_json_replicas.py --check    # check only
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

DATA, OUT = "benchmarks/data", "benchmarks/data/json"
UNITS = {"times": "h", "X": "gDW L^-1 (biomass)", "M": "mM (mmol L^-1, extracellular concentration)",
         "mu": "h^-1 (specific growth rate)"}
SPLIT = 1_000_000      # arrays with more values than this are split along the policy / member axis


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"), allow_nan=False)


def _layout(rel, z):
    """Axis names, labels and the split axis of each kind of npz in benchmarks/data."""
    if rel.startswith("alt_optima/"):                       # alt_optima.py: (T, policy, ...)
        labels = {"times": z["times"].tolist(), "policy": z["policies"].tolist(),
                  "species": z["species"].tolist(), "metabolite": z["env_mets"].tolist()}
        axes = {"X": ["times", "policy", "species"], "M": ["times", "policy", "metabolite"],
                "mu": ["times", "policy", "species"]}
        return axes, labels, "policy", "Alternative-optima envelope: 49 flux-selection policies (Table 2, Figure 2)."
    if rel.startswith("dfba/"):                             # bench_dfba.py: (T, member, ...)
        side = json.load(open(f"{DATA}/{rel[:-4]}.json"))
        env = np.load(f"{DATA}/alt_optima/envelope.npz")    # same community, same metabolite order
        assert side["species"] == env["species"].tolist() and np.array_equal(np.load(f"{DATA}/{rel}")["M"][0, 0],
                                                                             env["M"][0, 0])
        labels = {"times": z["times"].tolist(), "member": list(range(z["X"].shape[1])),
                  "species": side["species"], "metabolite": env["env_mets"].tolist()}
        axes = {"X": ["times", "member", "species"], "M": ["times", "member", "metabolite"],
                "mu": ["times", "member", "species"]}
        return axes, labels, None, "Plain-FBA dFBA trajectory of the gut community (Figure 2)."
    if rel.startswith("muode/"):                            # bench_muode.py: (member, T, ...)
        side = json.load(open(f"{DATA}/{rel[:-4]}.json"))
        species = list(side["final_biomass_member0"])
        assert np.array_equal(np.array(list(side["final_biomass_member0"].values())), z["X"][0, -1])
        n = z["X"].shape[1]
        labels = {"member": list(range(z["X"].shape[0])),
                  "times": np.linspace(0.0, side["t_end"], n).tolist(),
                  "species": species, "metabolite": list(range(z["M"].shape[2]))}
        axes = {"X": ["member", "times", "species"], "M": ["member", "times", "metabolite"]}
        note = ("µODE end-to-end dFBA run (Table 3; agreement with HiGHS and COBRApy + GLPK). The npz stores "
                "no labels: 'species' comes from final_biomass_member0 in the companion JSON (checked against "
                "X); 'times' is linspace(0, t_end, steps) as in µODE; metabolites are the community's exchange "
                "metabolites in µODE's column order, given here by column index.")
        return axes, labels, "member", note
    raise ValueError(f"no layout for {rel}")


def _meta(rel, z, axes, labels, note):
    return {"source": f"{DATA}/{rel}", "source_sha256": _sha256(f"{DATA}/{rel}"), "description": note,
            "arrays": {k: {"dtype": str(z[k].dtype), "shape": list(z[k].shape), "axes": axes.get(k),
                           "unit": UNITS.get(k)} for k in z.files},
            "labels": labels}


def write_npz(rel):
    z = np.load(f"{DATA}/{rel}")
    axes, labels, split, note = _layout(rel, z)
    meta = _meta(rel, z, axes, labels, note)
    num = [k for k in z.files if z[k].dtype.kind == "f" and k in axes]
    if split is None or sum(z[k].size for k in num) <= SPLIT:
        _dump({**meta, "data": {k: z[k].tolist() for k in z.files}}, f"{OUT}/{rel[:-4]}.json")
        return 1
    parts = []
    for i, name in enumerate(labels[split]):
        f = f"{i:02d}.json"
        part = {"source": meta["source"], split: name, "index": i,
                "data": {k: np.take(z[k], i, axis=axes[k].index(split)).tolist() for k in num}}
        _dump(part, f"{OUT}/{rel[:-4]}/{f}")
        parts.append({"file": f, split: name})
    _dump({**meta, "split_axis": split, "parts": parts,
           "data": {k: z[k].tolist() for k in z.files if k not in num}}, f"{OUT}/{rel[:-4]}/index.json")
    return len(parts) + 1


def check_npz(rel):
    z = np.load(f"{DATA}/{rel}")
    single = f"{OUT}/{rel[:-4]}.json"
    if os.path.exists(single):
        d = json.load(open(single))
        got = {k: np.array(v, dtype=z[k].dtype) for k, v in d["data"].items()}
    else:
        base = f"{OUT}/{rel[:-4]}"
        idx = json.load(open(f"{base}/index.json"))
        got = {k: np.array(v, dtype=z[k].dtype) for k, v in idx["data"].items()}
        parts = [json.load(open(f"{base}/{p['file']}")) for p in idx["parts"]]
        for k in parts[0]["data"]:
            ax = idx["arrays"][k]["axes"].index(idx["split_axis"])
            got[k] = np.stack([np.array(p["data"][k], dtype=z[k].dtype) for p in parts], axis=ax)
        d = idx
    assert d["source_sha256"] == _sha256(f"{DATA}/{rel}"), f"{rel}: source changed since export"
    assert set(got) == set(z.files), rel
    for k in z.files:
        assert got[k].shape == z[k].shape and np.array_equal(got[k], z[k]), f"{rel}: {k} differs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only compare existing replicas with their sources")
    a = ap.parse_args()
    rels = sorted(os.path.relpath(p, DATA) for p in glob.glob(f"{DATA}/**/*.npz", recursive=True))
    n = 0 if a.check else sum(write_npz(r) for r in rels)
    for r in rels:
        check_npz(r)
    print(f"{len(rels)} npz files -> {n or 'existing'} JSON files, identical on read-back")


if __name__ == "__main__":
    main()
