"""Per-group cache of certified bases (critical regions) and Farkas certificates."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from manylp.basis import BasisEntry, FarkasEntry


class CertificatePool:
    """Bounded cache of :class:`BasisEntry` and :class:`FarkasEntry` objects.

    Entries are ranked by (hits, recency); when the pool is full the coldest
    entry is evicted.  A warm-start id that has been evicted simply misses.
    """

    def __init__(self, max_bases: int = 512, max_farkas: int = 64) -> None:
        self.max_bases = max_bases
        self.max_farkas = max_farkas
        self.bases: Dict[int, BasisEntry] = {}
        self.farkas: Dict[int, FarkasEntry] = {}
        self._by_key: Dict[bytes, int] = {}
        #: HiGHS' un-normalised basis key -> entry id (skips re-factorising a known basis)
        self._by_raw: Dict[bytes, int] = {}
        self._fk_by_key: Dict[bytes, int] = {}
        self._next = 0
        self.clock = 0
        self.evictions = 0
        self._materialised = 0

    # -- bases -------------------------------------------------------------------
    def add_basis(self, e: BasisEntry) -> tuple[int, bool]:
        """Insert ``e``; returns ``(id, is_new)`` (deduplicated by basis key)."""
        old = self._by_key.get(e.key)
        if old is not None and old in self.bases:
            return old, False
        eid = self._next
        self._next += 1
        e.id = eid
        e.last_used = self.clock
        self.bases[eid] = e
        self._by_key[e.key] = eid
        if len(self.bases) > self.max_bases:
            self._evict_basis(exclude=eid)
        return eid, True

    def _evict_basis(self, exclude: int) -> None:
        victim = min((e for e in self.bases.values() if e.id != exclude),
                     key=lambda e: (e.hits, e.last_used))
        del self.bases[victim.id]
        self._by_key.pop(victim.key, None)
        self.evictions += 1

    def add_lazy(self, basic: np.ndarray, status: np.ndarray, raw: bytes, anchor) -> int:
        """Register an unfactored candidate basis (prepared only if a later call tries it)."""
        known = self.lookup_raw(raw)
        if known is not None:
            return known.id
        e = BasisEntry(key=raw, basic=basic, status=status, n_np=0, static_unique=False,
                       host={"lazy": True}, anchor=anchor)
        eid, _ = self.add_basis(e)
        self.register_raw(raw, eid)
        return eid

    def lookup_raw(self, raw: bytes) -> Optional[BasisEntry]:
        eid = self._by_raw.get(raw)
        return None if eid is None else self.bases.get(eid)

    def register_raw(self, raw: Optional[bytes], eid: int) -> None:
        if raw is not None:
            self._by_raw[raw] = eid

    def get(self, eid: int) -> Optional[BasisEntry]:
        return self.bases.get(int(eid))

    def ranked(self, limit: int) -> List[BasisEntry]:
        live = (e for e in self.bases.values() if not e.host.get("lazy"))
        return sorted(live, key=lambda e: (-e.hits, -e.last_used))[:limit]

    def _anchor_matrix(self):
        """Anchors of the factorised entries, cached until the pool changes."""
        sig = (self._next, len(self.bases), self.evictions, self._materialised)
        if getattr(self, "_anc_sig", None) != sig:
            cands = [e for e in self.bases.values() if e.anchor is not None and not e.host.get("lazy")]
            if cands:
                a = np.stack([np.concatenate(e.anchor) for e in cands])
                a = np.where(np.isfinite(a), a, 0.0)
                span = np.maximum(a.max(axis=0) - a.min(axis=0), 1e-12)
            else:
                a, span = None, None
            self._anc = (cands, a, span)
            self._anc_sig = sig
        return self._anc

    def mark_materialised(self) -> None:
        self._materialised += 1

    def nearest(self, points: np.ndarray, limit: int) -> List[BasisEntry]:
        """Entries whose anchor bounds are closest (on average) to ``points``."""
        cands, anchors, span = self._anchor_matrix()
        if not cands:
            return []
        pts = np.where(np.isfinite(points), points, 0.0)
        if pts.shape[0] <= 256:
            # each entry's distance to its closest pending member (scaled L1)
            d = np.full(len(cands), np.inf)
            for p in pts:
                d = np.minimum(d, (np.abs(anchors - p) / span).sum(axis=1))
        else:
            # large batches: distance from the batch centroid keeps this O(E*p)
            d = (np.abs(anchors - pts.mean(axis=0)) / span).sum(axis=1)
        order = np.argsort(d)[:limit]
        return [cands[i] for i in order]

    def touch(self, e, hits: int) -> None:
        e.hits += int(hits)
        e.last_used = self.clock

    # -- infeasibility certificates ---------------------------------------------------
    def add_farkas(self, f: FarkasEntry) -> int:
        old = self._fk_by_key.get(f.key)
        if old is not None and old in self.farkas:
            return old
        fid = self._next
        self._next += 1
        f.id = fid
        f.last_used = self.clock
        self.farkas[fid] = f
        self._fk_by_key[f.key] = fid
        if len(self.farkas) > self.max_farkas:
            victim = min((x for x in self.farkas.values() if x.id != fid),
                         key=lambda x: (x.hits, x.last_used))
            del self.farkas[victim.id]
            self._fk_by_key.pop(victim.key, None)
        return fid

    def __len__(self) -> int:
        return len(self.bases)

    @property
    def device_bytes(self) -> int:
        return sum(e.nbytes for e in self.bases.values())
