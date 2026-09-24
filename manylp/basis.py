"""Certified bases and the batched certification kernel.

Theory (all statements are for the z-space form of :mod:`manylp.lp`)
--------------------------------------------------------------------
A *basis* is a set ``Bset`` of ``m`` columns of ``Abar`` plus, for every other
(nonbasic) variable, a status: at its lower bound, at its upper bound, or at zero
(free).  Given a member's bounds, the nonbasic values ``z_N`` follow from the
statuses and the basic values are

    z_B = -B^{-1} Abar_N z_N  =  h + T z_NP                                  (1)

where ``z_NP`` are the nonbasic *parametric* variables, ``T = -B^{-1} Abar_NP``
and ``h`` collects the (fixed) static nonbasics.  So within one basis the
solution is an affine function of the member's bounds.

Stage-``k`` reduced costs ``d_k = c_k - Abar^T B^{-T} c_{k,B}`` do **not** depend on
bounds.  Call the basis *lexicographically dual feasible* when, for every
nonbasic ``j``, the first nonzero entry of ``(d_1j, ..., d_Kj)`` has the sign its
status requires (``<= 0`` at lower, ``>= 0`` at upper; all zero if free at zero),
unless ``j`` is fixed (``lb_j == ub_j``), in which case any sign is fine.

**Theorem.**  If a basis is lexicographically dual feasible for a member and (1)
lies within that member's bounds, then ``z`` is the lexicographic optimum of the
member's LP.  If moreover every nonbasic, non-fixed ``j`` has a nonzero
lex-reduced-cost vector, that optimum is unique.

*Proof sketch.*  For any feasible ``z``, ``Abar z = 0`` gives
``c_k^T z = sum_{j in N} d_kj z_j``.  Stage 1: each term is maximised by ``z_j`` at the
bound its sign points to, which is what the status says, so ``z`` is optimal and
the optimal face is ``F_1 = {feasible z : z_j = bound_j whenever d_1j != 0}``.
On ``F_1`` only nonbasics with ``d_1j = 0`` can move, and their ``d_2j`` are
correctly signed, so ``z`` maximises ``c_2`` over ``F_1``; induct.  If every
nonbasic has a nonzero lex vector, ``F_K`` pins every ``z_N``, hence ``z_B`` by (1).

Certification of one member is therefore a GEMV (a GEMM for a batch) plus
bound comparisons, and it is *exact*: the returned point is a basic optimal
solution to floating-point accuracy, not a first-order approximation.

Infeasibility is certified symmetrically by a Farkas vector ``y``: with
``r = Abar^T y`` every feasible ``z`` has ``r^T z = 0``, so the member is
infeasible when ``max_{lb<=z<=ub} r^T z < 0`` (or ``min > 0``) -- a batched
reduction over the member's bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse.linalg as spla

from manylp.backend import Backend
from manylp.lp import LexLP

BASIC = -1
AT_LOWER = 0
AT_UPPER = 1
AT_ZERO = 2


class CertificateError(Exception):
    """The candidate basis cannot be turned into a valid certified entry."""


@dataclass
class Tolerances:
    #: primal feasibility: ``lb - (abs + rel*|lb|) <= z <= ub + (abs + rel*|ub|)``
    primal_abs: float = 1e-9
    primal_rel: float = 1e-9
    #: a reduced cost is nonzero when ``|d| > dual * max(1, ||c_k||_inf)``
    dual: float = 1e-9
    #: Farkas margin, relative to ``sum_j |r_j| max(1, |bound_j|)``
    farkas: float = 1e-9


def lex_signs(D: np.ndarray, scales: np.ndarray, tol: float) -> np.ndarray:
    """Sign of the first entry of each column of ``D`` exceeding ``tol*scale``."""
    K, N = D.shape
    s = np.zeros(N, dtype=np.int8)
    undecided = np.ones(N, dtype=bool)
    for k in range(K):
        big = undecided & (np.abs(D[k]) > tol * scales[k])
        s[big] = np.sign(D[k][big]).astype(np.int8)
        undecided &= ~big
    return s


def _classify_outputs(out_z: np.ndarray, pos_of_basic: np.ndarray, pos_of_np: np.ndarray,
                      z_static_full: np.ndarray):
    """Split requested z-indices into (from basic rows, from z_NP, static constants)."""
    rb = pos_of_basic[out_z]
    rn = pos_of_np[out_z]
    ob = np.nonzero(rb >= 0)[0]
    on = np.nonzero((rb < 0) & (rn >= 0))[0]
    os_ = np.nonzero((rb < 0) & (rn < 0))[0]
    return (ob, rb[ob]), (on, rn[on]), (os_, z_static_full[out_z[os_]])


@dataclass
class BasisEntry:
    """A certified basis and its affine solution law.

    The law is anchored at the *template* bounds::

        z_B = h_tmpl + T_act (z_NP - z_NP_tmpl)[act]

    and columns of ``T = -B^{-1} Abar_NP`` are computed lazily: only for
    nonbasic parametric variables whose value actually differs from the template
    in some member seen so far (in dFBA: the handful of uptake-limited
    exchanges, not every exchange).  The host keeps the sparse LU of ``B`` to add
    columns on demand; everything the batched kernel touches lives in ``dev``.
    """

    key: bytes
    basic: np.ndarray                 # (m,) z-indices, host
    status: np.ndarray                # (N,) int8, host
    n_np: int                         # |nonbasic parametric|
    static_unique: bool
    host: dict = field(default_factory=dict)
    #: kernel arrays on the host (always) and on the device (GPU backends)
    harr: dict = field(default_factory=dict)
    dev: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    hits: int = 0
    last_used: int = 0
    #: parametric bounds of the member this basis was first computed for (host)
    anchor: Optional[tuple] = None
    id: int = -1

    @property
    def nbytes(self) -> int:
        return int(sum(getattr(a, "nbytes", 0) for a in self.dev.values()))

    @property
    def n_active(self) -> int:
        return int(self.host["act_pos"].size)


def basis_key(basic: np.ndarray, status: np.ndarray) -> bytes:
    st = status.astype(np.int8).copy()
    st[basic] = BASIC
    return np.sort(basic).astype(np.int64).tobytes() + st.tobytes()


def prepare_basis_entry(
    lp: LexLP,
    basic: np.ndarray,
    status: np.ndarray,
    tol: Tolerances,
    out_z: np.ndarray,
    source_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> BasisEntry:
    """Factorise and certify a basis on the host (thread-safe, no device work).

    ``status`` is in z-space (``BASIC``/``AT_LOWER``/``AT_UPPER``/``AT_ZERO``).
    ``source_bounds`` -- the full z-space ``(lb, ub)`` of the member the basis came
    from -- (a) lets statuses of variables *fixed* in that member be normalised
    to the side their reduced cost prefers, which maximises the region of bounds
    the entry certifies, and (b) seeds the lazily computed columns of ``T``.

    Raises :class:`CertificateError` if the basis is singular or not
    lexicographically dual feasible on its static part.  Call :func:`upload_entry`
    before certifying with it.
    """
    m, N = lp.m, lp.N
    basic = np.asarray(basic, dtype=np.int64).reshape(-1)
    if basic.size != m or np.unique(basic).size != m:
        raise CertificateError(f"basis has {np.unique(basic).size} distinct columns, need {m}")
    status = np.asarray(status, dtype=np.int8).copy()
    status[basic] = BASIC
    nonbasic = status != BASIC

    try:
        lu = spla.splu(lp.Abar[:, basic].tocsc(), permc_spec="COLAMD")
    except RuntimeError as e:  # exactly singular
        raise CertificateError(f"singular basis: {e}") from e

    # ---- lexicographic reduced costs (bound independent) -------------------
    Cb = lp.C_z[:, basic]                                  # K x m
    Y = lu.solve(np.ascontiguousarray(Cb.T), trans="T")    # m x K
    if Y.ndim == 1:
        Y = Y[:, None]
    D = lp.C_z - (lp.AbarT @ Y).T                          # K x N
    D[:, basic] = 0.0
    s = lex_signs(D, lp.objective_scales(), tol.dual)

    tmpl_fixed = lp.lb_z == lp.ub_z
    if source_bounds is not None:
        src_lb = np.where(lp.is_param, source_bounds[0], lp.lb_z)
        src_ub = np.where(lp.is_param, source_bounds[1], lp.ub_z)
    else:
        src_lb, src_ub = lp.lb_z, lp.ub_z

    def _mismatch(st):
        return nonbasic & (((st == AT_LOWER) & (s > 0)) | ((st == AT_UPPER) & (s < 0))
                           | ((st == AT_ZERO) & (s != 0)))

    # Point every nonbasic whose status contradicts its lexicographic reduced
    # cost at the bound the cost prefers.  For a fixed variable this changes
    # nothing but makes the entry valid once the bound opens; for a *nearly*
    # fixed one (lb ~ ub within HiGHS' tolerance, e.g. a CFL-capped uptake) the
    # value-based status guess was ambiguous.  Always safe: primal feasibility of
    # the resulting point is re-checked by `certify` for every member.
    mismatch = _mismatch(status)
    status[mismatch & (s > 0) & np.isfinite(src_ub)] = AT_UPPER
    status[mismatch & (s < 0) & np.isfinite(src_lb)] = AT_LOWER
    mismatch = _mismatch(status)
    static_nb = nonbasic & ~lp.is_param
    if np.any(mismatch & static_nb & ~tmpl_fixed):
        bad = np.nonzero(mismatch & static_nb & ~tmpl_fixed)[0][:5]
        raise CertificateError(f"not lexicographically dual feasible at static columns {bad}")

    # ---- static nonbasic values ------------------------------------------------
    sidx = np.nonzero(static_nb)[0]
    st_s = status[sidx]
    zs = np.where(st_s == AT_LOWER, lp.lb_z[sidx], np.where(st_s == AT_UPPER, lp.ub_z[sidx], 0.0))
    if not np.all(np.isfinite(zs)):
        raise CertificateError("static nonbasic variable sits at an infinite bound")
    zero_s = st_s == AT_ZERO
    if np.any(zero_s & ((lp.lb_z[sidx] > 0) | (lp.ub_z[sidx] < 0))):
        raise CertificateError("static free nonbasic at zero lies outside its bounds")

    # ---- parametric nonbasics: template anchor ---------------------------------
    np_idx = np.nonzero(nonbasic & lp.is_param)[0]
    n_np = np_idx.size
    np_status = status[np_idx]
    np_s = s[np_idx]
    tmpl_val = np.where(np_status == AT_LOWER, lp.lb_z[np_idx],
                        np.where(np_status == AT_UPPER, lp.ub_z[np_idx], 0.0))
    tmpl_val = np.where(np.isfinite(tmpl_val), tmpl_val, 0.0)

    rhs = np.zeros(m)
    if sidx.size:
        rhs += lp.Abar[:, sidx] @ zs
    if n_np:
        rhs += lp.Abar[:, np_idx] @ tmpl_val
    h = -lu.solve(rhs)

    # columns of T needed by the source member (value differs from the template)
    if source_bounds is not None and n_np:
        lb_s, ub_s = source_bounds
        src_val = np.where(np_status == AT_LOWER, lb_s[np_idx],
                           np.where(np_status == AT_UPPER, ub_s[np_idx], 0.0))
        act_pos = np.nonzero(np.isfinite(src_val) & (src_val != tmpl_val))[0]
    else:
        act_pos = np.zeros(0, dtype=np.int64)
    T_act = _t_columns(lp, lu, np_idx[act_pos])

    static_unique = bool(np.all((s[sidx] != 0) | tmpl_fixed[sidx]))

    # ---- basic-variable bounds for the primal check -------------------------
    b_param = lp.is_param[basic]
    lo_b = lp.lb_z[basic].copy()
    hi_b = lp.ub_z[basic].copy()
    lo_b[b_param] = -np.inf     # checked separately against member bounds
    hi_b[b_param] = np.inf
    bp_rows = np.nonzero(b_param)[0]
    bp_pidx = lp.param_pos[basic[bp_rows]]

    # ---- objectives ----------------------------------------------------------
    obj_const = lp.C_z[:, sidx] @ zs if sidx.size else np.zeros(lp.K)
    Cnp = lp.C_z[:, np_idx].T                              # n_np x K

    # ---- outputs -------------------------------------------------------------
    pos_of_basic = np.full(N, -1, dtype=np.int64)
    pos_of_basic[basic] = np.arange(m)
    pos_of_np = np.full(N, -1, dtype=np.int64)
    pos_of_np[np_idx] = np.arange(n_np)
    z_static_full = np.zeros(N)
    z_static_full[sidx] = zs
    (ob, obr), (on, onr), (os_, osv) = _classify_outputs(out_z, pos_of_basic, pos_of_np, z_static_full)

    needs_fixed = mismatch[np_idx]
    host = {
        "lu": lu,
        "np_idx": np_idx,
        "act_pos": act_pos.astype(np.int64),
        "active": np.isin(np.arange(n_np), act_pos),
        "T_act": T_act,                                    # m x n_act (host copy until upload)
        "arrays": {
            "h": h,
            "np_pidx": lp.param_pos[np_idx],
            "np_tmpl": tmpl_val,
            "np_lower": np_status == AT_LOWER,
            "np_upper": np_status == AT_UPPER,
            "np_zero": np_status == AT_ZERO,
            "np_needs_fixed": needs_fixed,
            "np_nonunique": np_s == 0,
            "lo_b": lo_b,
            "hi_b": hi_b,
            "bp_rows": bp_rows,
            "bp_pidx": bp_pidx,
            "Cb": np.ascontiguousarray(Cb.T),                # m x K
            "Cnp": np.ascontiguousarray(Cnp),
            "obj_const": obj_const,
            "ob": ob, "obr": obr, "on": on, "onr": onr, "os": os_, "osv": osv,
        },
    }
    flags = {
        "any_zero": bool(np.any(np_status == AT_ZERO)),
        "any_needs_fixed": bool(np.any(needs_fixed)),
        "any_nonunique": bool(np.any(np_s == 0)),
        "n_bp": int(bp_rows.size),
    }
    return BasisEntry(key=basis_key(basic, status), basic=basic, status=status, n_np=n_np,
                      static_unique=static_unique, host=host, flags=flags)


def _t_columns(lp: LexLP, lu, cols: np.ndarray) -> np.ndarray:
    """``-B^{-1} Abar[:, cols]`` as a dense ``m x len(cols)`` array."""
    if cols.size == 0:
        return np.zeros((lp.m, 0))
    T = -lu.solve(lp.Abar[:, cols].toarray())
    return T[:, None] if T.ndim == 1 else T


def upload_entry(entry: BasisEntry, backend: Backend) -> BasisEntry:
    """Finalise an entry: host arrays always, device arrays on a GPU backend."""
    arrs = entry.host.pop("arrays")
    T_act = entry.host.pop("T_act")
    arrs["Tt_act"] = np.ascontiguousarray(T_act.T)                       # n_act x m
    arrs["act_pos"] = entry.host["act_pos"]
    m = arrs["h"].shape[0]
    n_np = entry.n_np
    # index maps used by the fused GPU kernels
    act_slot = np.full(n_np, -1, dtype=np.int32)
    act_slot[entry.host["act_pos"]] = np.arange(entry.host["act_pos"].size, dtype=np.int32)
    arrs["act_slot"] = act_slot
    st = np.where(arrs["np_upper"], 1, np.where(arrs["np_lower"], 0, 2)).astype(np.int8)
    arrs["np_st"] = st
    bp_of_row = np.full(m, -1, dtype=np.int64)
    bp_of_row[arrs["bp_rows"]] = arrs["bp_pidx"]
    arrs["bp_of_row"] = bp_of_row
    n_out = arrs["ob"].size + arrs["on"].size + arrs["os"].size
    kind = np.zeros(n_out, dtype=np.int8)
    idx = np.zeros(n_out, dtype=np.int64)
    val = np.zeros(n_out)
    kind[arrs["ob"]], idx[arrs["ob"]] = 0, arrs["obr"]
    kind[arrs["on"]], idx[arrs["on"]] = 1, arrs["onr"]
    kind[arrs["os"]], val[arrs["os"]] = 2, arrs["osv"]
    arrs["out_kind"], arrs["out_idx"], arrs["out_val"] = kind, idx, val
    entry.harr = arrs
    if backend.is_gpu:
        with backend.device_ctx():
            entry.dev = {k: backend.asarray(v) for k, v in arrs.items()}
            for k in ("np_needs_fixed", "np_nonunique"):
                entry.dev[k + "_u8"] = backend.asarray(arrs[k].astype(np.uint8))
    else:
        entry.dev = entry.harr
    return entry


def build_basis_entry(lp: LexLP, basic, status, backend: Backend, tol: Tolerances,
                      out_z: np.ndarray, source_bounds=None) -> BasisEntry:
    """:func:`prepare_basis_entry` followed by :func:`upload_entry`."""
    return upload_entry(prepare_basis_entry(lp, basic, status, tol, out_z, source_bounds), backend)


def ensure_active(entry: BasisEntry, lp: LexLP, need: np.ndarray, backend: Backend) -> bool:
    """Add columns of ``T`` for NP positions in the boolean mask ``need``.

    Returns ``True`` if the law changed (callers must then re-evaluate)."""
    new = np.nonzero(need & ~entry.host["active"])[0]
    if new.size == 0:
        return False
    Tn = np.ascontiguousarray(_t_columns(lp, entry.host["lu"], entry.host["np_idx"][new]).T)
    h = entry.harr
    start = entry.host["act_pos"].size
    h["Tt_act"] = np.concatenate([h["Tt_act"], Tn], axis=0)
    entry.host["act_pos"] = np.concatenate([entry.host["act_pos"], new])
    entry.host["active"][new] = True
    h["act_pos"] = entry.host["act_pos"]
    h["act_slot"][new] = np.arange(start, start + new.size, dtype=np.int32)
    if backend.is_gpu:
        with backend.device_ctx():
            for k in ("Tt_act", "act_pos", "act_slot"):
                entry.dev[k] = backend.asarray(h[k])
    return True


@dataclass
class CertifyResult:
    ok: np.ndarray          # (Bg,) bool
    obj: np.ndarray         # (Bg, K)
    z_out: Optional[np.ndarray]   # (Bg, n_out)
    unique: np.ndarray      # (Bg,) bool


def certify(entry: BasisEntry, Lp, Up, backend: Backend, tol: Tolerances,
            n_out: int, want_out: bool = True, lp: Optional[LexLP] = None) -> CertifyResult:
    """Evaluate the affine law of ``entry`` for a batch and certify each member.

    ``Lp``/``Up`` hold the members' parametric bounds, shape ``(Bg, p)``: NumPy
    arrays select the host path, CuPy arrays the fused GPU path.  Results are
    always host arrays.  Cost: one ``(Bg x n_act) @ (n_act x m)`` GEMM plus
    O(Bg*(m + n_np)) fused checks.  ``lp`` is needed to extend the lazy law.
    """
    if isinstance(Lp, np.ndarray):
        return _certify_host(entry, Lp, Up, tol, n_out, want_out, lp, backend)
    return _certify_gpu(entry, Lp, Up, backend, tol, n_out, want_out, lp)


#: host groups at least this large use the fused multi-threaded Numba kernels
HOST_FUSED_MIN_BATCH = 16


def _certify_host(entry, Lp, Up, tol, n_out, want_out, lp, backend) -> CertifyResult:
    from manylp import cpu_kernels

    if cpu_kernels.available() and Lp.shape[0] >= HOST_FUSED_MIN_BATCH and entry.n_np:
        return _certify_host_fused(entry, Lp, Up, tol, n_out, want_out, lp, backend)
    return _certify_host_numpy(entry, Lp, Up, tol, n_out, want_out, lp, backend)


def _certify_host_fused(entry, Lp, Up, tol, n_out, want_out, lp, backend) -> CertifyResult:
    from manylp import cpu_kernels as ck

    d = entry.harr
    Bg = Lp.shape[0]
    m = d["h"].shape[0]
    n_np = entry.n_np
    Lp = np.ascontiguousarray(Lp)
    Up = np.ascontiguousarray(Up)
    while True:
        n_act = entry.n_active
        ok = np.ones(Bg, dtype=np.bool_)
        uniq = np.full(Bg, entry.static_unique, dtype=np.bool_)
        zNP = np.empty((Bg, n_np))
        dact = np.empty((Bg, n_act))
        need = np.zeros(n_np, dtype=np.int8)
        ck.np_prep(Lp, Up, d["np_pidx"], d["np_st"], d["np_tmpl"], d["np_needs_fixed"],
                   d["np_nonunique"], d["act_slot"], n_act, zNP, dact, ok, uniq, need)
        if need.any():
            if lp is None:
                raise ValueError("certify needs `lp` to extend the affine law")
            ensure_active(entry, lp, need.astype(bool), backend)
            d = entry.harr
            continue
        break
    zB = dact @ d["Tt_act"] if n_act else np.zeros((Bg, m))
    ck.basic_check(zB, d["h"], d["lo_b"], d["hi_b"], d["bp_of_row"], Lp, Up,
                   tol.primal_abs, tol.primal_rel, ok)
    obj = zB @ d["Cb"] + d["obj_const"]
    obj += zNP @ d["Cnp"]
    z_out = None
    if want_out and n_out:
        z_out = np.empty((Bg, n_out))
        ck.gather_out(zB, zNP, d["out_kind"], d["out_idx"], d["out_val"], z_out)
    return CertifyResult(ok=ok, obj=obj, z_out=z_out, unique=uniq)


def _certify_host_numpy(entry, Lp, Up, tol, n_out, want_out, lp, backend) -> CertifyResult:
    d = entry.harr
    f = entry.flags
    Bg = Lp.shape[0]
    if entry.n_np:
        Lnp = Lp[:, d["np_pidx"]]
        Unp = Up[:, d["np_pidx"]]
        zNP = np.where(d["np_upper"], Unp, np.where(d["np_lower"], Lnp, 0.0))
        finite = np.isfinite(zNP)
        ok = finite.all(axis=1)
        if f["any_zero"] or f["any_needs_fixed"] or f["any_nonunique"]:
            fixed = Lnp == Unp
        if f["any_zero"]:
            ok &= np.all(~d["np_zero"] | ((Lnp <= 0.0) & (Unp >= 0.0)), axis=1)
        if f["any_needs_fixed"]:
            ok &= np.all(fixed | ~d["np_needs_fixed"], axis=1)
        zNP = np.where(finite, zNP, d["np_tmpl"])
        delta = zNP - d["np_tmpl"]
        varying = np.any(delta != 0.0, axis=0)
        if np.any(varying & ~entry.host["active"]):
            if lp is None:
                raise ValueError("certify needs `lp` to extend the affine law")
            ensure_active(entry, lp, varying, backend)
        if entry.n_active:
            zB = delta[:, d["act_pos"]] @ d["Tt_act"]
            zB += d["h"]
        else:
            zB = np.broadcast_to(d["h"], (Bg, d["h"].shape[0])).copy()
    else:
        zNP = np.zeros((Bg, 0))
        ok = np.ones(Bg, dtype=bool)
        zB = np.broadcast_to(d["h"], (Bg, d["h"].shape[0])).copy()

    lo, hi = d["lo_b"], d["hi_b"]
    with np.errstate(invalid="ignore"):
        ok &= np.all(lo - zB <= tol.primal_abs + tol.primal_rel * np.abs(lo), axis=1)
        ok &= np.all(zB - hi <= tol.primal_abs + tol.primal_rel * np.abs(hi), axis=1)
        if f["n_bp"]:
            zbp = zB[:, d["bp_rows"]]
            Lb = Lp[:, d["bp_pidx"]]
            Ub = Up[:, d["bp_pidx"]]
            ok &= np.all(Lb - zbp <= tol.primal_abs + tol.primal_rel * np.abs(Lb), axis=1)
            ok &= np.all(zbp - Ub <= tol.primal_abs + tol.primal_rel * np.abs(Ub), axis=1)

    if f["any_nonunique"]:
        unique = np.all(fixed | ~d["np_nonunique"], axis=1) & entry.static_unique
    else:
        unique = np.full(Bg, entry.static_unique, dtype=bool)

    obj = zB @ d["Cb"] + d["obj_const"]
    if entry.n_np:
        obj += zNP @ d["Cnp"]

    z_out = None
    if want_out and n_out:
        z_out = np.empty((Bg, n_out))
        z_out[:, d["ob"]] = zB[:, d["obr"]]
        if d["on"].size:
            z_out[:, d["on"]] = zNP[:, d["onr"]]
        if d["os"].size:
            z_out[:, d["os"]] = d["osv"]
    return CertifyResult(ok=ok, obj=obj, z_out=z_out, unique=unique)


def _certify_gpu(entry, Ld, Ud, backend, tol, n_out, want_out, lp) -> CertifyResult:
    from manylp.kernels import launch

    cp = backend.xp
    d = entry.dev
    Bg, p = Ld.shape
    m = d["h"].shape[0]
    n_np = entry.n_np
    while True:
        ok = cp.ones(Bg, dtype=cp.int32)
        uniq = cp.full(Bg, 1 if entry.static_unique else 0, dtype=cp.int32)
        n_act = entry.n_active
        zNP = cp.empty((Bg, n_np))
        dact = cp.empty((Bg, n_act))
        need = cp.zeros(max(n_np, 1), dtype=cp.int32)
        if n_np:
            launch("np_prep", Bg * n_np, (
                Ld, Ud, cp.int64(p), d["np_pidx"], d["np_st"], d["np_tmpl"],
                d["np_needs_fixed_u8"], d["np_nonunique_u8"], d["act_slot"],
                cp.int64(n_act), cp.int64(n_np), cp.int64(Bg), zNP, dact, ok, uniq, need))
        if n_act:
            zB = dact @ d["Tt_act"]
        else:
            zB = cp.zeros((Bg, m))
        launch("basic_check", Bg * m, (
            zB, d["h"], d["lo_b"], d["hi_b"], d["bp_of_row"], Ld, Ud, cp.int64(p),
            cp.float64(tol.primal_abs), cp.float64(tol.primal_rel), cp.int64(m), cp.int64(Bg), ok))
        if n_np:
            need_h = need.get().astype(bool)
            if need_h.any():
                # a parameter moved that no member had moved before: add its column, redo
                ensure_active(entry, lp, need_h, backend)
                d = entry.dev
                continue
        break
    obj = zB @ d["Cb"] + d["obj_const"]
    if n_np:
        obj += zNP @ d["Cnp"]
    z_out = None
    if want_out and n_out:
        z_out_d = cp.empty((Bg, n_out))
        launch("gather_out", Bg * n_out, (zB, zNP, d["out_kind"], d["out_idx"], d["out_val"],
                                         cp.int64(m), cp.int64(n_np), cp.int64(n_out), cp.int64(Bg),
                                         z_out_d))
        z_out = z_out_d
    # only the validity mask comes back now; values stay on the device for the
    # caller to scatter there and download once per batch
    return CertifyResult(ok=ok.get().astype(bool), obj=obj, z_out=z_out, unique=uniq != 0)


def _to_host_pinned(a) -> np.ndarray:
    """Device -> host through page-locked memory (~2-3x faster for large arrays)."""
    import cupyx

    if a.nbytes < (1 << 20):
        return a.get()
    buf = cupyx.empty_pinned(a.shape, dtype=a.dtype)
    a.get(out=buf)
    return np.array(buf, copy=True)


def explain_certify(entry: BasisEntry, lp: LexLP, lp_row: np.ndarray, up_row: np.ndarray,
                    tol: Tolerances, backend: Backend) -> dict:
    """Why does (or doesn't) ``entry`` certify one member?  For diagnostics."""
    d = entry.harr
    Lnp, Unp = lp_row[d["np_pidx"]], up_row[d["np_pidx"]]
    zNP = np.where(d["np_upper"], Unp, np.where(d["np_lower"], Lnp, 0.0))
    out = {"nonfinite_np": int((~np.isfinite(zNP)).sum()),
           "needs_fixed_violations": int((d["np_needs_fixed"] & (Lnp != Unp)).sum())}
    delta = np.where(np.isfinite(zNP), zNP, d["np_tmpl"]) - d["np_tmpl"]
    res = _certify_host_numpy(entry, lp_row[None, :], up_row[None, :], tol, 0, False, lp, backend)
    d = entry.harr
    zB = d["h"] + (delta[d["act_pos"]] @ d["Tt_act"] if entry.n_active else 0.0)
    lo, hi = d["lo_b"].copy(), d["hi_b"].copy()
    lo[d["bp_rows"]] = lp_row[d["bp_pidx"]]
    hi[d["bp_rows"]] = up_row[d["bp_pidx"]]
    viol = np.maximum(lo - zB, zB - hi)
    mag = np.maximum(np.abs(np.where(np.isfinite(lo), lo, 0.0)), np.abs(np.where(np.isfinite(hi), hi, 0.0)))
    out["max_primal_violation"] = float(viol.max()) if viol.size else 0.0
    out["n_primal_violations"] = int((viol > tol.primal_abs + tol.primal_rel * mag).sum())
    out["ok"] = bool(res.ok[0])
    return out


# ---------------------------------------------------------------------------
# infeasibility certificates
# ---------------------------------------------------------------------------


@dataclass
class FarkasEntry:
    key: bytes
    dev: dict
    max_static: float
    min_static: float
    scale_static: float
    hits: int = 0
    last_used: int = 0
    id: int = -1


def _box_extremes(r: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> tuple[float, float, float]:
    with np.errstate(invalid="ignore"):
        hi = np.where(r > 0, r * ub, np.where(r < 0, r * lb, 0.0))
        lo = np.where(r > 0, r * lb, np.where(r < 0, r * ub, 0.0))
    mag = np.abs(r) * np.maximum(1.0, np.maximum(np.abs(np.where(np.isfinite(lb), lb, 0.0)),
                                                 np.abs(np.where(np.isfinite(ub), ub, 0.0))))
    return float(hi.sum()), float(lo.sum()), float(mag.sum())


def build_farkas_entry(lp: LexLP, y: np.ndarray) -> FarkasEntry:
    """Turn a dual ray ``y`` (length ``m``) into a batched infeasibility test."""
    y = np.asarray(y, dtype=float).reshape(-1)
    r = lp.AbarT @ y
    scale = np.abs(r).max()
    if not np.isfinite(scale) or scale == 0.0:
        raise CertificateError("zero or non-finite Farkas vector")
    r = r / scale
    r[np.abs(r) < 1e-13] = 0.0
    static = ~lp.is_param
    mx, mn, mag = _box_extremes(r[static], lp.lb_z[static], lp.ub_z[static])
    if mx == np.inf and mn == -np.inf:
        raise CertificateError("Farkas vector unbounded over the static box")
    rP = r[lp.param_z]
    nz = np.nonzero(rP)[0]
    dev = {"r": rP[nz], "pidx": nz, "absr": np.abs(rP[nz])}
    key = np.round(r, 12).tobytes()
    return FarkasEntry(key=key, dev=dev, max_static=mx, min_static=mn, scale_static=mag)


def check_farkas(entry: FarkasEntry, Lp: np.ndarray, Up: np.ndarray, tol: Tolerances) -> np.ndarray:
    """Boolean mask of members certified infeasible by ``entry`` (host; O(B*p))."""
    d = entry.dev
    r = d["r"]
    B = Lp.shape[0]
    if r.shape[0] == 0:
        mx = np.full(B, entry.max_static)
        mn = np.full(B, entry.min_static)
        scale = np.full(B, entry.scale_static)
    else:
        L = Lp[:, d["pidx"]]
        U = Up[:, d["pidx"]]
        pos = r > 0
        with np.errstate(invalid="ignore"):
            mx = entry.max_static + np.sum(r * np.where(pos, U, L), axis=1)
            mn = entry.min_static + np.sum(r * np.where(pos, L, U), axis=1)
        fin = lambda a: np.where(np.isfinite(a), np.abs(a), 0.0)  # noqa: E731
        scale = entry.scale_static + np.sum(d["absr"] * np.maximum(1.0, np.maximum(fin(L), fin(U))), axis=1)
    thr = tol.farkas * scale
    return (mx < -thr) | (mn > thr)


# ---------------------------------------------------------------------------
# point certificate: verify one member's own basic solution (no affine law)
# ---------------------------------------------------------------------------


def certify_point(lp: LexLP, basic: np.ndarray, status: np.ndarray, lb_z: np.ndarray, ub_z: np.ndarray,
                  tol: Tolerances):
    """Certify the basic solution defined by ``(basic, status)`` for one member.

    Same conditions as Theorem 1 (lexicographic dual feasibility + primal
    feasibility, uniqueness when every free nonbasic has a nonzero lexicographic
    reduced cost), evaluated for a single bound vector without building the
    affine law.  Returns ``(ok, unique, z, stage_objectives)``.
    """
    m, N = lp.m, lp.N
    basic = np.asarray(basic, dtype=np.int64)
    if basic.size != m or np.unique(basic).size != m:
        return False, False, None, None
    st = np.asarray(status, dtype=np.int8).copy()
    st[basic] = BASIC
    nonbasic = st != BASIC
    try:
        lu = spla.splu(lp.Abar[:, basic].tocsc(), permc_spec="COLAMD")
    except RuntimeError:
        return False, False, None, None
    Cb = lp.C_z[:, basic]
    Y = lu.solve(np.ascontiguousarray(Cb.T), trans="T")
    if Y.ndim == 1:
        Y = Y[:, None]
    D = lp.C_z - (lp.AbarT @ Y).T
    D[:, basic] = 0.0
    s = lex_signs(D, lp.objective_scales(), tol.dual)
    fixed = lb_z == ub_z
    mism = nonbasic & (((st == AT_LOWER) & (s > 0)) | ((st == AT_UPPER) & (s < 0)) | ((st == AT_ZERO) & (s != 0)))
    st[mism & (s > 0) & np.isfinite(ub_z)] = AT_UPPER
    st[mism & (s < 0) & np.isfinite(lb_z)] = AT_LOWER
    mism = nonbasic & (((st == AT_LOWER) & (s > 0)) | ((st == AT_UPPER) & (s < 0)) | ((st == AT_ZERO) & (s != 0)))
    if np.any(mism & ~fixed):
        return False, False, None, None
    z = np.zeros(N)
    nb = np.nonzero(nonbasic)[0]
    zn = np.where(st[nb] == AT_LOWER, lb_z[nb], np.where(st[nb] == AT_UPPER, ub_z[nb], 0.0))
    if not np.all(np.isfinite(zn)):
        return False, False, None, None
    zero = st[nb] == AT_ZERO
    if np.any(zero & ((lb_z[nb] > 0) | (ub_z[nb] < 0))):
        return False, False, None, None
    z[nb] = zn
    z[basic] = -lu.solve(lp.Abar[:, nb] @ zn)
    lo, hi = lb_z[basic], ub_z[basic]
    zb = z[basic]
    with np.errstate(invalid="ignore"):
        ok = bool(np.all(lo - zb <= tol.primal_abs + tol.primal_rel * np.abs(lo)) and
                  np.all(zb - hi <= tol.primal_abs + tol.primal_rel * np.abs(hi)))
    unique = bool(np.all((s[nb] != 0) | fixed[nb]))
    return ok, unique, z, lp.C_z @ z
