import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# -------------------------------------------
# 핵심 아이디어
# - 각 변환을 적용 → 표준화 → 분포 품질 점수 산출
# - 점수 = |skew| + 0.1*|excess_kurtosis| + KS(normal)  (낮을수록 좋음)
# - 컬럼마다 상위 추천 변환을 반환
# -------------------------------------------

def _make_apply_identity():
    def f(a): 
        a = np.asarray(a, np.float64)
        out = a.copy()
        return out
    return f

def _make_apply_sqrt(shift=None):
    def f(a):
        a = np.asarray(a, np.float64)
        if shift is not None: a = a + shift
        out = np.empty_like(a); m = np.isfinite(a) & (a >= 0)
        out[:] = np.nan; out[m] = np.sqrt(a[m])
        return out
    return f

def _make_apply_cbrt(shift=None):
    def f(a):
        a = np.asarray(a, np.float64)
        if shift is not None: a = a + shift
        out = np.empty_like(a); m = np.isfinite(a)
        out[:] = np.nan; out[m] = np.cbrt(a[m])
        return out
    return f

def _make_apply_log(base='e', eps=None):
    def f(a):
        a = np.asarray(a, np.float64)
        if eps is not None: a = a + eps
        out = np.empty_like(a); m = np.isfinite(a) & (a > 0)
        out[:] = np.nan
        if base == 'e':
            out[m] = np.log(a[m])
        elif base == '10':
            out[m] = np.log10(a[m])
        elif base == '1p':  # log1p
            out[m] = np.log1p(a[m]-1.0)  # a>0 보장되면 동일
        return out
    return f

def _make_apply_arcsinh():
    def f(a):
        a = np.asarray(a, np.float64)
        out = np.empty_like(a); m = np.isfinite(a)
        out[:] = np.nan; out[m] = np.arcsinh(a[m])
        return out
    return f

def _make_apply_boxcox(lmb):
    def f(a):
        a = np.asarray(a, np.float64)
        out = np.empty_like(a); m = np.isfinite(a) & (a > 0)
        out[:] = np.nan
        if np.any(m):
            out[m] = stats.boxcox(a[m], lmbda=lmb)
        return out
    return f

def _make_apply_yeojohnson(pt):  # fitted PowerTransformer
    def f(a):
        a = np.asarray(a, np.float64)
        out = np.empty_like(a); m = np.isfinite(a)
        out[:] = np.nan
        if np.any(m):
            out[m] = pt.transform(a[m].reshape(-1,1)).ravel()
        return out
    return f

def _make_apply_exp(clip_abs=None):
    def f(a):
        a = np.asarray(a, np.float64)
        out = np.empty_like(a); m = np.isfinite(a)
        out[:] = np.nan
        if np.any(m):
            v = a[m]
            if clip_abs is not None:
                v = np.clip(v, -clip_abs, clip_abs)  # 오버플로 가드
            out[m] = np.exp(v)
        return out
    return f

def _make_apply_reciprocal(c=None):
    def f(a):
        a = np.asarray(a, np.float64)
        out = np.empty_like(a); m = np.isfinite(a)
        out[:] = np.nan
        if np.any(m):
            denom = a[m].copy()
            if c is not None:
                # |x|<c 인 구간을 ±c로 스냅, 0은 +c로
                s = np.sign(denom); s[s==0] = 1.0
                denom = np.where(np.abs(denom) < c, s*c, denom)
            out[m] = 1.0 / denom
        return out
    return f

def _safe_standardize(x):
    x = np.asarray(x, np.float64)
    m = np.nanmean(x); s = np.nanstd(x)
    return (x - m) / (s + 1e-12) if np.isfinite(s) and s > 0 else x * 0.0

def _score_gaussianity(z):
    z = z[np.isfinite(z)]
    if z.size < 20: return np.inf
    skew = stats.skew(z, bias=False, nan_policy='omit')
    kurt = stats.kurtosis(z, fisher=True, bias=False, nan_policy='omit')
    zs = (z - np.nanmean(z)) / (np.nanstd(z) + 1e-12)
    xs = np.sort(zs); ecdf = np.arange(1, xs.size+1)/xs.size
    ks = np.max(np.abs(ecdf - stats.norm.cdf(xs)))
    return abs(skew) + 0.1*abs(kurt) + ks  # 낮을수록 좋음

def _min_pos(x):
    xp = x[x > 0]
    return (np.nanpercentile(xp, 1) if xp.size else None)

# ---------- 채널 벡터 → 후보 변환 ----------
def _transform_candidates_1d(x, random_state=42):
    x = np.asarray(x, np.float64)
    xf = x[np.isfinite(x)]
    out = []
    rng = np.random.RandomState(random_state)

    # identity
    out.append(("identity", _make_apply_identity(), ""))

    # sqrt / cbrt (shift 필요 여부 고정)
    if np.nanmin(x) >= 0:
        out += [("sqrt", _make_apply_sqrt(shift=None), "nonneg"),
                ("cbrt", _make_apply_cbrt(shift=None), "nonneg")]
    else:
        shift = -np.nanmin(x) + 1e-6
        out += [(f"sqrt_shift({shift:.3g})", _make_apply_sqrt(shift=shift), "shift"),
                (f"cbrt_shift({shift:.3g})", _make_apply_cbrt(shift=shift), "shift")]

    # log류(고정 eps)
    if np.nanmin(x) > 0:
        out += [("log",   _make_apply_log('e',  eps=None), "pos"),
                ("log10", _make_apply_log('10', eps=None), "pos"),
                ("log1p", _make_apply_log('1p', eps=None), "pos")]
    else:
        eps = _min_pos(x)
        if eps is not None:
            eps = max(eps*0.1, 1e-6)
            out += [(f"log(x+{eps:.2g})",   _make_apply_log('e',  eps=eps), "shift"),
                    (f"log1p(x+{eps:.2g})", _make_apply_log('1p', eps=eps), "shift")]

    # arcsinh
    out.append(("arcsinh", _make_apply_arcsinh(), ""))

    # Box-Cox (양수만)
    if np.nanmin(x) > 0:
        try:
            lmb = stats.boxcox_normmax(xf, method='mle')
            out.append((f"boxcox(λ={lmb:.3f})", _make_apply_boxcox(lmb), "pos"))
        except Exception:
            pass

    # Yeo-Johnson (fit once)
    try:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        pt.fit(xf.reshape(-1,1))
        out.append((f"yeo_johnson(λ={pt.lambdas_[0]:.3f})", _make_apply_yeojohnson(pt), "robust"))
    except Exception:
        pass

    # exp (오버플로 가드)
    xmax = np.nanpercentile(np.abs(xf), 99) if xf.size else np.inf
    if np.isfinite(xmax) and xmax <= 5:
        out.append(("exp", _make_apply_exp(clip_abs=10.0), "careful(max<=5)"))

    # reciprocal (0 분모 방지)
    amin = np.nanmin(np.abs(xf)) if xf.size else np.inf
    if amin > 1e-8:
        out.append(("reciprocal", _make_apply_reciprocal(c=None), "nozero"))
    else:
        c = np.nanpercentile(np.abs(xf), 1) if xf.size else 1e-6
        c = max(c, 1e-6)
        out.append((f"reciprocal_stab({c:.2g})", _make_apply_reciprocal(c=c), "stab"))
    return out

def _best_transforms_for_channel(x, topk=3):
    cands = _transform_candidates_1d(x)
    scored = []
    for name, xt, note in cands:
        s = _score_gaussianity(_safe_standardize(xt))
        scored.append((name, s, note))
    scored.sort(key=lambda t: t[1])
    return scored[:topk], cands  # 상위 topk, 전체 후보(재현용)

def _best_transforms_for_channel(x, topk=3, random_state=42):
    cands = _transform_candidates_1d(x, random_state=random_state)  # (name, apply_fn, note)
    scored = []
    for name, apply_fn, note in cands:
        xt = apply_fn(x)                      # 고정 파라미터로 변환
        s = _score_gaussianity(_safe_standardize(xt))
        scored.append((name, s, note, apply_fn))
    scored.sort(key=lambda t: t[1])
    return scored[:topk], scored  # 상위 topk, 전체(재현용)

# ---------- 공개 API ----------
def suggest_transforms_ndarray(X, topk=3, sample_limit=500_000, random_state=42):
    assert X.ndim == 4, "X must be (N,W,H,C)"
    N, W, H, C = X.shape
    rng = np.random.RandomState(random_state)
    recs, appliers = [], {}

    for c in range(C):
        x_c = X[..., c].ravel()
        idx = np.flatnonzero(np.isfinite(x_c))
        if idx.size == 0: 
            continue
        if idx.size > sample_limit:
            idx = rng.choice(idx, size=sample_limit, replace=False)
        x_sample = x_c[idx]

        top, all_scored = _best_transforms_for_channel(x_sample, topk=topk, random_state=random_state)
        # 추천 기록
        for rank, (name, score, note, _af) in enumerate(top, 1):
            recs.append({"channel": c, "rank": rank, "transform": name, "score": float(np.round(score, 6)), "note": note})

        # 고정 파라미터 apply_fn 맵 구축
        appliers[c] = {name: af for (name, _, note, af) in all_scored}
    return recs, appliers

def apply_channel_transforms(X, selection, appliers, inplace=False, standardize_after=True):
    assert X.ndim == 4
    X_t = X if inplace else X.copy()
    N, W, H, C = X_t.shape

    for c, tname in selection.items():
        c = int(c)
        if c not in appliers or tname not in appliers[c]:
            raise KeyError(f"transform '{tname}' for channel {c} not found in appliers")
        flat = X_t[..., c].reshape(-1)
        x_new = appliers[c][tname](flat)  # 고정 파라미터 적용
        X_t[..., c] = x_new.reshape(N, W, H)

        if standardize_after:
            x = X_t[..., c]
            m = np.nanmean(x); s = np.nanstd(x)
            if np.isfinite(s) and s > 0:
                X_t[..., c] = (x - m) / (s + 1e-12)
    return X_t

if __name__ == "__main__":
    # 가짜 데이터 (N,W,H,C) = (20, 64, 64, 4)
    rng = np.random.RandomState(0)
    N, W, H, C = 20, 64, 64, 4
    X = np.zeros((N, W, H, C), np.float64)
    X[...,0] = np.exp(rng.randn(N,W,H)) * 10          # heavy-tail → log/yeo_johnson 유리
    X[...,1] = rng.gamma(5., 2., size=(N,W,H))        # 양수, 오른꼬리 → boxcox/log 후보
    X[...,2] = rng.normal(0, 1, size=(N,W,H))         # 이미 정상분포 → identity
    X[...,3] = rng.randn(N,W,H) * 0.5 + 5             # shift된 정규 → yj/standardize

    recs, appliers = suggest_transforms_ndarray(X, topk=3)
    # 채널별 추천 상위 1개씩 선택(예시)
    selection = {r["channel"]: r["transform"] for r in recs if r["rank"] == 1}
    X_t = apply_channel_transforms(X, selection, appliers, inplace=False, standardize_after=True)
    print("selected:", selection)
    print("X_t shape:", X_t.shape)