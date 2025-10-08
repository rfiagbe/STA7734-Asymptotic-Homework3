# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 15:42:58 2025

@author: fiagb
"""

#==============================================================================
#================== IMPORTING THE NECESSARY LIBRARIES =========================
#==============================================================================

import numpy as np
from numpy.linalg import svd, norm
import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy.linalg import eigh


## Set working directory
directory = '/Users/fiagb/OneDrive/Desktop/PhD LIBRARY/FALL 2025/Asymptotic/HW3'
os.chdir(directory)
print(os.getcwd())

#==============================================================================
#=============================== PROBLEM 1  ===================================
#==============================================================================
# Part a
#SVT proximal operator + KKT check
def svt(M, tau):
    U, s, VT = svd(M, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ VT

def prox_kkt_residual(M, tau, X=None):
    if X is None:
        X = svt(M, tau)
    U, s, VT = svd(X, full_matrices=False)
    r = np.sum(s > 1e-12)
    Ur, Vr = U[:, :r], VT[:r, :].T
    G = Ur @ Vr.T
    R = (X - M) + tau * G
    return norm(R, 'fro')

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M = rng.normal(size=(30, 40))
    tau = 0.5
    X = svt(M, tau)
    print("[prox/SVT check] KKT residual Fro norm:", prox_kkt_residual(M, tau, X))

def mat_inner(A, B):
    return np.tensordot(A, B, axes=([0,1],[0,1]))

def grad_L(C, Xs, y):
    n = len(y)
    r = np.tensordot(Xs, C, axes=([1,2],[0,1])) 
    coeff = (2.0/n) * (r - y)                    
    G = np.tensordot(coeff, Xs, axes=(0,0))      
    return G

def ista_nucnorm(Xs, y, lam, C0=None, step=None, maxit=1000, tol=1e-6, verbose=False):
    n, p1, p2 = Xs.shape
    if C0 is None:
        C = np.zeros((p1, p2))
    else:
        C = C0.copy()

    def A_map(Delta):
        coeff = (2.0/n) * np.tensordot(Xs, Delta, axes=([1,2],[0,1]))
        return np.tensordot(coeff, Xs, axes=(0,0))

    if step is None:
        Z = rng.normal(size=(p1, p2))
        for _ in range(10):
            Z = A_map(Z)
            z = norm(Z, 'fro')
            if z == 0:
                break
            Z /= z
        L = mat_inner(Z, A_map(Z))
        step = 1.0 / max(L, 1e-12)
    if verbose:
        print(f"[ISTA] step size = {step:.3e}")

    for k in range(maxit):
        G = grad_L(C, Xs, y)
        C_next = svt(C - step*G, step*lam)
        if norm(C_next - C, 'fro') <= tol * max(1.0, norm(C, 'fro')):
            C = C_next
            break
        C = C_next
    return C

def make_lowrank(p1, p2, r, scale=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    U, _ = np.linalg.qr(rng.normal(size=(p1, r)))
    V, _ = np.linalg.qr(rng.normal(size=(p2, r)))
    s = np.linspace(scale, scale, r)  
    return (U * s) @ V.T, U, V

def sample_design(n, p1, p2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    Xs = rng.normal(size=(n, p1, p2))
    return Xs

def forward(Xs, Cstar, noise_sd, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = Xs.shape[0]
    y_clean = np.tensordot(Xs, Cstar, axes=([1,2],[0,1]))
    eps = rng.normal(scale=noise_sd, size=n)
    y = y_clean + eps
    return y, eps

def op_norm_of_noise_term(Xs, eps):
    n = len(eps)
    G = np.tensordot(eps / n, Xs, axes=(0,0))
    s = svd(G, full_matrices=False, compute_uv=False)
    return s[0], G

if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # problem sizes
    p1, p2, rstar = 40, 50, 3
    n = 600

    Cstar, Ustar, Vstar = make_lowrank(p1, p2, rstar, scale=1.0, rng=rng)

    Xs = sample_design(n, p1, p2, rng=rng)
    y, eps = forward(Xs, Cstar, noise_sd=0.5, rng=rng)

    opG, G = op_norm_of_noise_term(Xs, eps)
    lam = 2.1 * opG  

    Chat = ista_nucnorm(Xs, y, lam, maxit=2000, tol=1e-6, verbose=True)

    errF = norm(Chat - Cstar, 'fro')

    def A_quad(Delta):
        r = np.tensordot(Xs, Delta, axes=([1,2],[0,1]))
        return np.mean(r**2)
    kappa_est = 0.5 * min(A_quad(rng.normal(size=(p1,p2))) / (norm(rng.normal(size=(p1,p2)),'fro')**2) for _ in range(50))
    bound = (3.0 * lam / max(kappa_est, 1e-8)) * np.sqrt(2.0 * rstar)

    print(f"||Chat - C*||_F = {errF: .4f}")
    print(f"dual term ||(1/n)Σ eps X||_op = {opG: .4f}, lambda* used = {lam: .4f}")
    print(f"kappa (heuristic) = {kappa_est: .4f}")
    print(f"theoretical bound = {bound: .4f}")


#==============================================================================
#=============================== PROBLEM 2 ====================================
#==============================================================================
# Part a

def svt(M, tau):
    U, s, VT = svd(M, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ VT

def tv1d_denoise(y, lam):
    n = len(y)
    if n == 0 or lam <= 0:
        return y.copy()
    x = np.empty(n, dtype=float)
    k = k0 = 0
    vmin = y[0] - lam
    vmax = y[0] + lam
    umin = lam
    umax = -lam
    for i in range(1, n):
        di = y[i] - y[i-1]
        umin += di
        umax += di
        if umin > lam:
            while k <= k0:
                x[k] = vmin; k += 1
            vmin = y[i-1] - lam; vmax = y[i-1] + lam
            umin = lam + di; umax = -lam + di; k0 = i-1
        elif umax < -lam:
            while k <= k0:
                x[k] = vmax; k += 1
            vmin = y[i-1] - lam; vmax = y[i-1] + lam
            umin = lam + di; umax = -lam + di; k0 = i-1
        vmin = min(vmin, y[i] - lam - umin)
        vmax = max(vmax, y[i] + lam - umax)
    while k <= k0:
        x[k] = vmin; k += 1
    vmin = 0.5 * (vmin + vmax)
    for i in range(k0+1, n):
        x[i] = vmin
    return x

def prox_tv_rows(B, lam):
    return np.vstack([tv1d_denoise(B[i], lam) for i in range(B.shape[0])])

def fit_lr_frechet(Q, Xc, lam, lam_fuse, maxit=350, tol=1e-6, step=None, random_state=0):
    rng = np.random.default_rng(random_state)
    n, M = Q.shape
    q = Xc.shape[1]
    alpha = Q.mean(axis=0)
    B = np.zeros((q, M))

    if step is None:
        z = rng.normal(size=(q,))
        for _ in range(8):
            z = Xc.T @ (Xc @ z)
            nz = norm(z); z = z / (nz + 1e-12)
        Lop = float(z @ (Xc.T @ (Xc @ z)))
        L = (2.0 / n) * Lop
        step = 1.0 / max(L, 1e-10)

    obj_prev = np.inf
    for it in range(maxit):
        alpha = (Q - Xc @ B).mean(axis=0)
        R = (Xc @ B + alpha) - Q
        G = (2.0 / n) * (Xc.T @ R)
        V = B - step * G
        B1 = svt(V, step * lam)
        Bn = prox_tv_rows(B1, step * lam_fuse)

        Rn = (Xc @ Bn + alpha) - Q
        nuc = svd(Bn, full_matrices=False, compute_uv=False).sum()
        BD = np.diff(Bn, axis=1)
        obj = (Rn**2).mean() + lam * nuc + lam_fuse * np.abs(BD).sum()

        if abs(obj - obj_prev) <= tol * (1.0 + obj_prev):
            B = Bn; break
        B = Bn; obj_prev = obj

    alpha = (Q - Xc @ B).mean(axis=0)
    svals = svd(B, full_matrices=False, compute_uv=False)
    rankB = int((svals > 1e-6).sum())
    return B, alpha, rankB, obj, it+1

def make_lowrank_B(q, M, r, scale=1.0, seed=1):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(q, r)))
    V, _ = np.linalg.qr(rng.normal(size=(M, r)))
    s = np.linspace(scale, scale, r)
    return (U * s) @ V.T

def simulate_frechet(n=220, q=18, M=32, r=3, noise_sd=0.35, seed=0):
    rng = np.random.default_rng(seed)
    B_true = make_lowrank_B(q, M, r, scale=1.0, seed=seed)
    Xc = rng.normal(size=(n, q))
    alpha_true = rng.normal(scale=0.2, size=M)
    Q_mean = Xc @ B_true + alpha_true
    Q = Q_mean + rng.normal(scale=noise_sd, size=Q_mean.shape)
    return Q, Xc, B_true, alpha_true

def lam_max_ls(Q, Xc):
    n, M = Q.shape
    Qbar = Q.mean(axis=0)
    ones = np.ones((n,1))
    G0 = (2.0/n) * Xc.T @ (ones @ Qbar[None,:] - Q)
    smax = svd(G0, compute_uv=False, full_matrices=False)[0]
    return float(smax), G0

def lambda_fuse_anchor(G0):
    BD = np.diff(G0, axis=1)
    return float(np.max(np.abs(BD)))

# Simulation
Q, Xc, B_true, alpha_true = simulate_frechet()

lam_max, G0 = lam_max_ls(Q, Xc)
Lf_anchor = lambda_fuse_anchor(G0)
lam_grid = [lam_max*(0.5**k) for k in range(6)]
lf_grid  = [0.0, 0.5*Lf_anchor, 1.0*Lf_anchor, 2.0*Lf_anchor]

rows = []
for lam in lam_grid:
    for lf in lf_grid:
        Bhat, ahat, r_eff, obj, iters = fit_lr_frechet(Q, Xc, lam, lf, maxit=350, tol=1e-6, random_state=42)
        errF = norm(Bhat - B_true, 'fro')
        rows.append({"lambda": lam, "lambda_fuse": lf, "rank(Bhat)": r_eff, "||Bhat - B*||_F": errF, "iters": iters})

df = pd.DataFrame(rows).sort_values(["lambda","lambda_fuse"]).reset_index(drop=True)
df_round = df.copy()
for c in ["lambda","lambda_fuse","||Bhat - B*||_F"]:
    df_round[c] = df_round[c].round(4)

print(df_round)

# Heatmaps
lam_vals = sorted(df["lambda"].unique())
lf_vals = sorted(df["lambda_fuse"].unique())
err_grid = df.pivot(index="lambda", columns="lambda_fuse", values="||Bhat - B*||_F").loc[lam_vals, lf_vals].values
rank_grid = df.pivot(index="lambda", columns="lambda_fuse", values="rank(Bhat)").loc[lam_vals, lf_vals].values

plt.figure()
plt.imshow(err_grid, aspect="auto")
plt.colorbar(label="‖B̂ − B*‖_F")
plt.xticks(range(len(lf_vals)), [round(v,4) for v in lf_vals])
plt.yticks(range(len(lam_vals)), [round(v,4) for v in lam_vals])
plt.xlabel("lambda_fuse"); plt.ylabel("lambda")
plt.title("Error heatmap: ‖B̂ − B*‖_F")
plt.tight_layout(); plt.show()

plt.figure()
plt.imshow(rank_grid, aspect="auto")
plt.colorbar(label="rank(B̂)")
plt.xticks(range(len(lf_vals)), [round(v,4) for v in lf_vals])
plt.yticks(range(len(lam_vals)), [round(v,4) for v in lam_vals])
plt.xlabel("lambda_fuse"); plt.ylabel("lambda")
plt.title("Effective rank heatmap")
plt.tight_layout(); plt.show()



#Part b
rng = np.random.default_rng(21)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def svt(M, tau):
    U, s, VT = svd(M, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ VT

def fit_nuc_logit(Xs, y, lam=0.2, maxit=300, tol=1e-5):
    n = len(Xs); q, m = Xs[0].shape
    C = np.zeros((q, m))
    L = (0.25/n) * sum(np.linalg.norm(Xi, "fro")**2 for Xi in Xs)
    eta = 1.0/max(L, 1e-12)
    obj_prev = np.inf
    for _ in range(maxit):
        grad = np.zeros_like(C); loss = 0.0
        for Xi, yi in zip(Xs, y):
            z = float(np.tensordot(Xi, C, axes=2)); p = sigmoid(z)
            grad += (p - yi) * Xi
            loss += -yi*np.log(p+1e-12) - (1-yi)*np.log(1-p+1e-12)
        grad /= n; loss /= n
        Cn = svt(C - eta*grad, eta*lam)
        if abs(loss - obj_prev) <= tol*(1+obj_prev):
            C = Cn; break
        C, obj_prev = Cn, loss
    s = svd(C, compute_uv=False, full_matrices=False)
    r_eff = int(np.sum(s > 1e-6))
    return C, r_eff

def auc_score(y_true, y_score):
    order = np.argsort(y_score)
    y = np.array(y_true)[order]
    n0 = (y==0).sum(); n1 = (y==1).sum()
    if n0==0 or n1==0: return np.nan
    rank_sum = np.sum(np.where(y==1)[0] + 1)
    return float((rank_sum - n1*(n1+1)/2) / (n0*n1))

def make_shift(X, dr=2, dc=4):
    return np.roll(np.roll(X, dr, axis=0), dc, axis=1)

def block_diag_concat(mats):
    q, m = mats[0].shape
    k = len(mats)
    out = np.zeros((q*k, m*k))
    for i, Mi in enumerate(mats):
        out[i*q:(i+1)*q, i*m:(i+1)*m] = Mi
    return out

n, q, m, r_true = 1000, 20, 60, 2
U,_ = np.linalg.qr(rng.normal(size=(q, r_true)))
V,_ = np.linalg.qr(rng.normal(size=(m, r_true)))
s = np.array([2.0, 1.5])
C0 = (U * s) @ V.T

Xs = [rng.normal(size=(q, m)) for _ in range(n)]
def score_true(X):
    Xsft = make_shift(X, dr=2, dc=4)
    return float(np.tensordot(X, C0, axes=2) + np.tensordot(Xsft, C0, axes=2))

etas = np.array([score_true(Xi) for Xi in Xs])
p = sigmoid(1.4 * etas)
y = rng.binomial(1, p)

# Train/test split
idx = rng.permutation(n); tr = idx[:750]; te = idx[750:]
Xs_tr = [Xs[i] for i in tr]; y_tr = y[tr]
Xs_te = [Xs[i] for i in te]; y_te = y[te]

def augment_pair(X):
    Xsft = make_shift(X, dr=2, dc=4)
    return block_diag_concat([X, Xsft])

lams = [0.05, 0.08, 0.12, 0.18, 0.26]
rows = []

for lam in lams:
    C1, r1 = fit_nuc_logit(Xs_tr, y_tr, lam=lam, maxit=350, tol=1e-5)
    scores1 = np.array([np.tensordot(Xi, C1, axes=2) for Xi in Xs_te])
    auc1 = auc_score(y_te, scores1)
    rows.append({"setting":"original", "lambda":lam, "AUC":round(auc1,3), "rank(Ĉ)":r1})
    Xs_tr_aug = [augment_pair(X) for X in Xs_tr]
    Xs_te_aug = [augment_pair(X) for X in Xs_te]
    C2, r2 = fit_nuc_logit(Xs_tr_aug, y_tr, lam=lam, maxit=350, tol=1e-5)
    scores2 = np.array([np.tensordot(Xi, C2, axes=2) for Xi in Xs_te_aug])
    auc2 = auc_score(y_te, scores2)
    rows.append({"setting":"original+shift", "lambda":lam, "AUC":round(auc2,3), "rank(Ĉ)":r2})

results = pd.DataFrame(rows)
results

# Plots
plt.figure()
for setting in ["original","original+shift"]:
    sub = results[results["setting"]==setting]
    plt.plot(sub["lambda"], sub["AUC"], marker="o", label=setting)
plt.xlabel("λ"); plt.ylabel("AUC"); plt.title("ROC AUC vs λ (original vs +shift)")
plt.legend(); plt.ylim(0.5,1.0); plt.tight_layout(); plt.show()

plt.figure()
for setting in ["original","original+shift"]:
    sub = results[results["setting"]==setting]
    plt.plot(sub["lambda"], sub["rank(Ĉ)"], marker="o", label=setting)
plt.xlabel("λ"); plt.ylabel("effective rank of Ĉ"); plt.title("Rank Path vs λ")
plt.legend(); plt.tight_layout(); plt.show()



#==============================================================================
#=============================== PROBLEM 3 ====================================
#==============================================================================
# Part a
rng = np.random.default_rng(10)

def swiss_roll(n, rng):
    theta = rng.uniform(1.5*np.pi, 4.5*np.pi, size=n)
    h = rng.uniform(0.0, 10.0, size=n)
    X = np.stack([theta*np.cos(theta), h, theta*np.sin(theta)], axis=1)
    return X, theta, h

def build_P_alpha1(X, eps):
    XX = np.sum(X**2, axis=1, keepdims=True)
    d2 = XX + XX.T - 2*X@X.T
    K = np.exp(-d2/eps)
    d = np.sum(K, axis=1, keepdims=True)
    Ktil = K / (d @ d.T)
    Dt = np.sum(Ktil, axis=1, keepdims=True)
    P = Ktil / Dt
    return P

def generator_estimate(P, f, eps):
    return (P @ f - f) / eps

def lb_proxy_theta(theta, f):
    idx = np.argsort(theta)
    th = theta[idx]; fs = f[idx]
    dth = np.gradient(th)
    fp  = np.gradient(fs, dth)
    fpp = np.gradient(fp,  dth)
    g   = 1.0 + th**2
    Delta = fpp/g - (th*fp)/(g**2)
    out = np.empty_like(Delta); out[idx] = Delta
    return out

n = 2500
X, theta, h = swiss_roll(n, rng)
f = np.sin(theta)

eps_list = [0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 3.0]
corrs, slopes, rmse = [], [], []

for eps in eps_list:
    P = build_P_alpha1(X, eps)
    Lhat = generator_estimate(P, f, eps)
    LB   = lb_proxy_theta(theta, f)
    c = float(np.corrcoef(Lhat, LB)[0,1])
    corrs.append(c)
    c_eta = float(np.dot(LB, Lhat) / (np.dot(LB, LB) + 1e-12))
    slopes.append(c_eta)
    rmse.append(float(np.sqrt(np.mean((Lhat - c_eta*LB)**2))))

df = pd.DataFrame({"eps": eps_list, "corr": corrs, "rmse": rmse})
df.round(4)

# Plots
plt.figure()
plt.plot(eps_list, corrs, marker="o")
plt.xlabel("ε"); plt.ylabel("Correlation"); plt.title("corr( (P−I)f/ε , Δ_g f ) vs ε")
plt.ylim(-1, 1.0); plt.grid(True, linewidth=0.3); plt.tight_layout(); plt.show()

eps0 = 1.6
P0 = build_P_alpha1(X, eps0)
Lhat0 = generator_estimate(P0, f, eps0)
LB0 = lb_proxy_theta(theta, f)
c_eta0 = float(np.dot(LB0, Lhat0) / (np.dot(LB0, LB0) + 1e-12))

plt.figure()
plt.scatter(LB0, Lhat0, s=8, alpha=0.6)
xline = np.linspace(LB0.min(), LB0.max(), 200)
plt.plot(xline, c_eta0*xline)
plt.xlabel("Δ_g f (proxy)"); plt.ylabel("(P−I)f / ε")
plt.title(f"Alignment at ε={eps0}  (ĉ={c_eta0:.3f}, corr={np.corrcoef(LB0,Lhat0)[0,1]:.3f})")
plt.tight_layout(); plt.show()


# Part b
def swiss_roll(n, rng):
    th = rng.uniform(1.5*np.pi, 4.5*np.pi, size=n)
    h  = rng.uniform(0.0, 10.0, size=n)
    X  = np.stack([th*np.cos(th), h, th*np.sin(th)], axis=1)
    return X, th, h

def build_P_alpha1(X, eps):
    XX = np.sum(X**2, axis=1, keepdims=True)
    d2 = XX + XX.T - 2*X@X.T
    K  = np.exp(-d2/eps)
    d  = np.sum(K, axis=1, keepdims=True)
    Kt = K / (d @ d.T)
    Dt = np.sum(Kt, axis=1, keepdims=True)
    P  = Kt / Dt
    return P

def smallest_eigs_IminusP(P, k=5):
    A = 0.5*((np.eye(P.shape[0]) - P) + (np.eye(P.shape[0]) - P).T)
    w = eigh(A)[0]
    w = np.sort(np.real(w))
    return w[:k]

rng = np.random.default_rng(321)

n = 1000
X, theta, h = swiss_roll(n, rng)
eps_list = [0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 3.0] #[0.9, 1.2, 1.6, 2.0, 2.6]
k = 5
vals_vs_eps = []
for eps in eps_list:
    P = build_P_alpha1(X, eps)
    vals_vs_eps.append(smallest_eigs_IminusP(P, k=k))

vals_vs_eps

plt.figure()
for j in range(k):
    plt.plot(eps_list, [vals_vs_eps[i][j] for i in range(len(eps_list))], marker="o", label=f"eig {j}")
plt.xlabel("ε"); plt.ylabel("eigs of (I − P)"); plt.title("Swiss roll (α=1): first 5 eigenvalues vs ε")
plt.legend(ncol=3, fontsize=8); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

# Rectangle
def rectangle_points(nx=12, ny=10):
    xs, ys = np.linspace(0,1,nx), np.linspace(0,1,ny)
    Xg, Yg  = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([Xg.ravel(), Yg.ravel()], axis=1)

def neumann_rect_first(kmax=5):
    vals = []
    for k in range(0,7):
        for l in range(0,7):
            vals.append(np.pi**2*(k**2 + l**2))
    vals.sort()
    return np.array(vals[:kmax])

pts = rectangle_points(12, 10)
P_rect = build_P_alpha1(pts, eps=0.1)
A_rect = 0.5*((np.eye(P_rect.shape[0]) - P_rect) + (np.eye(P_rect.shape[0]) - P_rect).T)
w_rect = np.sort(np.real(eigh(A_rect)[0]))[:k]

w_neu  = neumann_rect_first(k)
s = float(np.dot(w_neu, w_rect) / np.dot(w_neu, w_neu))
w_neu_scaled = s*w_neu

plt.figure()
plt.plot(range(k), w_rect, marker="o", label="graph (I − P)")
plt.plot(range(k), w_neu_scaled, marker="x", label=f"Neumann × {s:.2f}")
plt.xlabel("index"); plt.ylabel("eigenvalue"); plt.title("Rectangle: first 5 eigenvalues")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()


# Part c
rng = np.random.default_rng(35)

def swiss_roll(n, rng):
    th = rng.uniform(1.5*np.pi, 4.5*np.pi, size=n)
    h  = rng.uniform(0.0, 10.0, size=n)
    X  = np.stack([th*np.cos(th), h, th*np.sin(th)], axis=1)
    return X, th, h

def build_P_alpha1(X, eps):
    XX = np.sum(X**2, axis=1, keepdims=True)
    d2 = XX + XX.T - 2*X@X.T
    K  = np.exp(-d2/eps)
    d  = np.sum(K, axis=1, keepdims=True)
    Kt = K / (d @ d.T)
    Dt = np.sum(Kt, axis=1, keepdims=True)
    P  = Kt / Dt
    return P

n = 140
X, theta, h = swiss_roll(n, rng)
eps = 2.0
P = build_P_alpha1(X, eps)

w, V = eig(P)
w = np.real(w); V = np.real(V)
idx = np.argsort(-w)
lam = w[idx][1:]
psi = V[:, idx][:, 1:]

# random pairs
pairs = rng.integers(low=0, high=n, size=(240,2))

def gaussian_proxy_theta(theta, t, i, j):
    dth = theta[i] - theta[j]
    return float(np.exp(- (dth**2) / (4.0*t)))

for t in [0.5, 1.0, 2.0]:
    weights = lam**(2*t)
    diffs = psi[pairs[:,0], :] - psi[pairs[:,1], :]
    Dt2 = np.sum((diffs**2) * weights[None,:], axis=1)
    G = np.array([gaussian_proxy_theta(theta, t, i, j) for i,j in pairs])

    plt.figure()
    plt.scatter(G, Dt2, s=12, alpha=0.55)
    plt.xlabel(r"Gaussian proxy  $e^{-(\Delta\theta)^2/(4t)}$")
    plt.ylabel(r"$D_t^2(x_i,x_j)$")
    plt.title(f"Swiss roll (α=1): diffusion distance vs proxy, t={t}")
    plt.grid(alpha=.3); plt.tight_layout(); plt.show()

























