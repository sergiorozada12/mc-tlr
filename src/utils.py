import numpy as np
import torch
import tensorly as tl
from tqdm import tqdm
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import khatri_rao
from scipy.sparse.linalg import svds
from scipy.special import kl_div, rel_entr

tl.set_backend('pytorch')

# ----- Reshape utilities -----
def ten2mat(Xten, I):
    return Xten.reshape(I, I)

def mat2ten(Xmat, Is):
    return Xmat.reshape(tuple(Is.repeat(2)))

def chain_mat2ten(X, Is):
    result = []
    for x in X:
        result.append(torch.tensor(np.unravel_index(x, tuple(Is))))
    return result

def chain_ten2mat(X, Is):
    result = []
    for x in X:
        idx = int(np.ravel_multi_index(tuple(x.tolist()), tuple(Is.tolist())))
        result.append(idx)
    return result

# ----- Normalization utilities -----
def normalize_rows(P, eps=1e-8):
    row_sums = P.sum(dim=1, keepdim=True)
    mask = (row_sums <= eps).squeeze()  # shape [I]
    P = P / row_sums.clamp_min(eps)
    P[mask] = 1. / P.shape[0]
    return P

def normalize_tensor_rows(P, D, eps=1e-8):
    marginal = P.sum(dim=tuple(range(D, 2*D)), keepdim=True)
    mask = (marginal <= eps)
    P = P / marginal
    P[mask] = 1. / P.numel()
    return P

def laplace_smoothing(P, N, eps):
    return (P * N + eps) / (N + P.numel() * eps)

# ----- Matrix/vector conversions -----
def lowtri2mat(a):
    assert np.isclose(((2*len(a)+.25)**.5+.5)%1, 0.), "Invalid input vector. Expecting a vector of length N(N-1)/2 corresponding to some square N x N matrix."
    N = int((2*len(a)+.25)**.5+.5)
    A = np.zeros((N,N)) if not isinstance(a, torch.Tensor) else torch.zeros((N,N))
    low_tri_indices = np.triu_indices(N, 1)
    A[low_tri_indices[1], low_tri_indices[0]] = a
    A = A + A.T
    return A

def mat2lowtri(A, N=None):
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Invalid input matrix. Expecting a square matrix."
    low_tri_indices = np.triu_indices(A.shape[0], 1)
    return A[low_tri_indices[1], low_tri_indices[0]]

# ----- Distance and error metrics -----
def kld_err(P, Q):
    p = P.flatten()
    q = Q.flatten()
    return kl_div(p, q).sum()

def frob_err(Ph, P):
    return torch.norm(Ph - P, 'fro').item()

def normfrob_err(Ph, P):
    norm_P = torch.norm(P, 'fro')
    return (torch.norm(Ph - P, 'fro') / (norm_P + int(P.abs().max() == 0))).item()

def l1_err(Ph, P):
    return torch.norm(Ph - P, 1).item()

def norml1_err(Ph, P):
    norm_P = torch.norm(P, 1)
    return (torch.norm(Ph - P, 1) / (norm_P + int(P.abs().max() == 0))).item()

# ----- Projection operators -----
def proj_simplex(y):
    N = len(y)
    x = np.zeros(N)
    y_til = np.maximum(y, 0)
    if np.isclose(y_til.sum(), 1.):
        x = y_til.copy()
        assert np.isclose(x.sum(), 1.)
    else:
        u_til = np.flip(np.sort(y_til))
        sum_diff = (np.cumsum(u_til) - 1) / np.arange(1, N+1)
        K = np.max(np.where(sum_diff < u_til)[0]) + 1
        tau = (np.sum(u_til[:K]) - 1) / K
        x = np.maximum(y_til - tau, 0)
        assert np.isclose(x.sum(), 1)
    return x

def proj_bounded_simplex(z, s):
    N = len(z)
    x = np.zeros(N)
    assert (s >= 0 and s <= N)
    if s == 0:
        e = .5 * np.sum((x - z)**2)
        return x, e
    elif s == N:
        x = np.ones(N)
        e = .5 * np.sum((x - z)**2)
        return x, e
    idx = np.argsort(z)
    y = z[idx]
    if np.isclose(s, round(s)):
        b = N - int(round(s))
        if y[b] - y[b-1] >= 1:
            x[idx[b:]] = 1
            e = .5 * np.sum((x - z)**2)
    T = np.cumsum(y)
    y = np.concatenate((y, [np.inf]))
    for b in range(1, N+1):
        gamma = (s + b - N - T[b-1]) / b
        if (y[0] + gamma > 0) and (y[b-1] + gamma < 1) and (y[b] + gamma >= 1):
            xtmp = np.concatenate((y[:b] + gamma, np.ones(N - b)))
            x[idx] = xtmp
            e = .5 * np.sum((x - z)**2)
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            gamma = (s + b - N + T[a-1] - T[b-1]) / (b - a)
            if (y[a-1] + gamma <= 0) and (y[a] + gamma > 0) and (y[b-1] + gamma < 1) and (y[b] + gamma >= 1):
                xtmp = np.concatenate((np.zeros(a), y[a:b] + gamma, np.ones(N - b)))
                x[idx] = xtmp
                e = .5 * np.sum((x - z)**2)
    return x

# ----- Misc -----
def erank(A):
    assert A.ndim == 2
    svs = torch.linalg.svdvals(A)
    p = svs / (torch.linalg.norm(svs, 1) + int(svs.abs().max() == 0))
    return torch.exp(-torch.sum(p * torch.log(p)))

def soft_thresh(x, l):
    return torch.maximum(torch.abs(x) - l, torch.zeros_like(x)) * x.sign()

def Qmat2Pmat(Q):
    Q = Q/torch.linalg.norm(Q,1)
    R = .5 * (Q.sum(0) + Q.sum(1))
    P = torch.diag(1/(R + (R==0).to(torch.float)))@Q
    Marginal = P.sum(1,keepdim=True)
    Mask = (Marginal==0).expand_as(P)
    P = P / Marginal
    P[Mask] = 1/len(Marginal)
    return P

def Qten2Pten(Q):
    D = Q.ndim//2
    assert Q.ndim%2==0 and Q.size()[:D]==Q.size()[D:], \
           "Invalid transition probability tensor. Expecting an even number of dimensions."
    Is = torch.tensor(Q.shape[:D])
    I = Is.prod().item()
    return mat2ten( Qmat2Pmat( ten2mat(Q,I) ), Is )

def Pmat2Qmat(P):
    return MarkovChainMatrix(P).Q

def Pten2Qten(P):
    return MarkovChainTensor(P).Q

def check_valid_transition_mat(P):
    assert P.ndim==2 and P.shape[0]==P.shape[1] and (P>=0).all() and (P<=1).all() and torch.allclose(P.sum(1),torch.ones(P.shape[0])), \
        "Invalid transition probability matrix. Expecting a right-stochastic square matrix with entries in [0,1]."

def check_valid_joint_mat(Q):
    assert Q.ndim==2 and Q.shape[0]==Q.shape[1] and (Q>=0).all() and (Q<=1).all() and torch.isclose(Q.sum(),torch.ones_like(Q.sum())), \
        "Invalid joint probability matrix. Expecting a square matrix with entries in [0,1] that sum to 1."

def check_valid_transition_ten(P):
    D = P.ndim//2
    assert P.ndim%2==0 and P.size()[:D]==P.size()[D:] and (P>=0).all() and (P<=1).all(), \
           "Invalid transition probability tensor. Expecting right-stochastic tensor with an even number of dimensions with entries in [0,1]."
    Is = torch.tensor(P.shape[:D])
    I = Is.prod().item()
    check_valid_transition_mat(ten2mat(P,I))

def check_valid_joint_ten(Q):
    D = Q.ndim//2
    assert Q.ndim%2==0 and Q.size()[:D]==Q.size()[D:] and (Q>=0).all() and (Q<=1).all() and torch.isclose(Q.sum(),torch.ones_like(Q.sum())), \
           "Invalid joint probability tensor. Expecting tensor with an even number of dimensions with entries in [0,1] that sum to 1."

def check_valid_chain_mat(X,I=None):
    assert all(list(map(np.isscalar,X))) or all( list(map( lambda x:x.numel()==1, X )) ), "Inconsistent state dimensions across chain."
    if I is not None:
        assert (torch.FloatTensor(X)<I).all() and (torch.FloatTensor(X)>=0).all(), \
            "Inconsistent state dimensions between tensor dimensions and chain."

def check_valid_chain_ten(X,Is=None):
    assert all(len(X[0])==np.array(list(map(len,X)))), "Inconsistent state dimensions across chain."
    if Is is not None:
        assert (torch.stack(X) < Is[None]).flatten().all() and (torch.stack(X) >= 0).flatten().all(), \
            "Inconsistent state dimensions between tensor dimensions and chain."
