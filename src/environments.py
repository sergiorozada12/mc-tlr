from src.utils import *
from typing import Tuple, List

# ---------------------------------------------

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


# ---------------------------------------------
class MarkovChainMatrix:
    def __init__(self,P):
        check_valid_transition_mat(P)

        self.P = P
        self.I = P.shape[0]

        evals,evecs = torch.linalg.eig(P.T)
        idx_pi = torch.where(torch.abs(evals-1)<=1e-5)[0][0]
        self.R = torch.abs(torch.real(evecs[:,idx_pi]))
        self.R = self.R / self.R.sum()

        self.Q = torch.diag(self.R)@self.P
        self.current_state = torch.multinomial(self.R,1).item()

    def reset(self):
        self.current_state = torch.multinomial(self.R,1).item()

    def step(self):
        transition_prob = self.P[self.current_state]
        self.current_state = torch.multinomial(transition_prob,1).item()
        return self.current_state
    
    def simulate(self, num_steps:int):
        X = [self.current_state]
        for _ in range(1,num_steps):
            next_state = self.step()
            X.append(next_state)
        return X

class MarkovChainTensor:
    def __init__(self,P):
        check_valid_transition_ten(P)
        self.D = P.ndim//2
        self.Is = torch.tensor(P.shape[:self.D])
        self.I = torch.prod(self.Is).item()

        P_mat = P.reshape(self.I,self.I)
        self._mcm = MarkovChainMatrix(P_mat)
        self.P = self._mcm.P.reshape(tuple(self.Is.repeat(2)))
        self.Q = self._mcm.Q.reshape(tuple(self.Is.repeat(2)))
        self.R = self._mcm.R.reshape(tuple(self.Is))
        self.current_state = torch.tensor(np.unravel_index(self._mcm.current_state, tuple(self.Is)))

    def reset(self):
        self._mcm.reset()
        self.current_state = torch.tensor(np.unravel_index(self._mcm.current_state, tuple(self.Is)))

    def step(self):
        current_mat_state = self._mcm.step()
        self.current_state = torch.tensor(np.unravel_index(current_mat_state, tuple(self.Is)))
        # self.current_state = self._mcm.step()
        return self.current_state

    def simulate(self, num_steps:int):
        X = [self.current_state]
        for _ in range(1,num_steps):
            next_state = self.step()
            X.append(next_state)
        return X

# ---------------------------------------------

def generate_tensor(Is, K):
    D = len(Is)
    factors = []
    for d in range(D):
        factor = torch.rand(Is[d],K)
        factor = factor / factor.sum(0,keepdim=True)
        factors.append(factor)

    weights = torch.rand(K)
    weights = weights / weights.sum()

    return cp_to_tensor((weights,factors)), factors, weights

def generate_erdosrenyi_matrix_model(I, edge_prob:float=.3, eps:float=.02, beta:float=1.):
    '''
    Inputs:
    - I: Number of states
    - edge_prob: Probability of connecting two states
    - eps: Bound transition probabilities away from 0 and 1, that is, off-diagonal entries are in [eps,1-eps]
    - beta: Diagonal entries have value beta*(1-eps) + eps
    '''
    A = lowtri2mat(np.random.binomial(1,edge_prob,int(I*(I-1)/2)))
    L = np.diag(A.sum(0)) - A
    while np.sum(np.abs(np.linalg.eigvalsh(L))<1e-9)>1:
        A = lowtri2mat(np.random.binomial(1,edge_prob,int(I*(I-1)/2)))
        L = np.diag(A.sum(0)) - A
    At = A + beta * np.eye(I)
    P0 = At * (1-eps) + eps
    P = torch.tensor(P0 / P0.sum(1,keepdims=True)).to(torch.float)
    mc = MarkovChainMatrix(P)

    return mc

def generate_lowranktensor_model(Is, K:int):
    '''
    Inputs:
    - Is: Vector of state dimensions
    - K: Tensor rank
    '''
    D = len(Is)
    I = torch.prod(Is).item()

    # Zhu et al., 2022, Operations Research, "Learning Markov"
    U = [torch.randn(Is[d%D],K) for d in range(2*D)]
    U = [(U[d]*U[d])/(torch.linalg.norm(U[d],dim=0,keepdim=True)**2) for d in range(2*D)]
    # w = torch.FloatTensor(np.random.beta(.5,.5,K))
    w = torch.FloatTensor(np.random.rand(K))
    # w = (w / np.linalg.norm(w))**2
    w = w / w.sum()
    P = cp_to_tensor((w,U))
    P_mat = P.reshape(I,I)
    P_mat = P_mat / P_mat.sum(dim=1,keepdim=True)
    P = P_mat.reshape(tuple(Is.repeat(2)))
    mc = MarkovChainTensor(P)

    return mc

def generate_lowrankmatrix_model(I:int,K:int):
    '''
    Inputs:
    - I: Number of states
    - K: Matrix rank
    - 
    '''
    # Zhu et al., 2022, Operations Research, "Learning Markov"
    U0 = torch.randn(I,K)
    U0 = (U0*U0)/(torch.linalg.norm(U0,dim=1,keepdim=True)**2)
    V0 = torch.randn(I,K)
    V0 = (V0*V0)/(torch.linalg.norm(V0,dim=0,keepdim=True)**2)
    B = torch.diag(torch.FloatTensor(np.random.beta(.5,.5,I)))
    P0 = (U0@V0.T)@B
    P_1D = P0 / P0.sum(dim=1,keepdim=True)
    mc = MarkovChainMatrix(P_1D)

    return mc

def generate_blocktensor_model(I_per_block:int, D:int, K:int):
    '''
    Inputs:
    - I_per_block: Number of states in each block
    - D: Number of dimensions (each has same size I_per_block)
    - K: Tensor rank
    '''
    state_mat = torch.rand((K,)*2*D)
    Q = torch.kron(state_mat, torch.ones((I_per_block//K,)*2*D))
    Q = Q / torch.linalg.norm(Q.reshape(-1),1)
    P = Q / Q.sum(dim=tuple(range(D,2*D)),keepdim=True)
    mc = MarkovChainTensor(P)
    return mc

def generate_blockmatrix_model(I_per_block:int,K:int):
    '''
    Inputs:
    - I_per_block: Number of states in each block
    - K: Matrix rank
    '''
    state_mat = torch.rand((K,K))
    Q = torch.kron(state_mat, torch.ones((I_per_block//K,I_per_block//K)))
    Q = Q / torch.linalg.norm(Q.reshape(-1),1)
    P = Q / Q.sum(dim=1,keepdim=True)
    mc = MarkovChainMatrix(P)
    return mc

# ---------------------------------------------

def estimate_empirical_matrix(X, I:int):
    assert set(X) <= set(np.arange(I)), "Inconsistent number of states between input chain X and given dimensions I."

    N = len(X)
    pairwise_counts = torch.zeros((I,I))
    X_all_steps = np.array([np.array(X)[:-1],np.array(X)[1:]])
    X_steps,counts = np.unique(X_all_steps,axis=1,return_counts=True)
    pairwise_counts[X_steps[0],X_steps[1]] = torch.tensor(counts).to(torch.float)
    marginal_counts = pairwise_counts.sum(1,keepdim=True)
    Mask = (marginal_counts==0).expand_as(pairwise_counts)

    Q_emp = pairwise_counts / N
    P_emp = pairwise_counts / marginal_counts
    P_emp[Mask] = 1/I

    return P_emp, Q_emp

def estimate_empirical_tensor(X, Is):
    assert len(Is)==len(X[0]), "Inconsistent state dimensions between input chain X and given dimensions Is."
    I = torch.prod(Is).item()
    D = len(Is)
    N = len(X)

    pairwise_counts = torch.zeros(tuple(Is.repeat(2)))
    X_all_steps = np.concatenate([np.array(X)[:-1],np.array(X)[1:]],axis=1).T
    X_steps, counts = np.unique(X_all_steps,axis=1,return_counts=True)
    pairwise_counts[*list(map(tuple,X_steps))] = torch.tensor(counts).to(torch.float)
    marginal_counts = pairwise_counts.sum(tuple(range(D,2*D)),keepdim=True)
    Mask = (marginal_counts==0).expand_as(pairwise_counts)

    Q_emp = pairwise_counts / N
    P_emp = pairwise_counts / marginal_counts
    P_emp[Mask] = 1/I
    return P_emp, Q_emp
