from src.environments import *
from time import perf_counter

from scipy.sparse.linalg import svds

# ################################################################
def sgsADMM(P_emp, beta:float=1., gamma:float=1., pmin:float=0.,
            num_itrs:int=5000, tol:float=1e-6, K:int=None,
            verbose:bool=False):
    # ------------ CHECK ------------
    check_valid_transition_mat(P_emp)
    I = P_emp.shape[0]
    # ------------ CHECK ------------
    

    # ------------ ADMM UPDATES ------------
    update_y = lambda P,W,S,beta: (torch.eye(P.shape[0]) - P - beta*( W + S )).sum(1) / (beta*P.shape[0])
    def update_W(P,S,y,P_emp,beta):
        R = torch.outer(y,torch.ones(I)) + S + P/beta
        Z = .5 * ( R + torch.sqrt( R**2 + 4*P_emp/beta ) ) * (P_emp!=0) + torch.maximum(R,torch.zeros_like(R)) * (P_emp==0)
        return Z - R
    def update_S(P,W,y,beta,gamma,K=None):
        R = -( W + torch.outer(y,torch.ones(P.shape[0])) + P/beta )
        if not torch.allclose(R,torch.zeros_like(R)):
            try:
                if K is None:
                    U,sig,V = torch.svd( R )
                else:
                    U,sig,V = torch.svd_lowrank( R, K )
            except:
                U, sig, V = np.linalg.svd( R.numpy().astype(float) )
                U = torch.FloatTensor(U); sig = torch.FloatTensor(sig); V = torch.FloatTensor(V)
        else:
            U, sig, V = np.linalg.svd( R.numpy().astype(float) )
            U = torch.FloatTensor(U); sig = torch.FloatTensor(sig); V = torch.FloatTensor(V)
        sig_upd = torch.minimum(sig,gamma*torch.ones_like(sig))
        return U@torch.diag(sig_upd)@V.T
    update_P = lambda P,W,S,y,beta,gamma: torch.maximum( P + gamma*beta * ( W + torch.outer(y,torch.ones(P.shape[0])) + S ), pmin*torch.ones_like(P) )
    # ------------ ADMM UPDATES ------------


    # ------------ INITIALIZE ------------
    P_est = torch.rand(I,I); P_est = P_est / P_est.sum(1,keepdim=True)
    W_est = torch.zeros((I,I))
    S_est = torch.zeros((I,I))
    y_est = -torch.ones(I)

    diffs = []
    costs = []
    # ------------ INITIALIZE ------------


    # ------------ LOOP ------------
    tic = perf_counter()
    for itr in range(num_itrs):
        # ------------ UPDATE STEPS ------------
        y_last = y_est.clone()
        W_last = W_est.clone()
        S_last = S_est.clone()
        P_last = P_est.clone()

        y_est = update_y(P_est, W_est, S_est, beta)
        W_est = update_W(P_est, S_est, y_est, P_emp, beta)
        y_est = update_y(P_est, W_est, S_est, beta)
        S_est = update_S(P_est, W_est, y_est, beta, gamma, K)
        P_est = update_P(P_est, W_est, S_est, y_est, beta, gamma)
        # ------------ UPDATE STEPS ------------


        # ------------ SAVE ITERATION ------------
        diff = torch.tensor([ torch.norm(y_est-y_last), torch.norm(W_est-W_last), torch.norm(S_est-S_last), torch.norm(P_est-P_last) ]).max().item()
        cost = kld_err(P_emp,P_est)

        diffs.append(diff)
        costs.append(cost)

        toc = perf_counter()
        if verbose and toc-tic>5:
            print(f"Iter. {itr+1}/{num_itrs} | Cost: {cost:.1e} | Diff: {diff:.1e}")
            tic = perf_counter()
        # ------------ SAVE ITERATION ------------


        # ------------ TERMINATION ------------
        if diff<tol:
            print(f"Terminating early @ {itr+1}/{num_itrs} iters. Difference {diff:.1e} below threshold {tol:.2e}.")
            break
        # ------------ TERMINATION ------------
    # ------------ LOOP ------------

    P_est = P_est / P_est.sum(1,keepdim=True)
    mc_est = MarkovChainMatrix(P_est)

    return dict(mc_est=mc_est, diffs=diffs, costs=costs)
# ################################################################



# ################################################################
def iPDC(P_emp, K:int, beta:float=1., gamma:float=1., alpha:float=1e-1, pmin:float=0.,
            num_itrs:int=5000, tol:float=1e-6,
            num_inn_itrs:int=1, inn_tol:float=1e-6,
            verbose:bool=False):
    # ------------ CHECK ------------
    check_valid_transition_mat(P_emp)
    I = P_emp.shape[0]
    # ------------ CHECK ------------
    

    # ------------ PDC UPDATES ------------
    update_y = lambda P,W,S,beta: (torch.eye(P.shape[0]) - P - beta*( W + S )).sum(1) / (beta*P.shape[0])
    def update_W(P,S,T,y,P_emp,alpha,beta):
        R = torch.outer(y,torch.ones_like(y)) + S + P/beta
        Z = (.5/(alpha+1)) * ( (R-T/beta) + torch.sqrt( (R-T/beta)**2 + 4*(alpha+1)*P_emp/beta ) ) * (P_emp!=0) + torch.maximum(R-T/beta,torch.zeros_like(R)) * (P_emp==0)
        return Z - R
    def update_S(P,W,y,beta,gamma,K=None):
        R = -( W + torch.outer(y,torch.ones(P.shape[0])) + P/beta )
        if not torch.allclose(R,torch.zeros_like(R)):
            if K is None:
                U,sig,V = torch.svd( R )
            else:
                U,sig,V = torch.svd_lowrank( R, K )
        else:
            U, sig, V = np.linalg.svd( R.numpy().astype(float) )
            U = torch.FloatTensor(U); sig = torch.FloatTensor(sig); V = torch.FloatTensor(V)
        sig_upd = torch.minimum(sig,gamma*torch.ones_like(sig))
        return U@torch.diag(sig_upd)@V.T
    update_P = lambda P,W,S,y,beta,gamma: torch.maximum( P + gamma*beta * ( W + torch.outer(y,torch.ones(P.shape[0])) + S ), pmin*torch.ones_like(P) )
    def update_T(P,K):
        if not torch.allclose(P,torch.zeros_like(P)):
            U,sig,V = svds(P_est.numpy().astype(float),k=K)
            T = torch.FloatTensor(U@np.diag(sig)@V)
        else:
            T = P.clone()
        return T
    # ------------ PDC UPDATES ------------


    # ------------ INITIALIZE ------------
    P_est = torch.ones((I,I))/I
    W_est = torch.zeros((I,I))
    S_est = torch.zeros((I,I))
    y_est = -torch.ones(I)
    T_est = update_T(P_est,K)

    diffs = []
    costs = []
    # ------------ INITIALIZE ------------


    # ------------ LOOP ------------
    tic = perf_counter()
    for itr in range(num_itrs):
        # ------------ UPDATE STEPS ------------
        y_last = y_est.clone()
        W_last = W_est.clone()
        S_last = S_est.clone()
        P_last = P_est.clone()
        T_last = T_est.clone()

        T_est = update_T(P_est,K)

        for inn_itr in range(num_inn_itrs):
            y_inn_last = y_est.clone()
            W_inn_last = W_est.clone()
            S_inn_last = S_est.clone()
            P_inn_last = P_est.clone()

            y_est = update_y(P_est, W_est, S_est, beta)
            W_est = update_W(P_est, S_est, T_est, y_est, P_emp, alpha, beta)
            y_est = update_y(P_est, W_est, S_est, beta)
            S_est = update_S(P_est, W_est, y_est, beta, gamma, K)
            P_est = update_P(P_est, W_est, S_est, y_est, beta, gamma)

            inn_diff = torch.tensor([ torch.norm(y_est-y_inn_last), torch.norm(W_est-W_inn_last), torch.norm(S_est-S_inn_last), torch.norm(P_est-P_inn_last) ]).max().item()
            if inn_diff<inn_tol:
                break
        # ------------ UPDATE STEPS ------------
        
        
        # ------------ SAVE ITERATION ------------
        diff = torch.tensor([ torch.norm(y_est-y_last), torch.norm(W_est-W_last), torch.norm(S_est-S_last), torch.norm(P_est-P_last), torch.norm(T_est-T_last) ]).max().item()
        cost = kld_err(P_emp,P_est)

        diffs.append(diff)
        costs.append(cost)
        
        toc = perf_counter()
        if verbose and toc-tic>5:
            print(f"Iter. {itr} | Diff: {diff:.2e}")
            tic = perf_counter()
        # ------------ SAVE ITERATION ------------


        # ------------ TERMINATION ------------
        if diff<tol:
            print(f"Terminating early @ {itr+1}/{num_itrs} iters. Difference {diff:.1e} below threshold {tol:.2e}.")
            break
        # ------------ TERMINATION ------------
    # ------------ LOOP ------------

    P_est = P_est / P_est.sum(1,keepdim=True)
    mc_est = MarkovChainMatrix(P_est)

    return dict(mc_est=mc_est, diffs=diffs, costs=costs)
# ################################################################



# ################################################################
def SLRM(Q_emp, K:int, qmin:float=0.):
    # ------------ CHECK ------------
    check_valid_joint_mat(Q_emp)
    # ------------ CHECK ------------


    # ------------ SPECTRAL THRESHOLDING ------------
    if not torch.allclose(Q_emp,torch.zeros_like(Q_emp)):
        U,sig,V = svds(Q_emp.numpy().astype(float), k=K)
        Q_est = torch.maximum( torch.FloatTensor(U@np.diag(sig)@V), qmin*torch.ones_like(Q_emp) )
    else:
        U, sig, V = np.linalg.svd( Q_emp.numpy().astype(float) )
        U = torch.FloatTensor(U); sig = torch.FloatTensor(sig); V = torch.FloatTensor(V)
        Q_est = torch.maximum( torch.FloatTensor(U[:,:K]@np.diag(sig[:K])@V[:K,:]), qmin*torch.ones_like(Q_emp) )
    # ------------ SPECTRAL THRESHOLDING ------------
    
    
    # ------------ CONVERT TO TRANSITION MATRIX ------------
    Q_est = Q_est / torch.linalg.norm(Q_est,1)
    R_est = .5 * (Q_est.sum(0) + Q_est.sum(1))
    P_est = torch.diag(1/(R_est + (R_est==0).to(torch.float)))@Q_est
    Marginal = P_est.sum(1,keepdim=True)
    Mask = (Marginal==0).expand_as(P_est)
    P_est = P_est / Marginal
    P_est[Mask] = 1/len(Marginal)
    # ------------ CONVERT TO TRANSITION MATRIX ------------


    P_est = P_est / P_est.sum(1,keepdim=True)
    mc_est = MarkovChainMatrix(P_est)
    cost = kld_err(Qmat2Pmat(Q_emp),P_est)

    return dict(mc_est=mc_est, cost=cost)
# ################################################################



# ################################################################
def SCPD(chain, Q_emp, K:int, 
         SAMPLING_TYPE:str=None,
         qmin:float=0., qmax:float=1.,
         num_itrs:int=1000, tol:float=1e-8, slide_window:int=50, 
         ALPHA_TYPE:str="adam", alpha_factor:float=1., alpha_weight:float=1.,
         gamma_factor:float=0., gamma_weight:float=0.,
         beta:float=1e-1, eps:float=1e-9,
         B:int=10, B_max:int=None, INCREASE_B:bool=False,
         ACCELERATION:bool=False,
         ):
    '''
    Stochastic projected block gradient descent for CPD of joint PMF. 
    Updates are performed using fiber, entry-wise, or trajectory sampling.
    '''
    # ------------ CHECK ------------
    check_valid_joint_ten(Q_emp)
    DD = Q_emp.ndim
    D = DD//2
    IIs = torch.tensor(Q_emp.shape)
    Is = IIs[:D]
    II = IIs.prod().item()
    I = Is.prod().item()

    assert (1 - IIs*qmin >= 0).all() and (1 - K*qmin >= 0), "Invalid minimum probability. Impossible to satisfy for all dimensions."
    assert qmin<qmax, "Invalid maximum and minimum probabilities. Must have `qmin < qmax`."
    assert SAMPLING_TYPE in [None, "fiber", "entry", "trajectory"], "Invalid sampling type. Expecting `fiber`, `entry`, or `trajectory`."
    if SAMPLING_TYPE is None:
        SAMPLING_TYPE = "trajectory"
    assert ALPHA_TYPE in [None,"decay","adam","constant"], "Invalid step size type. Expecting `decay`, `adam`, or `constant`."
    if ALPHA_TYPE is None:
        ALPHA_TYPE = "constant"
    # ------------ CHECK ------------


    # ------------ INITIALIZE ------------
    dim_probs = np.ones(DD+1)
    dim_probs = dim_probs / dim_probs.sum()

    obs_pairs = None if not (SAMPLING_TYPE=="entry") else torch.where(Q_emp.flatten()!=0)[0]
    R_emp = None if not (SAMPLING_TYPE=="trajectory") else .5 * (Q_emp.sum(tuple(range(D,2*D))) + Q_emp.sum(tuple(range(D))))
    chain_transitions = None if not (SAMPLING_TYPE=="trajectory") else torch.hstack([torch.stack(chain[1:]), torch.stack(chain[:-1])])

    Q_est, Qds, l = generate_tensor(IIs,K)
    Qds_upd = [Qds[d].clone() for d in range(DD)]
    l_upd = l.clone()
    mttkrp = int(np.ceil( II / B / IIs.min() ).item()) if SAMPLING_TYPE=="fiber" else int(np.ceil( II / IIs.min() ).item())
    num_mttkrps = int(num_itrs/mttkrp + .5)

    if B_max is None:
        B_max = np.inf
    B_max = np.maximum(B,B_max)
    B_init = int(B)
    alpha_factor_init = float(alpha_factor)
    alpha_weight_init = float(alpha_weight)
    factor_grad_mags = torch.zeros(DD)
    weight_grad_mag = 0.
    diffs = [None]*num_itrs
    variances = [None]*num_itrs
    costs = [None]*num_itrs
    # ------------ INITIALIZE ------------


    # ------------ LOOP ------------
    t = 0.
    TERMINATE = False
    tot_itr = 0
    tic = perf_counter()
    for out_itr in range(num_mttkrps):
        for inn_itr in range(mttkrp):
            Q_last = Q_est.clone()
            t_last = float(t)
            t = .5 * (1 + np.sqrt(1 + 4 * t_last**2))

            # ------------ STOCHASTIC SAMPLING ------------
            # Sample d
            d = np.random.choice(DD+1,p=dim_probs)
            if INCREASE_B:
                B_new = int( np.ceil( (out_itr+1)**.5 ) )
                B = np.minimum( np.maximum( B_new, B_init ), B_max )
            # ------------ STOCHASTIC SAMPLING ------------

            if d<DD:
                # ------------ FACTOR UPDATE ------------
                Gd = _factor_compute_grad(d, chain, Q_emp, Qds_upd, l, SAMPLING_TYPE, gamma_factor, eps, B, Q_est, obs_pairs, R_emp, chain_transitions)
                factor_grad_mags[d] += torch.linalg.norm(Gd.reshape(-1),1)
                Qd_last = Qds[d].clone()
                Q_dot, alpha_factor_init = _scpd_grad_step(out_itr,Qds[d],Gd,alpha_factor_init,factor_grad_mags[d],beta,eps,ALPHA_TYPE)
                Qds[d] = _scpd_proj_factor(Q_dot,qmin,qmax)

                Qds_upd[d] = Qds[d] + (1-t_last)/t * (Qds[d] - Qd_last) if ACCELERATION else Qds[d].clone()
                # ------------ FACTOR UPDATE ------------
            else:
                # ------------ WEIGHT UPDATE ------------
                gl = _weight_compute_grad(chain, Q_emp, Qds, l_upd, SAMPLING_TYPE, gamma_weight, eps, B, obs_pairs, R_emp, chain_transitions)
                weight_grad_mag += torch.linalg.norm(gl.reshape(-1),1)
                l_last = l.clone()
                l_dot, alpha_weight_init = _scpd_grad_step(out_itr,l,gl,alpha_weight_init,weight_grad_mag,beta,eps,ALPHA_TYPE)
                l = _scpd_proj_weight(l_dot,qmin,qmax)

                l_upd = l + (1-t_last)/t * (l - l_last) if ACCELERATION else l.clone()
                # ------------ WEIGHT UPDATE ------------
            
            # ------------ SAVE ITERATION ------------
            Q_est = cp_to_tensor((l,Qds))
            diff = norml1_err(Q_est,Q_last)
            cost = kld_err(Q_emp,Q_est)
            var = np.var(costs[tot_itr-slide_window:tot_itr]) if tot_itr>=slide_window else 0.

            if d<DD:
                alpha_factor = _update_alpha(alpha_factor_init,out_itr,factor_grad_mags[d],beta,eps,ALPHA_TYPE)
            else:
                alpha_weight = _update_alpha(alpha_weight_init,out_itr,weight_grad_mag,beta,eps,ALPHA_TYPE)

            if len(diffs)<=tot_itr:
                diffs.append(diff)
                costs.append(cost)
                variances.append(var)
            else:
                diffs[tot_itr] = diff
                costs[tot_itr] = cost
                variances[tot_itr] = var

            toc = perf_counter()
            if (out_itr==0 and tot_itr==0) or (toc-tic>2):
                print(f"MTTKRP: {out_itr+1}/{num_mttkrps} | Cost: {cost:.3e} | Diff: {diff:.1e} | Var: {var:.1e} | af: {alpha_factor:.1e} | aw: {alpha_weight:.1e}")
                tic = perf_counter()
            # ------------ SAVE ITERATION ------------

            
            # ------------ TERMINATION ------------
            if np.isnan(cost) or torch.allclose(Q_est,torch.zeros_like(Q_est)):
                print(f"Invalid solution. Returning null.")
                return None
            
            if tot_itr>=slide_window and var<tol:
                print(f"Terminating early @ {out_itr+1}/{num_mttkrps} MTTKRPs. Variance of objective {var:.1e} below threshold {tol:.2e}.")
                TERMINATE = True
                break
            # ------------ TERMINATION ------------

            tot_itr+=1

        if TERMINATE:
            break
    # ------------ LOOP ------------
    
    if None in diffs:
        idx = diffs.index(None)
        diffs = diffs[:idx]
        costs = costs[:idx]

    mc_est = MarkovChainTensor(Qten2Pten(Q_est))

    return dict(mc_est=mc_est, diffs=diffs, costs=costs, Qds=Qds, l=l)
# ################################################################




# ################################################################
def _weight_compute_grad(chain, Q_emp, Qds, l, 
                         SAMPLING_TYPE:str, gamma_weight:float, eps:float, B:int,
                         obs_pairs=None, R_emp=None, chain_transitions=None,
                    ):
    DD = Q_emp.ndim
    D = DD//2
    IIs = torch.tensor(Q_emp.shape)
    Is = IIs[:D]
    II = IIs.prod().item()
    I = Is.prod().item()

    # ------------ FIBER GRADIENT ------------
    if SAMPLING_TYPE=="fiber":
        B = np.minimum(B,int(II))
        fibs = np.random.choice(int(II),B, replace=False)

        ml = Q_emp.reshape(-1)[fibs]
        Hl = khatri_rao(Qds)[fibs]
        gl = -( ml / (Hl@l + eps) ) @ Hl / B

        if gamma_weight>0:
            B_erg = int(np.minimum( B, I ))
            # B_erg = int(np.minimum( int(np.ceil(B/2)), I ))
            fibs_erg = np.random.choice(I,B_erg, replace=False)

            H0_erg = khatri_rao(Qds[:D])[fibs_erg]
            H1_erg = khatri_rao(Qds[D:])[fibs_erg]
            gl_erg = gamma_weight * ((H0_erg-H1_erg).T@(H0_erg-H1_erg))@l / B_erg
            gl += gl_erg
    # ------------ FIBER GRADIENT ------------

    # ------------ ENTRY GRADIENT ------------
    elif SAMPLING_TYPE=="entry":
        if obs_pairs is None:
            obs_pairs = torch.where(Q_emp.flatten()!=0)[0]
        B = np.minimum( B, len(obs_pairs) )
        inds = np.random.choice(obs_pairs,B, replace=False)
        steps = [torch.tensor(np.unravel_index(step, tuple(IIs))) for step in inds]

        hls = [torch.stack([Qds[d][step[d]] for d in range(DD)]).prod(0) for step in steps]
        gls = [-(Q_emp[*step]*II) * hls[i] / ( hls[i]@l + eps ) for i,step in enumerate(steps)]
        gl = torch.stack(gls).sum(0) / B

        if gamma_weight>0:
            B_erg = np.minimum(B,len(chain))
            steps_erg = torch.stack(chain)[np.random.choice(len(chain),B_erg,replace=False)]
            steps_erg = [step[:D] if np.random.rand()<.5 else step[D:] for step in steps]

            hs0 = [torch.stack([Qds[d][step[d%D]] for d in range(D)]).prod(0) for step in steps_erg]
            hs1 = [torch.stack([Qds[d][step[d%D]] for d in range(D,DD)]).prod(0) for step in steps_erg]
            gls_erg = [gamma_weight * (II*I) * torch.outer(hs0[i]-hs1[i], hs0[i]-hs1[i]) @ l for i,step in enumerate(steps_erg)]
            gl_erg = torch.stack(gls_erg).sum(0) / B_erg
            gl += gl_erg
    # ------------ ENTRY GRADIENT ------------

    # ------------ TRAJECTORY GRADIENT ------------
    elif SAMPLING_TYPE=="trajectory":
        if R_emp is None:
            R_emp = .5 * (Q_emp.sum(tuple(range(D,2*D))) + Q_emp.sum(tuple(range(D))))
        if chain_transitions is None:
            chain_transitions = torch.hstack([torch.stack(chain[1:]), torch.stack(chain[:-1])])
        
        B = np.minimum(B,len(chain)-1)
        inds = np.random.choice(len(chain)-1,B,replace=False)
        steps = chain_transitions[inds]

        hls = [torch.stack([Qds[d][step[d]] for d in range(DD)]).prod(0) for step in steps]
        gls = [-hls[i] / ( hls[i]@l + eps ) for i,step in enumerate(steps)]
        gl = torch.stack(gls).sum(0) / B

        if gamma_weight>0:
            B_erg = np.minimum(B,len(chain))
            steps_erg = torch.stack(chain)[np.random.choice(len(chain),B_erg,replace=False)]
            steps_erg = [step[:D] if np.random.rand()<.5 else step[D:] for step in steps]

            hs0 = [torch.stack([Qds[d][step[d%D]] for d in range(D)]).prod(0) for step in steps_erg]
            hs1 = [torch.stack([Qds[d][step[d%D]] for d in range(D,DD)]).prod(0) for step in steps_erg]
            gls_erg = [gamma_weight / R_emp[*step] * torch.outer(hs0[i]-hs1[i], hs0[i]-hs1[i]) @ l for i,step in enumerate(steps_erg)]
            gl_erg = torch.stack(gls_erg).sum(0) / B_erg
            gl += gl_erg
    # ------------ TRAJECTORY GRADIENT ------------

    return gl

def _factor_compute_grad(d:int, chain, Q_emp, Qds, l, 
                        SAMPLING_TYPE:str, gamma_factor:float, eps:float, B:int,
                        Q_est=None, obs_pairs=None, R_emp=None, chain_transitions=None,
                    ):
    DD = Q_emp.ndim
    D = DD//2
    IIs = torch.tensor(Q_emp.shape)
    Is = IIs[:D]
    II = IIs.prod().item()
    I = Is.prod().item()
    assert d in np.arange(DD), "Invalid dimension `d`."

    # ------------ FIBER GRADIENT ------------
    if SAMPLING_TYPE=="fiber":
        B = np.minimum(B, int(II/IIs[d]))
        fibs = np.random.choice(int(II/IIs[d]),B, replace=False)

        Md = (unfold(Q_emp,mode=d).T)[fibs]
        Hd = khatri_rao([Qds[i] for i in range(len(Qds)) if i!=d])[fibs]
        L = ( Md / (Hd@torch.diag(l)@Qds[d].T + eps ) ) * (Md>0)
        Gd = -L.T @ Hd @ torch.diag(l) / B

        if gamma_factor>0:
            B_erg = int(np.minimum( B, int(I/IIs[d]) ))
            # B_erg = int(np.minimum( int(np.ceil(B/2)), int(I/IIs[d]) ))
            fibs_erg = np.random.choice(int(I/IIs[d]),B_erg,replace=False)

            Q0_erg = (unfold( cp_to_tensor((l,Qds[:D])), mode=d%D ).T)[fibs_erg] * (2*(d<D)-1)
            Q1_erg = (unfold( cp_to_tensor((l,Qds[D:])), mode=d%D ).T)[fibs_erg] * -(2*(d<D)-1)
            H_erg = khatri_rao([Qds[i] for i in np.arange(D)+D*(d>=D) if i!=d])[fibs_erg] 
            Gd_erg = gamma_factor * (Q0_erg + Q1_erg).T @ H_erg @ torch.diag(l) / B_erg
            Gd += Gd_erg
    # ------------ FIBER GRADIENT ------------

    # ------------ ENTRY GRADIENT ------------
    elif SAMPLING_TYPE=="entry":
        if Q_est is None:
            Q_est = cp_to_tensor((l,Qds))
        if obs_pairs is None:
            obs_pairs = torch.where(Q_emp.flatten()!=0)[0]

        B = np.minimum(B, len(obs_pairs))
        inds = np.random.choice(obs_pairs,B, replace=False)
        steps = [torch.tensor(np.unravel_index(step, tuple(IIs))) for step in inds]

        hds = [torch.stack([Qds[j][step[j]]  for j in range(DD) if j!=d]).prod(0) for step in steps]
        gds = [ -(Q_emp[*step]*II) * hds[i]*l / (Q_est[*step] + eps) for i,step in enumerate(steps)]
        Gds = [torch.outer(torch.eye(IIs[d])[step[d]],gds[i]) for i,step in enumerate(steps)]
        Gd = torch.stack(Gds).sum(0) / B

        if gamma_factor>0:
            steps_erg = [step[:D] if d<D else step[D:] for step in steps]

            Q0 = cp_to_tensor((l,Qds[:D])) * (2*(d<D)-1)
            Q1 = cp_to_tensor((l,Qds[D:])) * -(2*(d<D)-1)
            hds_erg = [torch.stack([Qds[j][step[j%D]]  for j in np.arange(D)+D*(d>=D) if j!=d]).prod(0) for step in steps_erg]
            gds_erg = [gamma_factor * (II*I) * (Q0[*step]+Q1[*step]) * hds_erg[i] * l for i,step in enumerate(steps_erg)]
            Gds_erg = [torch.outer( torch.eye(IIs[d])[step[d%D]], gds_erg[i] ) for i,step in enumerate(steps_erg)]
            Gd_erg = torch.stack(Gds_erg).sum(0) / B
            Gd += Gd_erg
    # ------------ ENTRY GRADIENT ------------
    
    # ------------ TRAJECTORY GRADIENT ------------
    elif SAMPLING_TYPE=="trajectory":
        if Q_est is None:
            Q_est = cp_to_tensor((l,Qds))
        if R_emp is None:
            R_emp = .5 * (Q_emp.sum(tuple(range(D,2*D))) + Q_emp.sum(tuple(range(D))))
        if chain_transitions is None:
            chain_transitions = torch.hstack([torch.stack(chain[1:]), torch.stack(chain[:-1])])

        B = np.minimum( B, len(chain)-1 )
        inds = np.random.choice(len(chain)-1,B,replace=False)
        steps = chain_transitions[inds]

        hds = [torch.stack([Qds[j][step[j]]  for j in range(DD) if j!=d]).prod(0) for step in steps]
        gds = [-hds[i]*l / (Q_est[*step] + eps) for i,step in enumerate(steps)]
        Gds = [torch.outer(torch.eye(IIs[d])[step[d]],gds[i]) for i,step in enumerate(steps)]
        Gd = torch.stack(Gds).sum(0) / B

        if gamma_factor>0:
            # steps_erg = torch.stack(chain)[np.random.choice(len(chain),B,replace=False)]
            steps_erg = [step[:D] if d<D else step[D:] for step in steps]

            Q0 = cp_to_tensor((l,Qds[:D])) * (2*(d<D)-1)
            Q1 = cp_to_tensor((l,Qds[D:])) * -(2*(d<D)-1)
            hds_erg = [torch.stack([Qds[j][step[j%D]]  for j in np.arange(D)+D*(d>=D) if j!=d]).prod(0) for step in steps_erg]
            gds_erg = [gamma_factor / R_emp[*step] * (Q0[*step]+Q1[*step]) * hds_erg[i] * l for i,step in enumerate(steps_erg)]
            Gds_erg = [torch.outer( torch.eye(IIs[d])[step[d%D]], gds_erg[i] ) for i,step in enumerate(steps_erg)]
            Gd_erg = torch.stack(Gds_erg).sum(0) / B
            Gd += Gd_erg
    # ------------ TRAJECTORY GRADIENT ------------
    
    return Gd

def _update_alpha(alpha,itr:int=None,grad_mag:float=None,beta:float=None,eps:float=None,ALPHA_TYPE:str=None):
    assert ALPHA_TYPE in [None,"decay","adam","constant"], "Invalid step size type. Expecting `decay`, `adam`, or `constant`."
    if ALPHA_TYPE is None:
        ALPHA_TYPE = "constant"
    if ALPHA_TYPE=="decay":
        return alpha / (itr+1)**beta
    elif ALPHA_TYPE=="adam":
        return alpha / (grad_mag**beta + eps) 
    else:
        return alpha

def _scpd_grad_step(itr:int,X,G,alpha_init,grad_mag,beta:float=None,eps:float=None,ALPHA_TYPE:str=None):
    alpha = _update_alpha(alpha_init,itr,grad_mag,beta,eps,ALPHA_TYPE)
    X_dot = (X - alpha * G).numpy().astype(float)
    while all(X_dot.flatten()<=0.):
        alpha_init *= .9
        alpha = _update_alpha(alpha_init,itr,grad_mag,beta,eps,ALPHA_TYPE)
        X_dot = (X - alpha * G).numpy().astype(float)
    return X_dot, alpha_init
def _scpd_proj_factor(Qd,qmin:float=0.,qmax:float=1.):
    s = float((1 - Qd.shape[0]*qmin)/(qmax-qmin))
    return qmin + (qmax-qmin) * torch.FloatTensor(np.array([proj_bounded_simplex((Qd[:,k]-qmin)/(qmax-qmin), s) for k in range(Qd.shape[1])]).T)
def _scpd_proj_weight(l,qmin:float=0.,qmax:float=1.):
    return qmin + (qmax-qmin) * torch.FloatTensor(proj_bounded_simplex( (l-qmin)/(qmax-qmin), (1-len(l)*qmin)/(qmax-qmin) ))
# ################################################################



