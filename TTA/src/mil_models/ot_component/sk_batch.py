"""
modified from https://github.com/rhfeiyang/PPOT
"""
import torch

class SK_Class(torch.nn.Module):
    def set_rho(self,rho):  # Used to control the sparsity of pseudo labels
        if hasattr(self,"rho"):
            self.rho=rho
    def get_rho(self):
        if hasattr(self,"rho"):
            return self.rho
        else:
            return None
    def set_gamma(self,gamma):
        if hasattr(self,"gamma"):
            self.gamma=gamma
    def get_gamma(self):
        if hasattr(self,"gamma"):
            return self.gamma
        else:
            return None

class SemiCurrSinkhornKnopp(SK_Class):
    """
    naive SinkhornKnopp algorithm for semi-relaxed curriculum optimal transport, one side is equality constraint, the other side is KL divergence constraint (the algorithm is not stable)
    """
    def __init__(self, num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-6, numItermax=1000, rho=0., semi_use=True, prior = None):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon  # Regularization parameter, controls the smoothness of the Sinkhorn-Knopp algorithm.
        self.gamma = gamma  # Regularization coefficient, used to control the strength of semi-relaxed constraints.
        self.stoperr = stoperr  # Error threshold for stopping iteration.
        self.numItermax = numItermax
        self.rho = rho  # Total mass constraint, used to control the allocation quality of pseudo labels.
        self.b = None
        self.semi_use = semi_use  # Whether to enable semi-relaxed constraints.
        self.prior = prior.reshape(-1,1) if prior is not None else None  # Prior distribution, used to adjust the allocation of pseudo labels to make them more consistent with the target distribution.
        # print(f"prior: {prior}")
        # print(f"semi_use: {semi_use}")
        # print(f"epsilon: {epsilon}")
        # print(f"sk_numItermax: {numItermax}")
    
    # cost_forward() function: Calculate pseudo label allocation matrix (transport plan matrix).
    # Steps:
    # Initialize Pa and Pb, representing the initial distribution of samples and prior distribution of classes respectively.
    # Use Sinkhorn-Knopp algorithm for iterative updates, calculate the final transport plan matrix plan through successive scaling of a and b.
    # If semi_use parameter is set, adjust the update of b to ensure satisfaction of semi-relaxed constraints.
    # Return the final pseudo label allocation matrix plan.
    # @torch.no_grad()
    def cost_forward(self, cost, final=True, count=False, pred_order=None):
        device = cost.device
        cost = cost.double() # Cost matrix cost
        
        # check if batch input
        if cost.dim() == 3:
            # batch mode: cost shape is (B, M, N)
            batch_size, n, k = cost.shape
            is_batch = True
        else:
            # single sample mode: cost shape is (M, N)
            n, k = cost.shape
            batch_size = 1
            is_batch = False
            cost = cost.unsqueeze(0)  # add batch dimension
        
        # 为每个batch创建mu向量
        mu = torch.zeros(batch_size, n, 1).to(device)
        expand_cost = torch.cat([cost, mu], dim=-1)  # Add a column of zeros at the end of the cost matrix to support semi-relaxed constraints.
        Q = torch.exp(- expand_cost / self.epsilon)  # Initial transport plan matrix Q

        # prior distribution
        # Pa represents the initial distribution of samples, which is uniform
        Pa = torch.ones(batch_size, n, 1).to(device) / n  # how many samples 。
        # Pb represents the prior distribution of classes, by uniformly distributing the number of classes to the total mass constraint rho
        Pb = self.rho * torch.ones(batch_size, Q.shape[-1], 1).to(device) / k # how many prototypes
        # If prior exists, adjust Pb accordingly
        if self.prior is not None:
            if pred_order is None:
                pred_distribution = cost.sum(dim=1)  # (B, N)
                pred_order = pred_distribution.argsort(dim=-1, descending=True)
            # print(f"pred_order: {pred_sort_order}")
            for b in range(batch_size):
                Pb[b, pred_order[b], :] = self.prior * self.rho
        Pb[:, -1] = 1 - self.rho

        # a and b are vectors used for progressive scaling, adjusted iteratively to satisfy constraints.
        # init b
        if self.b is None:
            b = torch.ones(batch_size, Q.shape[-1], 1).double().to(device) / Q.shape[-1]
        else:
            if is_batch:
                b = self.b
            else:
                # check the shape of self.b
                if self.b.shape[0] != batch_size:
                    raise ValueError(f"self.b.shape[0] must be equal to batch_size, but got {self.b.shape[0]} and {batch_size}")
                b = self.b.unsqueeze(0)

        fi = self.gamma / (self.gamma + self.epsilon)
        err = 1
        last_b = b.clone()
        iternum = 0
        while err > self.stoperr and iternum < self.numItermax:
            # batch matrix multiplication
            a = Pa / torch.bmm(Q, b)
            b = Pb / torch.bmm(Q.transpose(-2, -1), a)

            # If semi_use is set, perform power operation adjustment on b to ensure satisfaction of semi-relaxed constraints.
            if self.semi_use:
                b[:, :-1, :] = torch.pow(b[:, :-1, :], fi)

            err = torch.norm(b - last_b)
            last_b = b.clone()
            iternum += 1

        plan = a * Q * b.transpose(-2, -1)
        # If final=True, multiply plan by the number of samples to ensure satisfaction of total mass constraint.
        if final:
            plan *= Q.shape[1]
        self.b = b # for two view speed up
        # print(f"sk_iter: {iternum}"
        # print(iternum,end=" ")
        # scale the plan
        # plan = plan / torch.sum(plan, dim=1, keepdim=True)
        
        # Reason for removing the last column.
        # Semi-relaxed constraint: In semi-relaxed optimal transport problems, the last column is to introduce a virtual class to absorb some remaining mass of samples, thus achieving semi-relaxed effect
        plan = plan[:, :, :-1].float()  
        # loss = (plan * cost).sum()
        # print(f"sk loss: {loss}")

        # 如果不是批量模式，去掉batch维度
        if not is_batch:
            plan = plan.squeeze(0)
            if count:
                return (plan, iternum)
            else:
                return plan

        # plan is the final pseudo label allocation matrix, i.e., the transport plan matrix from samples to classes.
        return (plan, iternum) if count else plan

    # forward(self, logits) method
    # Function: Calculate negative log-likelihood based on model logits and call cost_forward to calculate final pseudo label allocation.
    # Steps:
    # Convert logits to probability distribution and calculate its negative logarithm.
    # Calculate pseudo label allocation through cost_forward method.
    # @torch.no_grad()
    def forward(self, logits):
        # logits are the output scores of the model for samples, representing the scores of samples belonging to each class.
        # logits = logits.detach()
        
        # check if batch input
        if logits.dim() == 3:
            # batch mode: logits shape is (B, N, Dim)
            batch_size, n, dim = logits.shape
            is_batch = True
        else:
            # single sample mode: logits shape is (N, Dim)
            n, dim = logits.shape
            batch_size = 1
            is_batch = False
            logits = logits.unsqueeze(0)  # add batch dimension
        
        cost = -torch.log(torch.softmax(logits, dim=-1))  # Get cost matrix
        
        return self.cost_forward(cost)  # Use the calculated cost matrix to call cost_forward, generating pseudo label allocation matrix.



# This is the stable version of SemiCurrSinkhornKnopp class, improving numerical stability.
# Main difference: Added u and v variables for stable computation during iteration.
class SemiCurrSinkhornKnopp_stable(SemiCurrSinkhornKnopp):
    """
    naive SinkhornKnopp algorithm for semi-relaxed curriculum optimal transport, one side is equality constraint, the other side is KL divergence constraint (the algorithm is not stable)
    """
    def __init__(self, num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-10, numItermax=1000, rho=0., semi_use=True, prior = None):
        super().__init__(num_iters, epsilon, gamma, stoperr, numItermax, rho, semi_use, prior)
        self.u = None  # u and v: dual variables, used for numerical adjustment during iteration to avoid numerical overflow caused by excessive scaling operations.
        self.v = None
    
    # reset(self) method: Used to reset u, v and b.
    # Before each iteration, if the shape of dual variables is inconsistent with current input data, call this method to reinitialize variables, ensuring correctness of computation process.
    def reset(self):
        self.u=None
        self.v=None
        self.b=None
    
    # Calculate pseudo label allocation matrix (transport plan matrix)
    # @torch.no_grad()
    def cost_forward(self, cost, final=True,count=False, pred_order=None):
        # Process input cost matrix cost:
        # Convert cost to double precision type and add a column of zeros at the end to form expand_cost for supporting semi-relaxed optimal transport.
        # If dual variables u and v don't exist or shapes don't match, call reset() method to reinitialize.
        device = cost.device
        cost = cost.double()
        
        # check if batch input
        if cost.dim() == 3:
            # batch mode: cost shape is (B, M, N)
            batch_size, n, k = cost.shape
            is_batch = True
        else:
            # single sample mode: cost shape is (M, N)
            n, k = cost.shape
            batch_size = 1
            is_batch = False
            cost = cost.unsqueeze(0)  # add batch dimension
        
        if self.u is not None and self.u.shape[1] != n:
            self.reset()
        
        # 为每个batch创建mu向量
        mu = torch.zeros(batch_size, n, 1).to(device)
        expand_cost = torch.cat([cost, mu], dim=-1)  # B*N*(K+1)

        # Initialize dual variables u and v and initial plan matrix Q
        if self.u is None:
            u = torch.zeros(batch_size, n, 1).to(device)
            v = torch.zeros(batch_size, k+1, 1).to(device)
            Q = torch.exp(- expand_cost / self.epsilon)
        else:
            u = self.u
            v = self.v
            Q = torch.exp((u - expand_cost + v.transpose(1, 2)) / self.epsilon)

        # Pa and Pb represent the initial distribution of samples and prior distribution of classes respectively
        # prior distribution
        Pa = torch.ones(batch_size, n, 1).to(device) / n  # how many samples
        Pb = self.rho * torch.ones(batch_size, Q.shape[-1], 1).to(device) / k # how many prototypes
        if self.prior is not None:
            if pred_order is None:
                pred_distribution = cost.sum(dim=1)  # (B, N)
                pred_order = pred_distribution.argsort(dim=-1, descending=True)
            # print(f"pred_order: {pred_sort_order}")
            for b in range(batch_size):
                Pb[b, pred_order[b], :] = self.prior * self.rho
        Pb[:, -1] = 1 - self.rho
        fi = self.gamma / (self.gamma + self.epsilon)

        # Initialize scaling vector b and weight vector w
        if self.b is None:
            b = torch.ones(batch_size, Q.shape[-1], 1, dtype=Q.dtype).to(device) / Q.shape[-1]
        else:
            if is_batch:
                b = self.b
            else:
                # check the shape of self.b
                if self.b.shape[0] != batch_size:
                    raise ValueError(f"self.b.shape[0] must be equal to batch_size, but got {self.b.shape[0]} and {batch_size}")
                b = self.b.unsqueeze(0)
        
        w = torch.exp(v[:, :-1, :] * (fi - 1) / self.epsilon)
        
        err = 1
        last_b = b.clone()
        iternum = 0
        stabled = False
        while err > self.stoperr and iternum < self.numItermax:
            # batch matrix multiplication
            a = Pa / torch.bmm(Q, b)
            b = Pb / torch.bmm(Q.transpose(-2, -1), a)
            if self.semi_use:
                # b[:, :-1, :] = torch.pow(b[:, :-1, :], fi) * w
                b = torch.cat([
                    torch.pow(b[:, :-1, :], fi) * w,
                    b[:, -1:, :]
                ], dim=1)

            # print((a*Q*b.T).sum(), err)

            err = torch.norm(b - last_b)
            # In each iteration, if the maximum values of a and b exceed a certain threshold, enable update of dual variables u and v
            if max(a.max(), b.max()) > 1e8:
                # print(f"stabled at {iternum}")
                # u += self.epsilon * torch.log(a)
                # v += self.epsilon * torch.log(b + torch.finfo(b.dtype).eps)
                # w *= torch.pow(b[:-1,:], fi-1)
                u = u + self.epsilon * torch.log(a)
                v = v + self.epsilon * torch.log(b + torch.finfo(b.dtype).eps)
                w = w * torch.pow(b[:, :-1, :], fi-1)
                Q = torch.exp((u - expand_cost + v.transpose(1, 2)) / self.epsilon)
                b = torch.ones_like(b)  # Replace in-place operation
                # b[:,:] = 1
                # a[:,:] = 1
                stabled = True
            else:
                stabled = False

            last_b = b.clone()
            iternum += 1

        plan = Q if stabled else a * Q * b.transpose(-2, -1)
        if final:  # If final=True, multiply plan by the number of samples to satisfy total mass constraint.
            plan *= Q.shape[1]
        self.b = b # for two view speed up
        self.u = u
        self.v = v
        # print(f"sk_iter: {iternum}")
        # print(iternum,end=" ")

        plan = plan[:, :, :-1].float()
        # loss = (plan * cost).sum()
        # print(f"sk_stable loss: {loss}")
        
        # if not batch mode, remove batch dimension
        if not is_batch:
            plan = plan.squeeze(0)
            if count:
                return (plan, iternum)
            else:
                return plan
        
        # count is a boolean variable indicating whether to return iteration count
        return (plan, iternum) if count else plan


class MMOT(torch.nn.Module):
    def __init__(self, sk, lam1 = 0.5, numItermax = 10, stoperr=1e-6, lam_fix=False, ema=1):
        super().__init__()
        self.sk = sk
        self.lam_init = lam1
        self.lam1 = lam1
        self.numItermax = numItermax
        self.stoperr = stoperr
        self.lam_fix = lam_fix
        self.ema=ema
        # print(f"MM ema factor: {self.ema}")

    def set_rho(self,rho):
        self.sk.set_rho(rho)
        if not self.lam_fix:
            self.lam1 = (1-rho) * self.lam_init

    def set_gamma(self,gamma):
        self.sk.set_gamma(gamma)
    def get_rho(self):
        print(f"MM lam1: {self.lam1}")
        return self.sk.get_rho()
    def get_gamma(self):
        return self.sk.get_gamma()
    
    def get_grad_omega1(self, Q):
        # get gradient for <S, Q @ Q.T>
        # return (S+S.T) @ Q
        # check if batch input
        if Q.dim() == 3:
            # batch mode: Q shape is (B, N, K)
            return torch.bmm(self.s_st, Q)
        else:
            # single sample mode: Q shape is (N, K)
            return self.s_st @ Q

    def get_omega1(self, S, Q):
        # check if batch input
        if Q.dim() == 3:
            # batch mode: Q shape is (B, N, K), S shape is (B, N, N)
            return (S * torch.bmm(Q, Q.transpose(-2, -1))).sum(dim=(-2, -1))
        else:
            # single sample mode: Q shape is (N, K), S shape is (N, N)
            return (S * (Q @ Q.T)).sum()

    def get_cost(self, C, Q):
        f1_grad = C
        f2_grad = - self.lam1 * self.get_grad_omega1(Q)
        # print(f"min_f1_grad: {f1_grad.min()}, min_f2_grad: {f2_grad.min()}")
        return f1_grad+f2_grad

    def objective_func(self, Q, S, C):
        # check if batch input
        if Q.dim() == 3:
            # batch mode
            term1 = (Q * C).sum(dim=(-2, -1))  # shape: B
            term2 = self.lam1 * self.get_omega1(S=S, Q=Q)  # shape: B
            term3 = self.sk.epsilon * (Q * torch.log(Q + torch.finfo(torch.float).tiny)).sum(dim=(-2, -1))
            return term1 - term2 + term3
        else:
            # single sample mode
            return (Q * C).sum() - self.lam1 * self.get_omega1(S=S, Q=Q) + self.sk.epsilon * (Q * torch.log(Q + torch.finfo(torch.float).tiny)).sum()

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, S: torch.Tensor):
        # get C
        # S = S.detach()
        # logits = logits.detach()
        
        # 检查是否为批量输入
        if logits.dim() == 3:
            # batch mode: logits shape is (B, N, Dim), S shape is (B, N, N)
            batch_size, n, dim = logits.shape
            is_batch = True
        else:
            # single sample mode: logits shape is (N, Dim), S shape is (N, N)
            n, dim = logits.shape
            batch_size = 1
            is_batch = False
            logits = logits.unsqueeze(0)  # add batch dimension
            S = S.unsqueeze(0)  # add batch dimension
        
        # reset self.b to ensure correct shape
        self.sk.b = None
        
        self.s_st = S + S.transpose(-2, -1)
        prob = torch.softmax(logits, dim=-1)
        C = -torch.log(prob)
        
        pred_sort_order = None
        if self.sk.prior is not None:
            if is_batch:
                pred_distribution = C.sum(dim=1)  # (B, N)
                pred_sort_order = pred_distribution.argsort(dim=-1, descending=True)
            else:
                pred_distribution = C.sum(dim=0)
                pred_sort_order = pred_distribution.argsort(descending=True)
        
        # TODO: how to initialize Q ?
        if is_batch:
            Q = torch.ones_like(C) / (C.shape[1] * C.shape[2]) * self.sk.rho
        else:
            Q = torch.ones_like(C) / (C.shape[0] * C.shape[1]) * self.sk.rho
        # Q = prob * self.sk.rho # 38 170.5758

        # last_Q = Q
        old_fval = self.objective_func(Q, S, C)
        sk_iter_count = 0
        for i in range(self.numItermax):
            tmp_cost = self.get_cost(C, Q) # gradient of f(Q)
            Q_new, count = self.sk.cost_forward(tmp_cost, final=False, count=True, pred_order=pred_sort_order)
            Q = (1-self.ema) * Q + self.ema * Q_new
            sk_iter_count += count
            new_fval = self.objective_func(Q, S, C)
            # print(f"mm_fval: {new_fval}")
            fval_delta = abs(new_fval - old_fval)
            if is_batch:
                # check if all batch's fval_delta is less than stoperr
                if torch.all(fval_delta < self.stoperr):
                    break
            else:
                # check if fval_delta is less than stoperr
                if fval_delta < self.stoperr:
                    break
            old_fval = new_fval

        # print(f"{i+1},{sk_iter_count}", end=" ")
        # final_fval = self.objective_func(Q, S, C)
        if is_batch:
            return Q * Q.shape[1]
        else:
            return Q * Q.shape[0]


    