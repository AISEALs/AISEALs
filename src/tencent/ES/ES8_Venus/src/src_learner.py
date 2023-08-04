import numpy as np

from optimizers import Adam, SGD, Momentum, CorrectMomentum, VectorMomentum
from src_utils import compute_centered_ranks, weighted_sum, normalize


class Learner(object):
    """
    coef: (n,)
    """
    def __init__(self, coef_dim, optimizer, lr):
        self.coef_dim = coef_dim
        self.lr = lr
        self.coef = None

        self.l2_coeff = 0.005
        if optimizer == 'Adam':
            self.optimizer = Adam(coef_dim, lr)
        elif optimizer == 'SGD':
            self.optimizer = SGD(coef_dim, lr)
        elif optimizer == 'Momentum':
            self.optimizer = Momentum(coef_dim, lr)
        elif optimizer == 'CorrectMomentum':
            self.optimizer = CorrectMomentum(coef_dim, lr)
        elif optimizer == 'VectorMomentum':
            self.optimizer = VectorMomentum(coef_dim, lr)

    def get_weight(self):
        assert not self.coef is None, ('The coef of learner hasnt been set.')
        return self.coef

    def set_weight(self, new_coef):
        assert new_coef.ndim == 1, (new_coef.shape)
        self.coef = new_coef

    def generate_noises(self, noise_num):
        raise NotImplementedError()

    def learn(self, noises, noisy_rewards):
        raise NotImplementedError()        
    

class ESLearner(Learner):
    """
    mirror noises,
    centered-ranked reward
    """
    def __init__(self, coef_dim, optimizer, lr):
        super(ESLearner, self).__init__(coef_dim, optimizer, lr)

    def generate_noises(self, noise_num):
        """
        noises: (noise_num, coef_dim)
        """
        assert noise_num % 2 == 0, (noise_num)
        pos_noises = [np.random.randn(self.coef_dim) for _ in range(noise_num / 2)]     # (noise_num/2, coef_dim)
        neg_noises = [-x for x in pos_noises]
        noises = np.array(pos_noises + neg_noises)  # (noise_num, coef_dim)
        return noises

    def learn(self, noises, noisy_rewards):
        """
        noises: (noise_num, coef_dim)
        noisy_rewards: (noise_num,)
        """
        noise_num = len(noises)
        noisy_rewards = compute_centered_ranks(noisy_rewards)
        pos_noises = noises[:noise_num / 2]
        pos_noisy_rewards = noisy_rewards[:noise_num / 2]     # (noise_num/2,)
        neg_noisy_rewards = noisy_rewards[noise_num / 2:]     # (noise_num/2,)
        g = weighted_sum(pos_noisy_rewards - neg_noisy_rewards, np.array(pos_noises))   # (num_coef,)
        g /= noise_num

        cur_coef = self.get_weight()
        new_coef, update_ratio = self.optimizer.update(cur_coef, -g + self.l2_coeff * cur_coef)
        self.set_weight(new_coef)
        return g


class CMAESLearner(Learner):
    """
    Implementation of CMA-ES
    """
    def __init__(self, coef_dim, optimizer, lr):
        super(CMAESLearner, self).__init__(coef_dim, optimizer, lr)
        
        self.N = coef_dim
        self.xmeanw = self.get_weight().reshape([self.N, 1])        # (N, 1)
        self.sigma = 1.0 
        self.minsigma = 1e-15

        # Parameter setting: selection
        self.lambda_ = int(4 + np.floor(3*np.log(self.N)))
        self.mu = int(np.floor(self.lambda_ / 2))
        self.arweights = np.log((self.lambda_+1)/2) - np.log(np.arange(1, self.mu+1).reshape([-1, 1]))   # (mu, 1)

        # parameter setting: adaptation
        self.cc = 4.0 / (self.N+4.0)
        self.ccov = 2.0 / (self.N + 2**0.5)**2
        self.cs = 4.0 / (self.N+4.0) 
        self.damp = 1.0/self.cs + 1.0

        # Initialize dynamic strategy parameters and constants
        self.B = np.eye(self.N)             # (N, N)
        self.D = np.eye(self.N)             # (N, N)
        self.BD = np.matmul(self.B, self.D) # (N, N)
        self.C = np.matmul(self.BD, self.BD.T)  # (N, N)
        self.pc = np.zeros([self.N, 1])     # (N, 1)
        self.ps = np.zeros([self.N, 1])     # (N, 1)
        self.cw = np.sum(self.arweights) / np.linalg.norm(self.arweights)
        self.chiN = self.N**0.5 * (1.0 - 1.0/(4.0*self.N) + 1.0/(21*self.N**2))

        # IMPORTANT
        self.set_weight(self.xmeanw.flatten())

    def generate_noises(self, _):
        """
        noises: (lambda_, N)
        TODO: the influence of lambda_
        """
        lambda_ = self.lambda_
        self.arz = np.random.randn(self.N, lambda_)             # (N, lambda)
        noises = self.sigma * np.matmul(self.BD, self.arz)      # (N, N) * (N, lambda) -> (N, lambda)
        self.arx = self.xmeanw + noises         # (N, lambda)
        return noises.T

    def learn(self, noises, noisy_rewards):
        """
        noises: (lambda_, N), ignored since we have self.arz and self.arx
        noisy_rewards: (lambda_,)
        """
        arfitness = noisy_rewards

        # Sort by fitness and compute weighted mean
        arindex = np.argsort(arfitness)[::-1]
        self.xmeanw = np.matmul(self.arx[:, arindex[:self.mu]], self.arweights) / np.sum(self.arweights)     # (N, mu) * (mu, 1) -> (N, 1)
        self.zmeanw = np.matmul(self.arz[:, arindex[:self.mu]], self.arweights) / np.sum(self.arweights)     # (N, mu) * (mu, 1) -> (N, 1)

        # Adapt covariance matrix
        self.pc = (1-self.cc) * self.pc + np.sqrt(self.cc*(2-self.cc)) * self.cw * np.matmul(self.BD, self.zmeanw)  # (N, 1)
        self.C = (1-self.ccov) * self.C + self.ccov * np.matmul(self.pc, self.pc.T)         # (N, N)

        # adapt sigma
        self.ps = (1-self.cs) * self.ps + np.sqrt(self.cs*(2-self.cs)) * self.cw * np.matmul(self.B, self.zmeanw)   # (N, 1)
        self.sigma = self.sigma * np.exp((np.linalg.norm(self.ps) - self.chiN) / self.chiN / self.damp)     # scalar

        # Update B and D from C, omit an if-condition
        self.C = np.triu(self.C) + np.triu(self.C, 1).T     # enforce symmetry
        self.d, self.B = np.linalg.eigh(self.C)      # (N,), (N, N) 
        self.D = np.diag(self.d)    # (N, N)
        # limit condition of C to 1e14 + 1
        if np.max(self.d) > 1e14 * np.min(self.d):
            tmp = np.max(self.d) / 1e14 - np.min(self.d)
            self.C = self.C + tmp * np.eye(self.N)
            self.D = self.D + tmp * np.eye(self.N)
        self.D = np.sqrt(self.D)
        self.BD = np.matmul(self.B, self.D)     # for speed up only

        # Adjust minimal step size
        if self.sigma * np.min(np.diag(self.D)) < self.minsigma:        # TODO: not complete
            self.sigma *= 1.4

        # IMPORTANT
        self.set_weight(self.xmeanw.flatten())       

        return None


class SCMLearner(Learner):
    """
    mu: top k ratio.
    weight_type: logrank or centerrank.
    use_C: whether to use Covariance Matrix Adaptation.
    cc, ccov: only used for use_C = 1.
    note:
        no mirror noise, it will hurt performance on feed.

    """
    def __init__(self, 
                coef_dim, 
                optimizer, 
                lr, 
                mu=1.0, 
                weight_type='centerrank', 
                use_C=None, 
                cc=None, 
                ccov=None):
        super(SCMLearner, self).__init__(coef_dim, optimizer, lr)

        self.mu = mu
        self.weight_type = weight_type
        self.use_C = use_C

        # parameter setting: adaptation
        self.cc = cc
        self.ccov = ccov

        # Initialize dynamic strategy parameters and constants
        self.B = np.eye(self.coef_dim)             # (coef_dim, coef_dim)
        self.D = np.eye(self.coef_dim)             # (coef_dim, coef_dim)
        self.BD = np.matmul(self.B, self.D) # (coef_dim, coef_dim)
        self.C = np.matmul(self.BD, self.BD.T)  # (coef_dim, coef_dim)
        self.pc = np.zeros([self.coef_dim, 1])     # (coef_dim, 1)

    def generate_noises(self, noise_num):
        """
        noises: (noise_num, coef_dim)
        """
        lambda_ = noise_num
        Z = np.random.randn(self.coef_dim, lambda_) # (coef_dim, lambda_)
        noises = np.matmul(self.BD, Z)              # (coef_dim, lambda_)
        return noises.T

    def get_rank_weight(self, lambda_, mu):
        """return (-1, 1)"""
        return np.log((lambda_+1)/2) - np.log(np.arange(1, mu+1).reshape([-1, 1]))

    def update_C(self, g):
        """
        Note: don't use svd for eigen decomposition
        """
        assert g.ndim == 2, (g.shape)
        g /= np.linalg.norm(g)    # scale to norm = 1
        self.pc = (1-self.cc) * self.pc + np.sqrt(self.cc*(2-self.cc)) * g      # (coef_dim, 1)
        assert self.pc.ndim == 2, (self.pc.shape)
        self.C = (1-self.ccov) * self.C + self.ccov * np.matmul(self.pc, self.pc.T)         # (coef_dim, coef_dim)

        ### Update B and D from C, omit an if-condition
        self.C = np.triu(self.C) + np.triu(self.C, 1).T     # enforce symmetry
        self.d, self.B = np.linalg.eigh(self.C)      # (coef_dim,), (coef_dim, coef_dim) 
        assert not np.iscomplex(self.d).any(), (self.C, self.d)
        assert (self.d >= 0).all(), (self.C, self.d)
        self.D = np.diag(self.d)    # (coef_dim, coef_dim)
        # limit condition of C to 1e14 + 1
        if np.max(self.d) > 1e14 * np.min(self.d):
            tmp = np.max(self.d) / 1e14 - np.min(self.d)
            self.C = self.C + tmp * np.eye(self.coef_dim)
            self.D = self.D + tmp * np.eye(self.coef_dim)
        self.D = np.sqrt(self.D)
        self.BD = np.matmul(self.B, self.D)     # for speed up only

    def calculate_gradient(self, noises, noisy_rewards):
        """
        noises: (noise_num, coef_dim),
        noisy_rewards: (noise_num,)
        return:
            g: (coef_dim)
        """
        assert noises.ndim == 2, (noises.shape)
        assert noisy_rewards.ndim == 1, (noisy_rewards.shape)

        lambda_ = len(noisy_rewards)
        if self.weight_type == 'centerrank':
            noisy_rewards = compute_centered_ranks(noisy_rewards)

        ### Calculate gradient
        BDZ = noises.T      # (coef_dim, lambda_)
        arweights = noisy_rewards.reshape([-1, 1])    # (lambda_, 1)
        arindex = np.argsort(arweights.flatten())[::-1]
        top_k = int(lambda_ * self.mu)
        assert top_k >= 1, (top_k)
        BDZ, arweights = BDZ[:, arindex[:top_k]], arweights[arindex[:top_k]]
        if self.weight_type == 'logrank':
            arweights = self.get_rank_weight(top_k * 2, top_k)

        g = np.matmul(BDZ, arweights) / np.sum(np.abs(arweights))   # (coef_dim, 1)
        return g.flatten()

    def calculate_gradient_new(self, noises, noisy_rewards, sigma):
        """
        noises: (noise_num, coef_dim),
        noisy_rewards: (noise_num,)
        sigma: float
        return:
            g: (coef_dim)
        """
        assert noises.ndim == 2, (noises.shape)
        assert noisy_rewards.ndim == 1, (noisy_rewards.shape)
        
        lambda_ = len(noisy_rewards)
        if self.weight_type == 'centerrank':
            noisy_rewards = compute_centered_ranks(noisy_rewards)

        ### Calculate gradient
        BDZ = noises.T      # (coef_dim, lambda_)
        arweights = noisy_rewards.reshape([-1, 1])    # (lambda_, 1)
        arindex = np.argsort(arweights.flatten())[::-1]
        top_k = int(lambda_ * self.mu)
        assert top_k >= 1, (top_k)
        BDZ, arweights = BDZ[:, arindex[:top_k]], arweights[arindex[:top_k]]
        if self.weight_type == 'logrank':
            arweights = self.get_rank_weight(top_k * 2, top_k)

        g = np.matmul(BDZ, arweights) / (sigma * np.shape(arweights)[0])
        print(np.shape(g)) 
        return g.flatten()

    def update(self, g):
        """
        g: (coef_dim,)
        return:
            g: (coef_dim,)
            update_ratio: float

        Will update coef and C.
        """
        assert g.ndim == 1, (g.shape)
        g = g / np.linalg.norm(g)

        ### Update
        l2_coeff = 0.005
        cur_coef = self.get_weight()
        new_coef, update_ratio = self.optimizer.update(cur_coef, -g.flatten() + l2_coeff * cur_coef)
        self.set_weight(new_coef)

        ### Adapt covariance matrix
        if self.use_C:
            self.update_C(g.reshape([-1, 1]))

        return g, update_ratio

    def update_new(self, g):
         """
         g: (coef_dim,)
         return:
            g: (coef_dim,)
            update_ratio: float

         Will update coef and C.
         """
         assert g.ndim == 1, (g.shape)
         #g = g / np.linalg.norm(g) 

         ### Update
         cur_coef = self.get_weight()
         new_coef = self.optimizer.update_new(cur_coef, -g.flatten())
         self.set_weight(new_coef)

         return g

    def learn(self, noises, noisy_rewards):
        """
        noises: (noise_num, coef_dim),
        noisy_rewards: (noise_num,)
        """
        g = self.calculate_gradient(noises, noisy_rewards)  # (coef_dim,)
        norm_g, update_ratio = self.update(g)
        return norm_g, update_ratio



