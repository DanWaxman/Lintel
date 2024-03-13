import jax
import jax.numpy as jnp
import math
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax import math as tfp_math
import bayesnewton
import objax
from bayesnewton.utils import (
    softplus,
    softplus_inv,
    rotation_matrix,
)
import copy


def add_to_diagonal(K: Float[Array, "N M"], constant: float) -> Float[Array, "N M"]:
    return K.at[jnp.diag_indices(K.shape[0])].add(constant)


class ObjaxModuleWithDeepCopy(objax.Module):
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, objax.BaseVar):
                # TODO: There are more correct ways to do this, but it works for now
                v = v.__class__(v.value)
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class GP(ObjaxModuleWithDeepCopy):
    def __init__(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N"],
        C: Float,
        sigma_n: Float,
        kernel: bayesnewton.kernels.Kernel,
    ):
        self.C = C
        self.kernel = kernel
        self.transformed_sigma_n = objax.TrainVar(softplus_inv(jnp.asarray(sigma_n)))
        self.k = objax.Jit(self.kernel.K, vc=self.vars())
        self.cho_factor = jax.jit(jax.scipy.linalg.cholesky)
        self.update_training_set(X, y)

    @property
    def sigma_n(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_sigma_n.value), 1e2)

    def add_predictive_var(self, mat: Float[Array, "..."]) -> Float[Array, "..."]:
        """Adds the predictive variance to the diagonal of the matrix

        Args:
            mat (Float[Array, "..."]): matrix to add to diagonal of

        Returns:
            Float[Array, "..."]: matrix with sigma_n^2 added to the diagonal
        """
        return add_to_diagonal(
            mat,
            constant=self.sigma_n**2,
        )

    def predict(
        self, X_star: Float[Array, "M D"]
    ) -> tuple[Float[Array, "M"], Float[Array, "M M"]]:
        """Predicts the posterior predictive of X_star, with noise

        Returns:
            Float[Array, "M"]: the predictive mean
            Float[Array, "M M"]]: the predictive covariance
        """
        K_ox = self.k(self.X, X_star)
        K_xx = self.add_predictive_var(self.k(X_star, X_star))

        m = self.C + K_ox.T @ jax.scipy.linalg.cho_solve(self.A_cho, self.y - self.C)
        sigma2 = K_xx - K_ox.T @ jax.scipy.linalg.cho_solve(self.A_cho, K_ox)

        return m, sigma2

    def update_training_set(self, X: Float[Array, "N D"], y: Float[Array, "N"]):
        """Updates the training data, including precomputing the Cholesky factor
        of the new kernel matrix.

        Args:
            X (Float[Array, &quot;N D&quot;]): new training inputs
            y (Float[Array, &quot;N&quot;]): new training outputs
        """
        self.X = X
        self.y = y

        self.K_oo = self.add_predictive_var(self.k(X, X))
        self.A_cho = (
            self.cho_factor(self.K_oo),
            False,
        )

    def mnll(self, X: Float[Array, "N D"], y: Float[Array, "N"]) -> float:
        """Calculates the negative marginal likelihood of the data

        Args:
            X (Float[Array, &quot;N D&quot;]): inputs
            y (Float[Array, &quot;N&quot;]): outputs

        Returns:
            float: the negative marginal likelihood
        """
        K_oo = self.add_predictive_var(self.k(X, X))
        A_cho = jax.scipy.linalg.cho_factor(K_oo)

        return -(
            -(y - self.C).T @ jax.scipy.linalg.cho_solve(A_cho, self.y - self.C) / 2
            - jnp.sum(jnp.log(jnp.diag(A_cho[0])))
            - X.shape[0] / 2 * jnp.log(2 * jnp.pi)
        )

    def maximize_evidence(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N"],
        lr: float = 1e-2,
        iters: int = 200,
        verbose: bool = False,
    ):
        """Computes the type-II MLE hyperparameters and updates them in-place.
        Uses Adam for optimizing the MLL.

        Args:
            X (Float[Array, &quot;N D&quot;]): inputs
            y (Float[Array, &quot;N&quot;]): outputs
            lr (float, optional): learning rate of the Adam optimizer. Defaults to 1e-2.
            iters (int, optional): number of iterations to optimize for. Defaults to 200.
            verbose (bool, optional): if `True`, will print out results. Defaults to False.
        """
        opt = objax.optimizer.Adam(self.vars())
        gv = objax.GradValues(self.mnll, self.vars())

        @objax.Function.with_vars(self.vars() + gv.vars() + opt.vars())
        def train_op():
            df, f = gv(X, y)
            opt(lr, df)
            return f

        train_op = objax.Jit(train_op)

        for iter_idx in range(iters):
            f_value = train_op()
            if (iter_idx % 100 == 0 or iter_idx == iters - 1) and verbose:
                print(iter_idx, f_value)

        self.update_training_set(self.X, self.y)

        return f_value


class MarkovianGP(ObjaxModuleWithDeepCopy):
    def __init__(self, C, sigma_n, kernel):
        self.C = C
        self.transformed_sigma_n = objax.TrainVar(softplus_inv(jnp.asarray(sigma_n)))
        self.kernel = kernel
        self.t_last = jnp.inf

        self.n_dim = self.kernel.stationary_covariance().shape[0]
        self.m = jnp.zeros((self.n_dim, 1))
        self.Pinf = self.kernel.stationary_covariance()
        self.H = self.kernel.measurement_model()
        self.state_transition_func = objax.Jit(
            self.kernel.state_transition, vc=self.vars()
        )

    @property
    def sigma_n(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_sigma_n.value), 1e2)

    def get_Q(self, A):
        return self.Pinf - A @ self.Pinf @ A.T

    def get_A(self, dt):
        return self.state_transition_func(dt)

    def predict(self, t_star):
        dt = max(t_star - self.t_last, 0)
        A_n = self.get_A(dt)
        m_evolved = A_n @ self.m
        P_evolved = A_n @ self.P @ A_n.T + self.get_Q(A_n)

        m = self.H @ m_evolved + self.C
        sigma2 = self.H @ P_evolved @ self.H.T + self.sigma_n**2

        return m, sigma2, m_evolved, P_evolved

    def update(self, t_star, y_star, m, sigma2, m_evolved, P_evolved):
        k = jax.scipy.linalg.solve(sigma2, self.H @ P_evolved, assume_a="pos").T
        self.m = m_evolved + k @ (y_star - m)
        self.P = P_evolved - k @ self.H @ P_evolved
        self.t_last = t_star

    def reset_and_filter(self, t, y, mean):
        self.C = mean
        self.t_last = jnp.inf

        self.m = jnp.zeros((self.n_dim, 1))
        self.P = self.Pinf
        N = t.shape[0]

        for n in range(N):
            o = self.predict(t[n])
            self.update(t[n], y[n], *o)


class SubbandMatern32(bayesnewton.kernels.StationaryKernel):
    """
    Subband Matern-3/2 kernel in SDE form (product of Cosine and Matern-3/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the Matern3/2 and cosine components:
    letting λ = √3 / l
    F      = F_mat3/2 ⊕ F_cos  =  ( 0     -ω     1     0
                                    ω      0     0     1
                                   -λ²     0    -2λ   -ω
                                    0     -λ²    ω    -2λ )
    L      = L_mat3/2 ⊗ I      =  ( 0      0
                                    0      0
                                    1      0
                                    0      1 )
    Qc     = I ⊗ Qc_mat3/2     =  ( 4λ³σ²  0
                                    0      4λ³σ² )
    H      = H_mat3/2 ⊗ H_cos  =  ( 1      0     0      0 )
    Pinf   = Pinf_mat3/2 ⊗ I   =  ( σ²     0     0      0
                                    0      σ²    0      0
                                    0      0     3σ²/l² 0
                                    0      0     0      3σ²/l²)
    and the discrete-time transition matrix is (for step size Δt),
    R = ( cos(ωΔt)   -sin(ωΔt)
          sin(ωΔt)    cos(ωΔt) )
    A = exp(-Δt/l) ( (1+Δtλ)R   ΔtR
                     -Δtλ²R    (1-Δtλ)R )
    """

    def __init__(
        self, variance=1.0, lengthscale=1.0, radial_frequency=1.0, fix_variance=False
    ):
        self.transformed_radial_frequency = objax.TrainVar(
            jnp.array(softplus_inv(radial_frequency))
        )
        super().__init__(
            variance=variance, lengthscale=lengthscale, fix_variance=fix_variance
        )
        self.name = "Subband Matern-3/2"
        self.state_dim = 4

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    @property
    def radial_frequency(self):
        return softplus(self.transformed_radial_frequency.value)

    def K_r(self, r):
        k_cos = jnp.cos(self.radial_frequency * r * self.lengthscale)

        sqrt3 = jnp.sqrt(3.0)
        k_mat = (1.0 + sqrt3 * r) * jnp.exp(-sqrt3 * r)

        return self.variance * k_mat * k_cos

    def kernel_to_state_space(self, R=None):
        lam = 3.0**0.5 / self.lengthscale
        F_mat = jnp.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        L_mat = jnp.array([[0], [1]])
        Qc_mat = jnp.array([[12.0 * 3.0**0.5 / self.lengthscale**3.0 * self.variance]])
        H_mat = jnp.array([[1.0, 0.0]])
        Pinf_mat = jnp.array(
            [[self.variance, 0.0], [0.0, 3.0 * self.variance / self.lengthscale**2.0]]
        )
        F_cos = jnp.array([[0.0, -self.radial_frequency], [self.radial_frequency, 0.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        # F = (0   -ω   1   0
        #      ω    0   0   1
        #      -λ²  0  -2λ -ω
        #      0   -λ²  ω  -2λ)
        F = jnp.kron(F_mat, jnp.eye(2)) + jnp.kron(jnp.eye(2), F_cos)
        L = jnp.kron(L_mat, jnp.eye(2))
        Qc = jnp.kron(jnp.eye(2), Qc_mat)
        H = jnp.kron(H_mat, H_cos)
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        Pinf_mat = jnp.array(
            [[self.variance, 0.0], [0.0, 3.0 * self.variance / self.lengthscale**2.0]]
        )
        Pinf = jnp.kron(Pinf_mat, jnp.eye(2))
        return Pinf

    def measurement_model(self):
        H_mat = jnp.array([[1.0, 0.0]])
        H_cos = jnp.array([[1.0, 0.0]])
        H = jnp.kron(H_mat, H_cos)
        return H

    def state_transition(self, dt):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :return: state transition matrix A [4, 4]
        """
        lam = jnp.sqrt(3.0) / self.lengthscale
        R = rotation_matrix(dt, self.radial_frequency)
        A = jnp.exp(-dt * lam) * jnp.block(
            [[(1.0 + dt * lam) * R, dt * R], [-dt * lam**2 * R, (1.0 - dt * lam) * R]]
        )
        return A

    def feedback_matrix(self):
        lam = 3.0**0.5 / self.lengthscale
        F_mat = jnp.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        F_cos = jnp.array([[0.0, -self.radial_frequency], [self.radial_frequency, 0.0]])
        # F = (0   -ω   1   0
        #      ω    0   0   1
        #      -λ²  0  -2λ -ω
        #      0   -λ²  ω  -2λ)
        F = jnp.kron(F_mat, jnp.eye(2)) + jnp.kron(jnp.eye(2), F_cos)
        return F
