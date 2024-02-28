import jax
import jax.numpy as jnp
import math
from tensorflow_probability.substrates.jax import math as tfp_math
from jaxtyping import Array, Float
import objax


# Some utils taken from https://github.com/jejjohnson/gp_model_zoo


def cross_covariance(
    func: callable, x: Float[Array, "N D"], y: Float[Array, "M D"]
) -> Float[Array, "N M"]:
    """distance matrix"""
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)


def sqeuclidean_distance(x: Float[Array, "D"], y: Float[Array, "D"]) -> float:
    return jnp.sum((x - y) ** 2)


def euclidean_distance(x: Float[Array, "D"], y: Float[Array, "D"]) -> float:
    return jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-16)


def rbf_kernel(
    X: Float[Array, "N D"], Y: Float[Array, "M D"], variance: float, lengthscale: float
) -> Float[Array, "N M"]:
    deltaXsq = cross_covariance(sqeuclidean_distance, X / lengthscale, Y / lengthscale)

    K = variance * jnp.exp(-0.5 * deltaXsq)
    return K


def mat52_kernel(
    X: Float[Array, "N D"], Y: Float[Array, "M D"], variance: float, lengthscale: float
) -> Float[Array, "N M"]:
    mean = jnp.mean(X)

    distance = cross_covariance(
        euclidean_distance, (X - mean) / lengthscale, (Y - mean) / lengthscale
    )
    const_component = math.sqrt(5) * distance + 1 + 5.0 / 3.0 * distance**2
    exp_component = jnp.exp(-math.sqrt(5) * distance)

    return variance * const_component * exp_component


def add_to_diagonal(K: Float[Array, "N M"], constant: float) -> Float[Array, "N M"]:
    return K.at[jnp.diag_indices(K.shape[0])].add(constant)


def softplus(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Alias for `jax.nn.softplus`.

    Args:
        x (Float[Array, "..."]): x

    Returns:
        Float[Array, "..."]: softplus(x)
    """
    return jax.nn.softplus(x)


def softplus_inv(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Alias for `tfp_math.softplus_inverse`.

    Args:
        x (Float[Array, "..."]): x

    Returns:
        Float[Array, "..."]: softplus^{-1}(x)
    """
    return tfp_math.softplus_inverse(x)


class GP(objax.Module):
    def __init__(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N"],
        lengthscale: float,
        sigma_f: float,
        sigma_n: float,
        C: float,
    ):
        """Creates an instance of an Objax-implementation of kernel-based GPs, using the Matern-5/2
        kernel.

        Args:
            X (Float[Array, &quot;N D&quot;]): Training inputs
            y (Float[Array, &quot;N&quot;]): Training outputs
            lengthscale (float): lengthscale of kernel
            sigma_f (float): process scale of kernel
            sigma_n (float): additive noise scale
            C (float): value of a constant mean function in the prior
        """
        self.transformed_lengthscale = objax.TrainVar(
            softplus_inv(jnp.asarray(lengthscale))
        )
        self.transformed_sigma_f = objax.TrainVar(softplus_inv(jnp.array(sigma_f)))
        self.transformed_sigma_n = objax.TrainVar(softplus_inv(jnp.asarray(sigma_n)))
        self.C = C

        self.update_training_set(X, y)

    @property
    def lengthscale(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_lengthscale.value), 1e6)

    @property
    def sigma_f(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_sigma_f.value), 1e2)

    @property
    def sigma_n(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_sigma_n.value), 1e2)

    def k(
        self, X1: Float[Array, "N D"], X2: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        return mat52_kernel(
            X1, X2, variance=self.sigma_f**2, lengthscale=self.lengthscale
        )

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
        self.A_cho = jax.scipy.linalg.cho_factor(self.K_oo)

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


class MarkovianGP(objax.Module):
    def __init__(self, lengthscale, sigma_f, sigma_n, C):
        self.transformed_lengthscale = objax.TrainVar(
            softplus_inv(jnp.asarray(lengthscale))
        )
        self.transformed_sigma_f = objax.TrainVar(softplus_inv(jnp.array(sigma_f)))
        self.transformed_sigma_n = objax.TrainVar(softplus_inv(jnp.asarray(sigma_n)))
        self.C = C
        self.t_last = jnp.inf

        self.m = jnp.zeros((3, 1))
        self.P = self.Pinf

    @property
    def lengthscale(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_lengthscale.value), 1e6)

    @property
    def sigma_f(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_sigma_f.value), 1e2)

    @property
    def sigma_n(self):
        return 1e-6 + jnp.minimum(softplus(self.transformed_sigma_n.value), 1e2)

    @property
    def H(self):
        return jnp.array([[1, 0, 0]])

    @property
    def Pinf(self):
        kappa = 5.0 / 3.0 * self.sigma_f**2 / self.lengthscale**2.0

        return jnp.array(
            [
                [self.sigma_f**2, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, 25.0 * self.sigma_f**2 / self.lengthscale**4.0],
            ]
        )

    def get_Q(self, A):
        return self.Pinf - A @ self.Pinf @ A.T

    def get_A(self, dt):
        lam = jnp.sqrt(5.0) / self.lengthscale
        dtlam = dt * lam
        A = jnp.exp(-dtlam) * (
            dt
            * jnp.array(
                [
                    [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                    [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                    [
                        lam**3 * (0.5 * dtlam - 1.0),
                        lam**2 * (dtlam - 3),
                        lam * (0.5 * dtlam - 2.0),
                    ],
                ]
            )
            + jnp.eye(3)
        )
        return A

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

        self.m = jnp.zeros((3, 1))
        self.P = self.Pinf
        N = t.shape[0]

        for n in range(N):
            o = self.predict(t[n])
            self.update(t[n], y[n], *o)
