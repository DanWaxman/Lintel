from scipy import stats
from jaxtyping import Array, Float
from .gp_utils import MarkovianGP
import numpy as np
import scipy


class LINTEL:
    def __init__(
        self,
        N: int,
        alpha: float,
        weights: Float[Array, "M"],
        gps: list[MarkovianGP],
        L: int,
    ):
        """Creates an instance of the LINTEL algorithm, as described in [1].

        Args:
            N (int): number of consecutive outliers necessary to declare a regime switch
            tau (int): the window size for GPs
            alpha (float): the "forgetting factor"
            weights (Float[Array, &quot;M&quot;]): the initial weights to be used
            gps (list[GP]): list of candidate models
            L (int): period until the mean is updated

        References:
            [1] Waxman, D. & Djuric, P. M. (2024). Online Prediction of Switching Gaussian Process
                Time Series with Constant-Time Updates. Submitted.
        """
        self.N = N
        self.alpha = alpha
        self.weights = weights
        self.gps = gps
        self.M = len(gps)
        self.L = L

        # Potential Changepoint Bucket (PCB)
        self.tprime = []
        self.yprime = []

        # Number of points since $\mu(\cdot)$ has been changed
        self.t_since_mean_update = 0
        self.bin_total = 0

    def predict_and_update(
        self, ttp1: Float[Array, "1"], ytp1: Float[Array, "1"]
    ) -> tuple[Float[Array, "1"], Float[Array, "1"], int]:
        """Predicts and updates according to the INTEL algorithm.
        This corresponds to lines 4-29 of Algorithm 1 in [1].

        Args:
            ttp1 (Float[Array, "1"]): the time of the next observation
            ytp1 (Float[Array, "1"]): the output of the next observation

        Returns:
            Float[Array, "1"]: the predictive mean
            Float[Array, "1"]: the predictive variance
        """
        candidate_means = np.zeros(self.M)
        candidate_variances = np.ones(self.M)
        candidate_loglikes = np.zeros(self.M)
        outs = []

        outlier_flag: int = 0

        # Predict with each candidate model
        for m in range(self.M):
            out = self.gps[m].predict(ttp1)
            outs.append(out)
            candidate_means[m] = np.squeeze(out[0])
            candidate_variances[m] = np.squeeze(out[1])

            candidate_loglikes[m] = stats.norm.logpdf(
                ytp1, candidate_means[m], np.sqrt(candidate_variances[m])
            )

        # Calculate what from Eq. (15)
        whats = self.weights**self.alpha + 1e-4
        whats = whats / np.sum(whats)

        # Product of experts predictive distributions, Eqs. (21-22)
        phat = np.sum(whats / candidate_variances)
        sigmahat = 1 / phat
        mhat = np.sum(candidate_means * whats / candidate_variances) / phat

        # If the data is within 3sigma, accept it, otherwise reject it
        if (ytp1 < mhat + 3 * np.sqrt(sigmahat)) and (
            ytp1 > mhat - 3 * np.sqrt(sigmahat)
        ):
            for m in range(self.M):
                self.gps[m].update(ttp1, ytp1, *outs[m])

            # Reset PCB
            self.tprime = []
            self.yprime = []

            # Update means if time since mean update is more than L
            if self.t_since_mean_update >= self.L:
                for m in range(self.M):
                    self.gps[m].C = self.bin_total / self.L

                self.bin_total = 0.0
                self.t_since_mean_update = 0.0
            else:
                self.bin_total += ytp1
                self.t_since_mean_update += 1
        else:

            # Add to PCB and flag as an outlier
            self.tprime.append(ttp1)
            self.yprime.append(ytp1)
            outlier_flag = 1

            # If tprime has >= N elements, we've arrived at a changepoint
            if len(self.tprime) >= self.N:
                # Changepoint
                t = np.array(self.tprime)
                y = np.array(self.yprime)

                for m in range(self.M):
                    self.gps[m].reset_and_filter(t, y, np.mean(y))

                self.bin_total = 0.0
                self.t_since_mean_update = 0.0

        # Update weights with Eq. (16)
        log_w = np.log(whats)
        ell = -scipy.special.logsumexp(log_w + candidate_loglikes)
        log_w = log_w + (ell + candidate_loglikes)
        self.weights = np.exp(log_w)

        return mhat, sigmahat, outlier_flag
