"""
Pure NumPy Gaussian HMM (2-state, diagonal covariance)
=========================================================
A minimal from-scratch replacement for hmmlearn.hmm.GaussianHMM, used only
because hmmlearn ships compiled C++ extensions that can fail to import on
some cloud environments (undefined symbol / ABI mismatch errors). This
avoids that entirely -- pure Python + NumPy, nothing compiled.

Implements standard Baum-Welch (EM) fitting via the forward-backward
algorithm in log-space, plus Viterbi decoding for the most likely state
sequence. Interface mirrors what regime_model.py expects: after fit(),
.predict(X) and .predict_proba(X) work the same way hmmlearn's did.
"""

import numpy as np


def _log_gaussian_pdf(X, mean, var):
    """Log density of a diagonal Gaussian at each row of X. var: 1D array of variances per feature."""
    var = np.maximum(var, 1e-8)
    d = X.shape[1]
    diff = X - mean
    return -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum(diff ** 2 / var, axis=1))


def _logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0)
    out = np.log(np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)) + a_max_safe
    if axis is None:
        out = out.reshape(())  # scalar
    else:
        out = np.squeeze(out, axis=axis)
    return out


class SimpleGaussianHMM:
    def __init__(self, n_components=2, n_iter=200, random_state=42, tol=1e-4, covariance_type="diag"):
        # covariance_type accepted for interface compatibility with hmmlearn's
        # GaussianHMM call signature; this implementation is diagonal-only.
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        k = self.n_components

        # Init: split data into k chunks by sorted first-feature value for a stable, deterministic start
        order = np.argsort(X[:, 0])
        chunks = np.array_split(order, k)
        self.means_ = np.array([X[c].mean(axis=0) for c in chunks])
        self.covars_ = np.array([X[c].var(axis=0) + 1e-6 for c in chunks])
        self.startprob_ = np.full(k, 1.0 / k)
        # Sticky-ish initial transition matrix (regimes tend to persist)
        self.transmat_ = np.full((k, k), 0.05 / (k - 1) if k > 1 else 1.0)
        np.fill_diagonal(self.transmat_, 0.95)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            log_emit = np.column_stack([
                _log_gaussian_pdf(X, self.means_[j], self.covars_[j]) for j in range(k)
            ])  # (n, k)

            log_start = np.log(np.maximum(self.startprob_, 1e-12))
            log_trans = np.log(np.maximum(self.transmat_, 1e-12))

            # Forward (log-space)
            log_alpha = np.zeros((n, k))
            log_alpha[0] = log_start + log_emit[0]
            for t in range(1, n):
                log_alpha[t] = log_emit[t] + _logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)

            # Backward (log-space)
            log_beta = np.zeros((n, k))
            log_beta[-1] = 0.0
            for t in range(n - 2, -1, -1):
                log_beta[t] = _logsumexp(log_trans + log_emit[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

            log_ll = _logsumexp(log_alpha[-1])
            ll = float(log_ll)

            # Posteriors
            log_gamma = log_alpha + log_beta - log_ll
            gamma = np.exp(log_gamma)  # (n, k)

            # Pairwise posteriors (xi) for transition update
            xi_sum = np.zeros((k, k))
            for t in range(n - 1):
                log_xi_t = (log_alpha[t][:, None] + log_trans
                            + log_emit[t + 1][None, :] + log_beta[t + 1][None, :] - log_ll)
                xi_sum += np.exp(log_xi_t)

            # M-step
            self.startprob_ = gamma[0] / gamma[0].sum()
            denom = xi_sum.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1e-12
            self.transmat_ = xi_sum / denom

            gamma_sum = gamma.sum(axis=0)
            gamma_sum_safe = np.where(gamma_sum == 0, 1e-12, gamma_sum)
            for j in range(k):
                w = gamma[:, j][:, None]
                self.means_[j] = (w * X).sum(axis=0) / gamma_sum_safe[j]
                diff = X - self.means_[j]
                self.covars_[j] = (w * diff ** 2).sum(axis=0) / gamma_sum_safe[j] + 1e-6

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self._log_emit_cache_X = None
        return self

    def _log_emit(self, X):
        k = self.n_components
        return np.column_stack([_log_gaussian_pdf(X, self.means_[j], self.covars_[j]) for j in range(k)])

    def predict_proba(self, X):
        """Smoothed state posteriors (forward-backward gamma), same semantics as hmmlearn."""
        n, k = X.shape[0], self.n_components
        log_emit = self._log_emit(X)
        log_start = np.log(np.maximum(self.startprob_, 1e-12))
        log_trans = np.log(np.maximum(self.transmat_, 1e-12))

        log_alpha = np.zeros((n, k))
        log_alpha[0] = log_start + log_emit[0]
        for t in range(1, n):
            log_alpha[t] = log_emit[t] + _logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)

        log_beta = np.zeros((n, k))
        log_beta[-1] = 0.0
        for t in range(n - 2, -1, -1):
            log_beta[t] = _logsumexp(log_trans + log_emit[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

        log_ll = _logsumexp(log_alpha[-1])
        log_gamma = log_alpha + log_beta - log_ll
        return np.exp(log_gamma)

    def predict(self, X):
        """Most likely state sequence via Viterbi decoding."""
        n, k = X.shape[0], self.n_components
        log_emit = self._log_emit(X)
        log_start = np.log(np.maximum(self.startprob_, 1e-12))
        log_trans = np.log(np.maximum(self.transmat_, 1e-12))

        log_delta = np.zeros((n, k))
        psi = np.zeros((n, k), dtype=int)
        log_delta[0] = log_start + log_emit[0]
        for t in range(1, n):
            scores = log_delta[t - 1][:, None] + log_trans  # (k,k): from,to
            psi[t] = np.argmax(scores, axis=0)
            log_delta[t] = np.max(scores, axis=0) + log_emit[t]

        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states
