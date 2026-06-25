import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass(frozen=True)
class Optimizer:
    """Base class for all pypomp optimizers.

    Parameters
    ----------
    clip_norm : float, optional
        Maximum norm threshold for gradient clipping. Gradients are clipped to
        [-clip_norm, clip_norm] if provided. Defaults to None (no clipping).
    scale : bool, default False
        Whether to normalize the update search direction to unit length
        before applying the learning rate.
    ls : bool, default False
        Whether to enable the Armijo backtracking line search algorithm to
        determine optimal step size.
    c : float, default 0.1
        The Armijo condition constant for line search, controlling how much
        the objective must decrease to accept a step size. Only used when ls=True.
    max_ls_itn : int, default 10
        Maximum number of backtracking iterations per line search step.
        Only used when ls=True.
    """

    clip_norm: Optional[float] = None
    scale: bool = False
    ls: bool = False
    c: float = 0.1
    max_ls_itn: int = 10

    def init_state(self, theta: jax.Array) -> tuple:
        """Initialize the optimizer state buffers.

        Parameters
        ----------
        theta : jax.Array
            The parameters array. Can be 1D for standard parameters
            or ND for chunked unit parameters.

        Returns
        -------
        tuple
            The initial optimizer state buffers as a tuple of JAX arrays.
        """
        raise NotImplementedError

    def step(
        self,
        grad: jax.Array,
        state: tuple,
        step_num: int | jax.Array,
        compute_hessian_fn: Optional[Callable[[], jax.Array]] = None,
        eta_i: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, tuple]:
        """Compute the parameter update direction and update the optimizer state.

        Parameters
        ----------
        grad : jax.Array
            The gradient of the objective function with respect to the parameters.
        state : tuple
            The current optimizer state buffers.
        step_num : int or jax.Array
            The current iteration or step index (0-indexed).
        compute_hessian_fn : callable, optional
            A zero-argument callable that computes the model Hessian when invoked.
            Only called if the optimizer requires the Hessian (e.g. Newton methods).
        eta_i : jax.Array, optional
            The current step size/learning rate at this iteration.
            Only used if the optimizer requires it (e.g. BFGS).

        Returns
        -------
        direction : jax.Array
            The parameter update search direction.
        new_state : tuple
            The updated optimizer state buffers.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        from dataclasses import fields

        field_strs = []
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, float):
                field_strs.append(f"{f.name}={val:.4g}")
            else:
                field_strs.append(f"{f.name}={val}")
        return f"{self.__class__.__name__}({', '.join(field_strs)})"


@dataclass(frozen=True)
class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def init_state(self, theta: jax.Array) -> tuple:
        return ()

    def step(
        self,
        grad: jax.Array,
        state: tuple,
        step_num: int | jax.Array,
        compute_hessian_fn: Optional[Callable[[], jax.Array]] = None,
        eta_i: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, tuple]:
        return -grad, ()


@dataclass(frozen=True)
class Adam(Optimizer):
    """Adam optimizer.

    Parameters
    ----------
    beta1 : float, default 0.9
        The exponential decay rate for the first moment estimates (momentum).
    beta2 : float, default 0.999
        The exponential decay rate for the second moment estimates (variance).
    epsilon : float, default 1e-8
        A small constant for numerical stability.
    """

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def init_state(self, theta: jax.Array) -> tuple:
        return jnp.zeros_like(theta), jnp.zeros_like(theta)

    def step(
        self,
        grad: jax.Array,
        state: tuple,
        step_num: int | jax.Array,
        compute_hessian_fn: Optional[Callable[[], jax.Array]] = None,
        eta_i: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, tuple]:
        m, v = state
        m_new = self.beta1 * m + (1 - self.beta1) * grad
        v_new = self.beta2 * v + (1 - self.beta2) * (grad**2)
        m_hat = m_new / (1 - self.beta1 ** (step_num + 1))
        v_hat = v_new / (1 - self.beta2 ** (step_num + 1))
        direction = -m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        return direction, (m_new, v_new)


@dataclass(frozen=True)
class FullMatrixAdam(Optimizer):
    """Full-Matrix Adam optimizer.

    Parameters
    ----------
    beta1 : float, default 0.9
        The exponential decay rate for the first moment estimates.
    beta2 : float, default 0.999
        The exponential decay rate for the second moment estimates.
    epsilon : float, default 1e-4
        A small constant for numerical stability.
    """

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-4

    def init_state(self, theta: jax.Array) -> tuple:
        m = jnp.zeros_like(theta)
        if theta.ndim == 1:
            v = jnp.zeros((theta.shape[-1], theta.shape[-1]))
        else:
            v = jnp.zeros(theta.shape + (theta.shape[-1],))
        return m, v

    def _step_single(self, grad, m, v, step_num):
        m_new = self.beta1 * m + (1 - self.beta1) * grad
        m_hat = m_new / (1 - self.beta1 ** (step_num + 1))

        F_t = self.beta2 * v + (1 - self.beta2) * jnp.outer(grad, grad)
        F_hat = F_t / (1 - self.beta2 ** (step_num + 1))

        eigenvalues, eigenvectors = jnp.linalg.eigh(F_hat)
        inv_sqrt_evals = 1.0 / jnp.sqrt(jnp.maximum(eigenvalues, 0.0) + self.epsilon)
        F_inv_sqrt = eigenvectors @ jnp.diag(inv_sqrt_evals) @ eigenvectors.T

        direction = -F_inv_sqrt @ m_hat
        return direction, m_new, F_t

    def step(
        self,
        grad: jax.Array,
        state: tuple,
        step_num: int | jax.Array,
        compute_hessian_fn: Optional[Callable[[], jax.Array]] = None,
        eta_i: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, tuple]:
        m, v = state
        if grad.ndim == 1:
            direction, m_new, v_new = self._step_single(grad, m, v, step_num)
            return direction, (m_new, v_new)
        else:
            direction, m_new, v_new = jax.vmap(
                self._step_single, in_axes=(0, 0, 0, None)
            )(grad, m, v, step_num)
            return direction, (m_new, v_new)


@dataclass(frozen=True)
class BFGS(Optimizer):
    """Quasi-Newton BFGS optimizer."""

    def init_state(self, theta: jax.Array) -> tuple:
        return jnp.eye(theta.shape[-1]), jnp.zeros_like(theta)

    def step(
        self,
        grad: jax.Array,
        state: tuple,
        step_num: int | jax.Array,
        compute_hessian_fn: Optional[Callable[[], jax.Array]] = None,
        eta_i: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, tuple]:
        hess, prev_grad = state

        if eta_i is None:
            raise ValueError("BFGS optimizer requires eta_i")

        def bfgs_true(_):
            prev_direction = jax.lax.cond(
                step_num > 0,
                lambda __: -prev_grad,
                lambda __: -grad,
                operand=None,
            )
            s_k = jnp.mean(eta_i) * prev_direction
            y_k = grad - prev_grad
            rho_k = jnp.reciprocal(jnp.dot(y_k, s_k))

            Hy = hess @ y_k
            yHy = jnp.dot(y_k, Hy)
            term1 = rho_k * jnp.outer(s_k, Hy)
            term2 = rho_k * jnp.outer(Hy, s_k)
            term3 = rho_k * (rho_k * yHy + 1.0) * jnp.outer(s_k, s_k)

            new_hess = hess - term1 - term2 + term3
            new_hess = jnp.where(jnp.isfinite(rho_k), new_hess, hess)
            new_direction = -new_hess @ grad
            return new_hess, new_direction

        def bfgs_false(_):
            return hess, -grad

        new_hess, direction = jax.lax.cond(
            step_num > 1,
            bfgs_true,
            bfgs_false,
            operand=None,
        )
        return direction, (new_hess, grad)


@dataclass(frozen=True)
class Newton(Optimizer):
    """Classic Second-Order Newton-Raphson optimizer."""

    def init_state(self, theta: jax.Array) -> tuple:
        return ()

    def step(
        self,
        grad: jax.Array,
        state: tuple,
        step_num: int | jax.Array,
        compute_hessian_fn: Optional[Callable[[], jax.Array]] = None,
        eta_i: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, tuple]:
        if compute_hessian_fn is None:
            raise ValueError("Newton optimizer requires compute_hessian_fn")
        hess = compute_hessian_fn()
        direction = -jnp.linalg.pinv(hess, hermitian=True) @ grad
        return direction, ()


@dataclass(frozen=True)
class WeightedNewton(Optimizer):
    """Weighted Newton optimizer with decaying history."""

    def init_state(self, theta: jax.Array) -> tuple:
        return (jnp.eye(theta.shape[-1]),)

    def step(
        self,
        grad: jax.Array,
        state: tuple,
        step_num: int | jax.Array,
        compute_hessian_fn: Optional[Callable[[], jax.Array]] = None,
        eta_i: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, tuple]:
        if compute_hessian_fn is None:
            raise ValueError("WeightedNewton optimizer requires compute_hessian_fn")
        (prev_hess,) = state
        hess = compute_hessian_fn()

        def dir_weighted(_):
            i_f = jnp.asarray(step_num).astype(grad.dtype)
            wt = (i_f ** jnp.log(i_f)) / ((i_f + 1) ** jnp.log(i_f + 1))
            weighted_hess = wt * prev_hess + (1 - wt) * hess
            return -jnp.linalg.pinv(weighted_hess, hermitian=True) @ grad

        direction = jax.lax.cond(
            step_num == 0,
            lambda _: -jnp.linalg.pinv(hess, hermitian=True) @ grad,
            dir_weighted,
            None,
        )
        return direction, (hess,)
