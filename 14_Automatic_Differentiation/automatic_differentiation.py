"""
Automatic Differentiation (AD)
==============================
Compute EXACT derivatives of arbitrary functions — not numerical
(finite differences) and not symbolic (CAS). AD exploits the chain
rule on the computational graph of a function.

**Two Modes:**

1. Forward Mode (tangent linear):
   - Propagates derivatives alongside function evaluation
   - Uses "dual numbers": x + εẋ, where ε² = 0
   - Cost: one forward pass per input variable
   - Best when: n_inputs << n_outputs

2. Reverse Mode (adjoint / backpropagation):
   - Forward pass: record computational graph (tape)
   - Backward pass: propagate adjoints (∂output/∂intermediate)
   - Cost: one forward + one backward per output variable
   - Best when: n_outputs << n_inputs (e.g., scalar loss → many params)
   - This is what PyTorch/TensorFlow use

**Why AD matters in physics:**
- Compute forces from potentials: F = -∇V
- Sensitivity analysis of simulations
- Optimization (variational methods)
- Hamiltonian mechanics: ∂H/∂q, ∂H/∂p
- Machine learning potentials

**Dual Numbers (Forward Mode):**
    f(a + εb) = f(a) + ε·f'(a)·b

    (a + εa') + (b + εb') = (a+b) + ε(a'+b')
    (a + εa') × (b + εb') = ab + ε(ab' + a'b)

Where to start:
━━━━━━━━━━━━━━
1. Start with forward mode dual numbers — it's simple and elegant
2. Implement basic operations and test on known derivatives
3. Then try reverse mode for multi-variable functions
4. Compare: AD vs finite differences vs analytical
Prerequisite: basic calculus, chain rule
"""

import numpy as np


# ============================================================
# Forward Mode: Dual Numbers
# ============================================================

class Dual:
    """
    Dual number for forward-mode automatic differentiation.
    
    A dual number is a + εb where ε² = 0.
    - 'val' stores the function value
    - 'der' stores the derivative
    
    Arithmetic rules follow from ε² = 0.
    """
    
    def __init__(self, val, der=0.0):
        self.val = float(val)
        self.der = float(der)
    
    def __repr__(self):
        return f"Dual({self.val}, {self.der})"
    
    # --- Arithmetic ---
    
    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.der + other.der)
        return Dual(self.val + other, self.der)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.der - other.der)
        return Dual(self.val - other, self.der)
    
    def __rsub__(self, other):
        return Dual(other - self.val, -self.der)
    
    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val,
                       self.val * other.der + self.der * other.val)
        return Dual(self.val * other, self.der * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val / other.val,
                       (self.der * other.val - self.val * other.der) / other.val**2)
        return Dual(self.val / other, self.der / other)
    
    def __rtruediv__(self, other):
        return Dual(other / self.val,
                   -other * self.der / self.val**2)
    
    def __pow__(self, n):
        if isinstance(n, Dual):
            # x^y where both are dual: use exp(y*ln(x))
            return exp(n * ln(self))
        return Dual(self.val**n, n * self.val**(n-1) * self.der)
    
    def __rpow__(self, base):
        # base^self
        return Dual(base**self.val,
                   base**self.val * np.log(base) * self.der)
    
    def __neg__(self):
        return Dual(-self.val, -self.der)
    
    def __abs__(self):
        if self.val >= 0:
            return Dual(self.val, self.der)
        return Dual(-self.val, -self.der)
    
    # --- Comparison (on values only) ---
    def __lt__(self, other):
        return self.val < (other.val if isinstance(other, Dual) else other)
    
    def __gt__(self, other):
        return self.val > (other.val if isinstance(other, Dual) else other)


# --- Dual-aware math functions ---

def sin(x):
    if isinstance(x, Dual):
        return Dual(np.sin(x.val), np.cos(x.val) * x.der)
    return np.sin(x)

def cos(x):
    if isinstance(x, Dual):
        return Dual(np.cos(x.val), -np.sin(x.val) * x.der)
    return np.cos(x)

def tan(x):
    if isinstance(x, Dual):
        return Dual(np.tan(x.val), x.der / np.cos(x.val)**2)
    return np.tan(x)

def exp(x):
    if isinstance(x, Dual):
        e = np.exp(x.val)
        return Dual(e, e * x.der)
    return np.exp(x)

def ln(x):
    if isinstance(x, Dual):
        return Dual(np.log(x.val), x.der / x.val)
    return np.log(x)

def log(x):
    return ln(x)

def sqrt(x):
    if isinstance(x, Dual):
        s = np.sqrt(x.val)
        return Dual(s, x.der / (2 * s))
    return np.sqrt(x)

def tanh(x):
    if isinstance(x, Dual):
        t = np.tanh(x.val)
        return Dual(t, (1 - t**2) * x.der)
    return np.tanh(x)

def sinh(x):
    if isinstance(x, Dual):
        return Dual(np.sinh(x.val), np.cosh(x.val) * x.der)
    return np.sinh(x)

def cosh(x):
    if isinstance(x, Dual):
        return Dual(np.cosh(x.val), np.sinh(x.val) * x.der)
    return np.cosh(x)


def forward_diff(f, x):
    """
    Compute f(x) and f'(x) using forward-mode AD.
    
    Parameters
    ----------
    f : callable
        Function using Dual-compatible operations
    x : float
        Point at which to evaluate
    
    Returns
    -------
    val : float
        Function value f(x)
    deriv : float
        Derivative f'(x)
    """
    result = f(Dual(x, 1.0))
    return result.val, result.der


def gradient_forward(f, x):
    """
    Compute gradient of f: R^n → R using forward mode.
    
    Requires n forward passes (one per input dimension).
    """
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        # Seed the i-th component
        x_dual = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        result = f(x_dual)
        grad[i] = result.der
    
    return grad


def jacobian_forward(f, x, m):
    """
    Compute Jacobian of f: R^n → R^m using forward mode.
    
    Parameters
    ----------
    f : callable
        Vector-valued function
    x : array
        Input point
    m : int
        Output dimension
    """
    n = len(x)
    J = np.zeros((m, n))
    
    for i in range(n):
        x_dual = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        result = f(x_dual)
        for j in range(m):
            J[j, i] = result[j].der
    
    return J


# ============================================================
# Reverse Mode: Computational Graph (Tape)
# ============================================================

class Var:
    """
    Variable node for reverse-mode automatic differentiation.
    
    Builds a computational graph (tape) during forward pass,
    then traverses it backward to compute gradients.
    """
    
    def __init__(self, val, children=(), op=''):
        self.val = float(val)
        self.grad = 0.0
        self._children = children  # (child_var, local_gradient) pairs
        self._op = op
    
    def __repr__(self):
        return f"Var({self.val:.6f})"
    
    def backward(self):
        """Compute gradients via reverse accumulation."""
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child, _ in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Backward pass
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in v._children:
                child.grad += v.grad * local_grad
    
    def zero_grad(self):
        """Reset gradient."""
        self.grad = 0.0
    
    # --- Arithmetic ---
    
    def __add__(self, other):
        if not isinstance(other, Var):
            other = Var(other)
        return Var(self.val + other.val,
                  [(self, 1.0), (other, 1.0)], '+')
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if not isinstance(other, Var):
            other = Var(other)
        return Var(self.val - other.val,
                  [(self, 1.0), (other, -1.0)], '-')
    
    def __rsub__(self, other):
        if not isinstance(other, Var):
            other = Var(other)
        return Var(other.val - self.val,
                  [(self, -1.0), (other, 1.0)], '-')
    
    def __mul__(self, other):
        if not isinstance(other, Var):
            other = Var(other)
        return Var(self.val * other.val,
                  [(self, other.val), (other, self.val)], '*')
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if not isinstance(other, Var):
            other = Var(other)
        return Var(self.val / other.val,
                  [(self, 1.0 / other.val),
                   (other, -self.val / other.val**2)], '/')
    
    def __pow__(self, n):
        return Var(self.val**n,
                  [(self, n * self.val**(n-1))], f'^{n}')
    
    def __neg__(self):
        return Var(-self.val, [(self, -1.0)], 'neg')


def var_sin(x):
    return Var(np.sin(x.val), [(x, np.cos(x.val))], 'sin')

def var_cos(x):
    return Var(np.cos(x.val), [(x, -np.sin(x.val))], 'cos')

def var_exp(x):
    e = np.exp(x.val)
    return Var(e, [(x, e)], 'exp')

def var_log(x):
    return Var(np.log(x.val), [(x, 1.0 / x.val)], 'log')

def var_tanh(x):
    t = np.tanh(x.val)
    return Var(t, [(x, 1 - t**2)], 'tanh')

def var_sqrt(x):
    s = np.sqrt(x.val)
    return Var(s, [(x, 0.5 / s)], 'sqrt')


def gradient_reverse(f, x_vals):
    """
    Compute gradient of f: R^n → R using reverse mode.
    
    Only ONE forward + backward pass regardless of n.
    
    Parameters
    ----------
    f : callable
        Function using Var operations
    x_vals : array
        Input point
    
    Returns
    -------
    grad : array
        Gradient ∇f(x)
    """
    x_vars = [Var(xi) for xi in x_vals]
    result = f(x_vars)
    result.backward()
    return np.array([xi.grad for xi in x_vars])


# ============================================================
# Finite Differences (for comparison)
# ============================================================

def finite_diff(f, x, h=1e-8):
    """Central finite difference approximation."""
    return (f(x + h) - f(x - h)) / (2 * h)


def finite_diff_gradient(f, x, h=1e-8):
    """Gradient via finite differences."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


# ============================================================
# Higher-order derivatives
# ============================================================

def second_derivative(f, x):
    """
    Compute f''(x) using nested dual numbers.
    
    f(a + ε₁b + ε₂c + ε₁ε₂d) extracts f''(a) from the ε₁ε₂ component.
    
    Here we use a simpler approach: apply forward mode twice.
    """
    def df(t):
        result = f(Dual(t, 1.0))
        return result.der
    
    result = Dual(x, 1.0)
    # Use finite diff on the derivative for simplicity
    h = 1e-7
    d2 = (df(x + h) - df(x - h)) / (2 * h)
    return f(Dual(x, 1.0)).val, f(Dual(x, 1.0)).der, d2


# ============================================================
# Physics applications
# ============================================================

def force_from_potential_1d(V, x):
    """
    Compute F = -dV/dx using AD.
    """
    _, dV = forward_diff(V, x)
    return -dV


def hamiltonian_equations(H, q_val, p_val):
    """
    Compute Hamilton's equations using AD:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
    
    Parameters
    ----------
    H : callable
        Hamiltonian H(q, p) using Dual numbers
    q_val, p_val : float
        Phase space coordinates
    """
    # ∂H/∂q (seed q, freeze p)
    result_q = H(Dual(q_val, 1.0), Dual(p_val, 0.0))
    dHdq = result_q.der
    
    # ∂H/∂p (freeze q, seed p)
    result_p = H(Dual(q_val, 0.0), Dual(p_val, 1.0))
    dHdp = result_p.der
    
    dqdt = dHdp
    dpdt = -dHdq
    
    return dqdt, dpdt


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AUTOMATIC DIFFERENTIATION DEMO")
    print("=" * 60)

    # --- 1. Forward Mode Basics ---
    print("\n--- Forward Mode (Dual Numbers) ---")
    
    test_functions = [
        ("x²",          lambda x: x**2,                   lambda x: 2*x),
        ("sin(x)",      lambda x: sin(x),                 lambda x: np.cos(x)),
        ("exp(x²)",     lambda x: exp(x**2),              lambda x: 2*x*np.exp(x**2)),
        ("x·sin(x)",    lambda x: x * sin(x),             lambda x: np.sin(x) + x*np.cos(x)),
        ("ln(1+x²)",    lambda x: ln(1 + x**2),           lambda x: 2*x/(1+x**2)),
        ("tanh(x)",     lambda x: tanh(x),                lambda x: 1/np.cosh(x)**2),
    ]
    
    x0 = 1.5
    print(f"  Evaluating at x = {x0}:")
    print(f"  {'Function':15s} {'AD deriv':>12s} {'Exact':>12s} {'Error':>12s}")
    for name, f, f_exact in test_functions:
        val, der = forward_diff(f, x0)
        exact = f_exact(x0)
        err = abs(der - exact)
        print(f"  {name:15s} {der:12.8f} {exact:12.8f} {err:12.2e}")

    # --- 2. Reverse Mode ---
    print("\n--- Reverse Mode (Computational Graph) ---")
    
    def rosenbrock(x):
        """Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²"""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_np(x):
        """NumPy version for finite diff."""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    x0_2d = np.array([1.5, 2.3])
    
    grad_rev = gradient_reverse(rosenbrock, x0_2d)
    grad_fd = finite_diff_gradient(rosenbrock_np, x0_2d)
    
    # Analytical: ∂f/∂x = -2(1-x) - 400x(y-x²), ∂f/∂y = 200(y-x²)
    x, y = x0_2d
    grad_exact = np.array([
        -2*(1-x) - 400*x*(y - x**2),
        200*(y - x**2)
    ])
    
    print(f"  Rosenbrock at ({x}, {y}):")
    print(f"    Reverse AD gradient: [{grad_rev[0]:10.4f}, {grad_rev[1]:10.4f}]")
    print(f"    Finite diff gradient:[{grad_fd[0]:10.4f}, {grad_fd[1]:10.4f}]")
    print(f"    Exact gradient:      [{grad_exact[0]:10.4f}, {grad_exact[1]:10.4f}]")
    print(f"    Rev AD error:  {np.linalg.norm(grad_rev - grad_exact):.2e}")
    print(f"    Fin diff error:{np.linalg.norm(grad_fd - grad_exact):.2e}")

    # --- 3. Jacobian ---
    print("\n--- Jacobian (Forward Mode) ---")
    
    def polar_to_cart(x):
        """(r, θ) → (x, y) = (r cos θ, r sin θ)"""
        return [x[0] * cos(x[1]), x[0] * sin(x[1])]
    
    r, theta = 2.0, np.pi/4
    J = jacobian_forward(polar_to_cart, [r, theta], m=2)
    
    print(f"  Polar ({r}, {theta:.4f}) → Cartesian Jacobian:")
    print(f"    [dx/dr  dx/dθ] = [{J[0,0]:8.4f}  {J[0,1]:8.4f}]")
    print(f"    [dy/dr  dy/dθ] = [{J[1,0]:8.4f}  {J[1,1]:8.4f}]")
    print(f"  Expected:")
    print(f"    [cos θ   -r sin θ] = [{np.cos(theta):8.4f}  {-r*np.sin(theta):8.4f}]")
    print(f"    [sin θ    r cos θ] = [{np.sin(theta):8.4f}  { r*np.cos(theta):8.4f}]")

    # --- 4. Physics: Force from potential ---
    print("\n--- Physics: Force from Potential ---")
    
    # Lennard-Jones potential
    def lj_potential(r):
        sigma = 1.0
        epsilon = 1.0
        sr6 = (sigma / r)**6
        return 4 * epsilon * (sr6**2 - sr6)
    
    for r_val in [0.9, 1.0, 1.12, 1.5, 2.0]:
        F = force_from_potential_1d(lj_potential, r_val)
        V_val = lj_potential(r_val)
        print(f"    r = {r_val:.2f}: V = {V_val:8.4f}, F = -dV/dr = {F:8.4f}")

    # --- 5. Hamilton's equations ---
    print("\n--- Hamilton's Equations via AD ---")
    
    # Harmonic oscillator: H = p²/2m + kx²/2
    def H_harmonic(q, p):
        m, k = 1.0, 4.0
        return p**2 / (2*m) + k * q**2 / 2
    
    q0, p0 = 1.0, 0.5
    dqdt, dpdt = hamiltonian_equations(H_harmonic, q0, p0)
    print(f"  H = p²/2 + 2q²  at q={q0}, p={p0}:")
    print(f"    dq/dt = ∂H/∂p = {dqdt:.4f} (expected {p0:.4f})")
    print(f"    dp/dt = -∂H/∂q = {dpdt:.4f} (expected {-4*q0:.4f})")

    # --- 6. AD vs Finite Differences: accuracy comparison ---
    print("\n--- Accuracy: AD vs Finite Differences ---")
    
    def test_func(x):
        return exp(sin(x)) * cos(x**2)
    
    def test_func_np(x):
        return np.exp(np.sin(x)) * np.cos(x**2)
    
    x_test = 1.0
    _, ad_deriv = forward_diff(test_func, x_test)
    
    print(f"  f(x) = exp(sin(x)) · cos(x²) at x = {x_test}")
    print(f"  {'Step size h':>12s} {'FD derivative':>15s} {'|FD - AD|':>12s}")
    
    for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
        fd = (test_func_np(x_test + h) - test_func_np(x_test - h)) / (2 * h)
        err = abs(fd - ad_deriv)
        print(f"  {h:12.0e} {fd:15.10f} {err:12.2e}")
    
    print(f"\n  AD derivative:   {ad_deriv:.15f}")
    print("  Note: FD has optimal h ~ 1e-8, degrades for smaller h (roundoff)")
    print("  AD always gives machine-precision exact result!")

    # --- 7. Gradient descent with AD ---
    print("\n--- Gradient Descent with AD ---")
    
    x_gd = np.array([3.0, 3.0])
    lr = 0.002
    
    for step in range(201):
        grad = gradient_reverse(rosenbrock, x_gd)
        x_gd = x_gd - lr * grad
        if step % 50 == 0:
            fval = rosenbrock_np(x_gd)
            print(f"    Step {step:4d}: x = ({x_gd[0]:7.4f}, {x_gd[1]:7.4f}), "
                  f"f = {fval:.6f}, |∇f| = {np.linalg.norm(grad):.4f}")
    
    print(f"  Minimum at (1, 1), found ({x_gd[0]:.4f}, {x_gd[1]:.4f})")
