# pyrefly: ignore [missing-import]
import torch

def hadamard_ratio(A, eps=1e-30):
    """
    Stable Hadamard ratio for a square basis matrix.

    HR close to 1 => good/private basis
    HR close to 0 => bad/skewed/public basis

    The log form avoids determinant/product overflow for 64x64 matrices.
    """
    A = A.real.to(torch.float64)
    n = A.shape[0]

    sign, logabsdet = torch.linalg.slogdet(A)

    # Singular or numerically singular matrix
    if sign == 0:
        return torch.tensor(0.0, dtype=torch.float64)

    col_norms = torch.linalg.norm(A, dim=0).clamp_min(eps)
    log_denom = torch.sum(torch.log(col_norms))

    log_hr = (logabsdet - log_denom) / n
    return torch.exp(log_hr)


def hadamard_quality(M, goodTh=0.8, badTh=0.001):
    """
    Classify a basis using Abdallah-style Hadamard-ratio thresholds.
    """
    hr = hadamard_ratio(M).item()

    if hr >= goodTh:
        return "good", hr

    if hr <= badTh:
        return "bad", hr

    return "medium", hr


def matrix_condition_number(A):
    """
    Condition number diagnostic.
    Large condition number means decryption with inverse will amplify noise.
    """
    A = A.real.to(torch.float64)
    return torch.linalg.cond(A).item()


def max_l1_row_norm(M):
    """
    rho = max over rows of the L1 norm of M.

    Abdallah Sec. 4.2.4 defines the correctness/security bound on the error
    vector via rho = the maximum L1 norm of the *rows* of R^{-1}. We reuse the
    same quantity for B^{-1} when checking that an eavesdropper is actually
    blocked.
    """
    return torch.linalg.norm(M.real.to(torch.float64), ord=1, dim=1).max().item()


def generate_good_private_basis(n, k=None, pert=2, max_tries=60, goodTh=0.8):
    """
    GGH-style good (private) basis: R = k*I + E, with E a small integer
    perturbation in {-pert, ..., +pert}.

    Why not Abdallah's literal Algorithm 1 (random entries in {0,...,N-1})?
    Random integer matrices have a Hadamard ratio of ~0.3 at N=64 and worse as
    N grows, so the paper's "HR(R) >= 0.8" is essentially never met by pure
    rejection sampling, and the repo's previous fallback was R = identity.
    R = identity makes the lattice Z^n, whose CVP is trivial for *anyone*
    (an eavesdropper just rounds the ciphertext), so it provides no security.

    The k*I + E construction yields a non-trivial full-rank lattice with a
    well-conditioned, near-orthogonal private basis (high HR, small ||R^{-1}||).
    A larger diagonal k increases HR and shrinks ||R^{-1}|| (which widens the
    admissible error window), so we grow k until HR(R) >= goodTh.
    """
    k_list = [k] if k is not None else [12 * pert, 16 * pert, 24 * pert, 40 * pert, 64 * pert]
    best, best_hr = None, -1.0
    for kk in k_list:
        for _ in range(max_tries):
            E = torch.randint(-pert, pert + 1, (n, n), dtype=torch.float64)
            R = kk * torch.eye(n, dtype=torch.float64) + E
            if torch.linalg.slogdet(R)[0] == 0:
                continue
            hr = hadamard_ratio(R).item()
            if hr > best_hr:
                best_hr, best = hr, R
            if hr >= goodTh:
                return R
    return best


def random_unimodular_matrix(n, rounds=5, max_coeff=3):
    """
    Generates an integer unimodular matrix U using targeted row operations.

    det(U) = ±1, so R @ U is a different basis for the same lattice.

    Unlike the old num_ops approach (which randomly picked rows and often
    missed most of them for large N), this version performs `rounds` full
    sweeps, mixing *every* row exactly once per sweep. With rounds=5 and
    max_coeff=3, HR(R@U) drops below 1e-3 even for N=512.
    """
    U = torch.eye(n, dtype=torch.float64)

    for _ in range(rounds):
        perm = torch.randperm(n)
        for i in range(n):
            j = perm[i].item()
            if i == j:
                j = (j + 1) % n

            k = torch.randint(1, max_coeff + 1, (1,)).item()

            if torch.rand(1).item() < 0.5:
                k = -k

            # Elementary row operation. This preserves unimodularity.
            U[i, :] = U[i, :] + k * U[j, :]

    return U


def generate_lattice_bases(
    n,
    rounds=5,
    max_coeff=3,
    goodTh=0.8,
    badTh=1e-3,
    max_bad_attempts=30,
    pert=2,
    k=None,
    insecure_identity_R=False,
    verbose=True,
):
    """
    Abdallah Algorithm 1, security-faithful version.

    R = private GOOD basis  (high Hadamard ratio, small ||R^{-1}||)
    B = public  BAD  basis  (Hadamard ratio -> 0, large ||B^{-1}||)

    Returns (R, B, info) where info carries the quantities that actually govern
    correctness and security:
        info["rho_R"] : max L1 row-norm of R^{-1}  -> correctness:  sigma < 1/(2*rho_R)
        info["rho_B"] : max L1 row-norm of B^{-1}  -> security:     sigma > 1/(2*rho_B)
    so a usable error scale sigma must satisfy  1/(2*rho_B) < sigma < 1/(2*rho_R).

    Parameters
    ----------
    rounds : int
        Number of full-sweep rounds for the unimodular matrix generation.
        Each round mixes every row once. 5 rounds is sufficient to drive
        HR(B) below 1e-3 for N up to 512.
    """
    # 1. Private GOOD basis R
    if insecure_identity_R:
        R = torch.eye(n, dtype=torch.float64)
        if verbose:
            print("WARNING: insecure_identity_R=True -> lattice is Z^n, NOT secure.")
    else:
        R = generate_good_private_basis(n, k=k, pert=pert, goodTh=goodTh)

    quality_R, hr_R = hadamard_quality(R, goodTh=goodTh, badTh=badTh)
    R_inv = torch.linalg.pinv(R)
    rho_R = max_l1_row_norm(R_inv)

    # 2. Public BAD basis B = R @ U, pushed as bad as possible.
    #    The targeted unimodular generator reliably produces HR < badTh,
    #    so we just pick the best of a few attempts — no condition-number
    #    filter, because a high cond(B) is exactly what blocks the attacker.
    best_B, best_hr_B = None, float("inf")
    for _ in range(max_bad_attempts):
        U = random_unimodular_matrix(n=n, rounds=rounds, max_coeff=max_coeff)
        B_candidate = R @ U
        hr_B = hadamard_ratio(B_candidate).item()
        if hr_B < best_hr_B:
            best_hr_B, best_B = hr_B, B_candidate
        if hr_B <= badTh:
            break

    if best_B is None:
        best_B = R.clone()
        best_hr_B = hadamard_ratio(best_B).item()
        if verbose:
            print("Warning: could not generate B. Using B = R (INSECURE).")

    B_inv = torch.linalg.pinv(best_B)
    rho_B = max_l1_row_norm(B_inv)

    info = {
        "rho_R": rho_R,
        "rho_B": rho_B,
        "hr_R": hr_R,
        "hr_B": best_hr_B,
        "cond_B": matrix_condition_number(best_B),
        "sigma_max_correct": 1.0 / (2.0 * rho_R),
        "sigma_min_secure": 1.0 / (2.0 * rho_B),
    }

    if verbose:
        print(f"R quality: {quality_R}  HR(R)={hr_R:.4e}  rho_R={rho_R:.4f}")
        print(f"B quality: {'bad' if best_hr_B <= badTh else 'NOT below badTh'}  "
              f"HR(B)={best_hr_B:.4e}  rho_B={rho_B:.4f}")
        lo, hi = info["sigma_min_secure"], info["sigma_max_correct"]
        if lo < hi:
            print(f"Usable error scale: {lo:.4e} < sigma < {hi:.4e}")
        else:
            print(f"WARNING: empty sigma window (sigma_min_secure={lo:.3e} >= "
                  f"sigma_max_correct={hi:.3e}); basis pair is unsuitable.")

    return R.to(torch.complex128), best_B.to(torch.complex128), info