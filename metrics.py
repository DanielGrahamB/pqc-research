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


def random_unimodular_matrix(n, num_ops=45, max_coeff=2):
    """
    Generates an integer unimodular matrix U using safe row operations.

    det(U) = ±1, so R @ U is a different basis for the same lattice.
    For this OFDM notebook, keep num_ops/max_coeff moderate to avoid
    floating-point explosion during decryption.
    """
    U = torch.eye(n, dtype=torch.float64)

    for _ in range(num_ops):
        i = torch.randint(0, n, (1,)).item()
        j = torch.randint(0, n, (1,)).item()

        if i == j:
            continue

        k = torch.randint(1, max_coeff + 1, (1,)).item()

        if torch.rand(1).item() < 0.5:
            k = -k

        # Elementary row operation. This preserves unimodularity.
        U[i, :] = U[i, :] + k * U[j, :]

    return U


def generate_lattice_bases(
    n,
    num_ops=45,
    max_coeff=2,
    goodTh=0.8,
    badTh=0.01,
    max_bad_attempts=70,
    max_cond=1e5,
    use_random_private=False,
    verbose=True
):
    """
    Notebook-friendly Abdallah-style basis generation.

    R = private good basis
    B = public degraded basis

    Important simulation choice:
    Abdallah's paper uses BadTh = 0.001. For a 64x64 floating-point OFDM
    mini-simulation, forcing B that low often makes inv(B) numerically unstable.
    This notebook therefore uses a practical default badTh=0.01 and rejects
    matrices with too large a condition number.
    """

    # 1. Private good basis R
    # Identity is a valid very-good basis with HR = 1 and is stable for the notebook.
    if use_random_private:
        R = None

        for _ in range(500):
            candidate = torch.randint(
                low=0,
                high=n,
                size=(n, n),
                dtype=torch.float64
            )

            quality, _ = hadamard_quality(candidate, goodTh=goodTh, badTh=badTh)

            if quality == "good":
                R = candidate
                break

        if R is None:
            if verbose:
                print("Could not find a random good R. Falling back to identity R.")
            R = torch.eye(n, dtype=torch.float64)
    else:
        R = torch.eye(n, dtype=torch.float64)

    quality_R, hr_R = hadamard_quality(R, goodTh=goodTh, badTh=badTh)

    # 2. Public basis B = R @ U
    # We search for a basis that is degraded, but still numerically usable.
    best_B = None
    best_score = float("inf")
    best_hr_B = None
    best_quality_B = None
    best_cond_B = None

    for _ in range(max_bad_attempts):
        U = random_unimodular_matrix(
            n=n,
            num_ops=num_ops,
            max_coeff=max_coeff
        )

        B_candidate = R @ U
        quality_B, hr_B = hadamard_quality(B_candidate, goodTh=goodTh, badTh=badTh)
        cond_B = matrix_condition_number(B_candidate)

        # Reject matrices that will amplify noise too much in this notebook.
        if not torch.isfinite(torch.tensor(cond_B)) or cond_B > max_cond:
            continue

        # Prefer lower Hadamard ratio, while staying below max_cond.
        if hr_B < best_score:
            best_score = hr_B
            best_B = B_candidate
            best_hr_B = hr_B
            best_quality_B = quality_B
            best_cond_B = cond_B

        if quality_B == "bad":
            break

    # Fallback in case all candidates were too ill-conditioned
    if best_B is None:
        best_B = R.clone()
        best_quality_B, best_hr_B = hadamard_quality(best_B, goodTh=goodTh, badTh=badTh)
        best_cond_B = matrix_condition_number(best_B)
        if verbose:
            print("Warning: all public-basis candidates were rejected. Using B = R.")

    if verbose:
        print(f"R quality: {quality_R}")
        print(f"Hadamard ratio R: {hr_R:.6e}")
        print(f"B quality: {best_quality_B}")
        print(f"Hadamard ratio B: {best_hr_B:.6e}")
        print(f"Condition number B: {best_cond_B:.6e}")

        if quality_R != "good":
            print(f"Warning: R is not a good basis under goodTh={goodTh}")

        if best_quality_B != "bad":
            print(
                f"Note: B is not below badTh={badTh}. "
                "That is acceptable for this OFDM mini-simulation because numerical stability matters."
            )

    return R.to(torch.complex64), best_B.to(torch.complex64)
