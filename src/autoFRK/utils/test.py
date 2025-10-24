import torch
from typing import Union

def K(
    X: torch.Tensor,
    Y: torch.Tensor,
    n: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """

    """
    xx = torch.zeros((n, n), dtype=dtype, device=device)
    for i in range(n):
        for j in range(n):
            xx[i, j] = Kf(L1    = X[i],
                          l1    = Y[i],
                          L2    = X[j],
                          l2    = Y[j],
                          dtype = dtype,
                          device= device
                          )

    return xx

def Kf(
    L1: float,
    l1: float,
    L2: float,
    l2: float,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """

    """
    pi = torch.tensor(torch.pi, dtype=dtype, device=device)
    mia = pi / 180.0
    a = torch.sin(L1 * mia) * torch.sin(L2 * mia) + torch.cos(L1 * mia) * torch.cos(L2 * mia) * torch.cos(l1 * mia - l2 * mia)
    b = torch.clamp(a, -1.0, 1.0)
    aaa = torch.acos(b)

    if torch.cos(aaa) == -1:
        result = 1 - pi ** 2 / 6
    else:
        upper = 0.5 + torch.cos(aaa) / 2
        lower = 0.0
        num_steps = 100000

        # func3 f(x) = log(1 - x) / x
        x_vals = torch.linspace(lower + 1e-10, upper - 1e-10, num_steps, dtype=dtype, device=device)
        y_vals = torch.log(1 - x_vals) / x_vals

        # integrate(f, lower, aa, ...)
        res = torch.trapz(y_vals, x_vals)

        result = 1 - pi ** 2 / 6 - res

    return result

def inprod(
    A: torch.Tensor,
    B: torch.Tensor,
    m: int
) -> torch.Tensor:
    return torch.dot(A[:m], B[:m])

def fk(
    L1: float,
    l1: float,
    KK: int,
    X: torch.Tensor,
    Konev: torch.Tensor,
    eiKvecmval: torch.Tensor,
    n: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:

    f1 = torch.zeros(KK, dtype=dtype, device=device)
    f2 = torch.zeros(n, dtype=dtype, device=device)
    f1[0] = torch.sqrt(torch.tensor(1.0 / n, dtype=dtype, device=device))

    for i in range(n):
        f2[i] = Kf(L1       = L1,
                   l1       = l1,
                   L2       = X[i, 0],
                   l2       = X[i, 1],
                   dtype    = dtype,
                   device   = device
                   )

    t = torch.zeros(n, dtype=dtype, device=device)
    for i in range(n):
        t[i] = f2[i] - Konev[i]

    for i in range(1, KK):
        f1[i] = inprod(A = t,
                       B = eiKvecmval[:, i - 1],
                       m = n
                       )

    return f1

def Kmatrix(
    KK: int,
    X: torch.Tensor,
    ggrids: torch.Tensor,
    Konev: torch.Tensor,
    eiKvecmval: torch.Tensor,
    n: int,
    N: int,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """

    """
    pi = torch.tensor(torch.pi, dtype=dtype, device=device)
    xx = torch.zeros((N, KK), dtype=dtype, device=device)
    mia = pi / 180.0

    for i in range(N):
        L1 = ggrids[i, 0]
        l1 = ggrids[i, 1]

        f2 = torch.zeros(n, dtype=dtype, device=device)
        for j in range(n):
            L2 = X[j, 0]
            l2 = X[j, 1]

            a = torch.sin(L1 * mia) * torch.sin(L2 * mia) + torch.cos(L1 * mia) * torch.cos(L2 * mia) * torch.cos(l1 * mia - l2 * mia)
            b = torch.clamp(a, -1.0, 1.0)
            aaa = torch.acos(b)

            if torch.cos(aaa) == -1:
                result = 1.0 - pi ** 2 / 6.0
            else:
                upper = 0.5 + torch.cos(aaa) / 2.0
                lower = 0.0
                num_steps = 100000

                # f(x) = log(1 - x) / x
                x_vals = torch.linspace(lower + 1e-10, upper - 1e-10, num_steps, dtype=dtype, device=device)
                y_vals = torch.log(1 - x_vals) / x_vals
                res = torch.trapz(y_vals, x_vals)

                result = 1.0 - pi ** 2 / 6.0 - res

            f2[j] = result

        t = f2 - Konev
        xx[i, 0] = torch.sqrt(torch.tensor(1.0 / n, dtype=dtype, device=device))
        for k in range(1, KK):
            s = torch.dot(t, eiKvecmval[:, k - 1])
            xx[i, k] = s

    return xx