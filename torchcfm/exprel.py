import torch 
import math 


NTERMS_FLOAT32 = 9
XBND_FLOAT32 = 1.1920928955078125e-07

NTERMS_FLOAT64 = 16
XBND_FLOAT64 = 2.220446049250313e-16

DTYPE_PARAMS = {
    torch.float32: (NTERMS_FLOAT32, XBND_FLOAT32),
    torch.float64: (NTERMS_FLOAT64, XBND_FLOAT64)
}


def exprel(x: torch.Tensor) -> torch.Tensor:
    """
    Fortran-1977 style exprel implementation in PyTorch.
    exprel(x) = (exp(x) - 1)/x, with series expansion for small |x|.
    """
    nterms, xbnd = DTYPE_PARAMS[x.dtype]

    result = torch.empty_like(x)
    absx = torch.abs(x)

    # Case 1: large |x|
    mask_large = absx > 0.5
    x_mask_large = x[mask_large]
    result[mask_large] = torch.expm1(x_mask_large) / x_mask_large

    # Case 2: very small |x|
    mask_tiny = absx < xbnd
    result[mask_tiny] = torch.ones_like(x[mask_tiny])

    # Case 3: moderate |x| → series
    mask_series = ~(mask_large | mask_tiny)
    x_mask_series = x[mask_series]
    out_series = torch.zeros_like(x_mask_series)
    for i in range(1, nterms + 1):
        out_series = 1.0 + out_series * x_mask_series / float(nterms + 2 - i)
    result[mask_series] = out_series

    return result


def _compute_constants(dtype):
    eps = torch.finfo(dtype).eps
    alneps = math.log(eps)
    xn = 3.72 - 0.3 * alneps
    xln = math.log((xn + 1.0) / 1.36)
    nterms = int(xn - (xn * xln + alneps) / (xln + 1.36) + 1.5)
    return nterms, eps


def _validate_constants(dtype_params):
    for dtype, (nterms, xbnd) in dtype_params.items():
        nterms_computed, xbnd_computed = _compute_constants(dtype)
        if nterms != nterms_computed or xbnd != xbnd_computed:
            raise ValueError(
                f"dtype {dtype} has wrong constants: "
                f"Got: nterms={nterms}, xbnd={xbnd}, "
                f"Expected: nterms={nterms_computed}, xbnd={xbnd_computed}"
            )