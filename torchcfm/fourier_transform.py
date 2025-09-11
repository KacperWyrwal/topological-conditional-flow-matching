import torch 
from scipy.sparse.linalg import LaplacianNd
from abc import ABC, abstractmethod


def grid_laplacian_eigenpairs(
    shape: tuple[int, ...],
    boundary_conditions: str = 'neumann',
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the eigenvectors and eigenvalues of the Laplacian on a grid
    in increasing order of eigenvalues.

    Parameters
    ----------
    n_rows : int
        The number of rows in the grid.
    n_cols : int
        The number of columns in the grid.
    boundary_conditions : str
        The boundary conditions to use.
    dtype : torch.dtype, optional
        The dtype of the eigenvectors and eigenvalues.
    device : torch.device, optional
        The device of the eigenvectors and eigenvalues.

    Returns
    -------
    eigvecs : torch.Tensor, shape (N * M, N * M)
        The eigenvectors of the Laplacian.
    eigvals : torch.Tensor, shape (N * M,)
        The eigenvalues of the Laplacian.
    """
    assert boundary_conditions in ['neumann', 'dirichlet', 'periodic']

    if dtype is None: dtype = torch.get_default_dtype()
    if device is None: device = torch.get_default_device()

    L = LaplacianNd(
        grid_shape=shape,
        boundary_conditions=boundary_conditions
    )
    eigvecs, eigvals = L.eigenvectors(), -L.eigenvalues() # Change sign from negative to positive
    eigvecs = torch.from_numpy(eigvecs).to(dtype=dtype, device=device)
    eigvals = torch.from_numpy(eigvals).to(dtype=dtype, device=device)

    # reverse order of eigenvectors and eigenvalues
    eigvecs = eigvecs.flip(-1)
    eigvals = eigvals.flip(-1)

    return eigvecs, eigvals


class FourierTransform(ABC):

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        params:
            x: torch.Tensor, shape (..., N, M)
        returns:
            torch.Tensor, shape (..., N * M)
        """

    @abstractmethod
    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        params:
            y: torch.Tensor, shape (..., N * M)
        returns:
            torch.Tensor, shape (..., N, M)
        """


class Naive1DFourierTransform(FourierTransform):

    def __init__(self, eigenvectors: torch.Tensor) -> None:
        """
        params:
            eigenvectors: torch.Tensor, shape (K, D)
        """
        super().__init__()
        self.eigenvectors = eigenvectors

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        params:
            x: torch.Tensor, shape (..., D)
        returns:
            torch.Tensor, shape (..., K)
        """
        y = torch.einsum('ij, ...j -> ...i', self.eigenvectors, x)
        return y

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        params:
            y: torch.Tensor, shape (..., K)
        returns:
            torch.Tensor, shape (..., D)
        """
        x = torch.einsum('ij, ...j -> ...i', self.eigenvectors.mT, y)
        return x


class NaiveGridFourierTransform(FourierTransform):

    def __init__(self, shape: tuple[int, ...], eigenvectors: torch.Tensor) -> None:
        """
        params:
            eigenvectors: torch.Tensor, shape (N1 * N2 * ... * Nk, K)
        """
        super().__init__()
        self.shape = shape
        self.dim = len(shape)
        self.eigenvectors = eigenvectors.mT # (K, N1 * N2 * ... * Nk)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        params:
            x: torch.Tensor, shape (..., N1, N2, ..., Nk)
        returns:
            torch.Tensor, shape (..., K)
        """
        x = torch.flatten(x, start_dim=-self.dim) # (..., N1 * N2 * ... * Nk)
        y = torch.einsum('ij, ...j -> ...i', self.eigenvectors, x)
        return y

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        params:
            y: torch.Tensor, shape (..., K)
        returns:
            torch.Tensor, shape (..., N1, N2, ..., Nk)
        """
        x = torch.einsum('ij, ...j -> ...i', self.eigenvectors.mT, y) # (..., N1 * N2 * ... * Nk)
        x = x.reshape(*x.shape[:-1], *self.shape) # (..., N1, N2, ..., Nk)
        return x
