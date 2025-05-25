import numpy as np

def Improved_DMDc_new(X_data: np.ndarray, U_control: np.ndarray, rank_p: int, rank_r: int):
    """
    Compute Improved Dynamic Mode Decomposition with Control (Improved DMDc).

    Parameters:
    - X_data: ndarray of shape (state_dim, num_time_steps)
              Time series data of states.
    - U_control: ndarray of shape (u_dim, num_time_steps)
                 Time series data of control inputs.
    - rank_p: int
              Truncation rank for SVD of Omega.
    - rank_r: int
              Truncation rank for SVD of H.

    Returns:
    - Lamda_a: ndarray
               Eigenvalues of the reduced transition matrix A.
    - Phi: ndarray
           DMD modes.
    - Y_dmd_recons: ndarray
                    Reconstructed future states using DMDc.
    - bar_A: ndarray
             Reduced system matrix A.
    - bar_B: ndarray
             Reduced input matrix B.
    - bar_V: ndarray
             Projection matrix for reduced space.
    """
    # 1: Get dimensions and Construct snapshot matrices
    state_dim, num_time_steps = X_data.shape
    u_dim, _ = U_control.shape
    assert num_time_steps == U_control.shape[1] + 1, "X_data and U_control must have the same number of time steps."

    # Construct snapshot matrices
    X = X_data[:, :-1]  # Current states (time k)
    Y = X_data[:, 1:]   # Future states (time k+1)
    Gamma = U_control  # Control inputs aligned with X

    # Augmented data matrix
    Omega = np.vstack((X, Gamma))

    # 2: Perform SVD for Omega
    UU, SS, VVt = np.linalg.svd(Omega, full_matrices=False)

    # Truncate SVD components
    p = min(rank_p, len(SS))
    U_p = UU[:, :p]
    S_p = SS[:p]
    V_p = VVt.T[:, :p]  # V from SVD

    # Split U_p into state and control parts
    U1_p = U_p[:state_dim, :]  # State part
    U2_p = U_p[state_dim:, :]  # Control part

    # 3: Compute H for Improved DMDc
    if p > 0:
        inv_S_p = np.linalg.inv(np.diag(S_p))
        H = inv_S_p @ U1_p.T  # H = Sigma^{-1} * U1^T
    else:
        raise ValueError("Truncation rank p must be greater than 0.")

    # SVD of H
    Uh, Sh, Vth = np.linalg.svd(H, full_matrices=False)
    r = min(rank_r, len(Sh))
    Uh_r = Uh[:, :r]
    Sh_r = Sh[:r]
    bar_V = Vth.T[:, :r]  # bar_V = V_h

    # Compute bar_U (projection matrix)
    bar_U = V_p @ Uh_r  # V_p is m x p, Uh_r is p x r -> m x r

    # 4: Compute reduced approximations of A and B
    bar_A = bar_V.T @ Y @ bar_U @ np.diag(Sh_r)
    Omega_2 = V_p @ inv_S_p @ U2_p.T  # m x p @ p x p @ p x u_dim -> m x u_dim
    bar_B = bar_V.T @ Y @ Omega_2  # r x n @ n x m @ m x u_dim -> r x u_dim

    # 5: Eigen decomposition of reduced transition matrix bar_A
    Lamda_a, W_a = np.linalg.eig(bar_A)

    # 6: Compute DMD modes of original system (aka eigenvectors of original matrix A)
    Phi = Y @ bar_U @ np.diag(Sh_r) @ W_a  # n x m @ m x r @ r x r @ r x r -> n x r

    # 7: Reconstruction
    z_k = bar_V.T @ X  # Reduced states z_k = bar_V^T * X
    Y_tilta_recons = bar_A @ z_k + bar_B @ Gamma  # r x r @ r x (T-1) + r x u_dim @ u_dim x (T-1)
    Y_dmd_recons = bar_V @ Y_tilta_recons  # n x r @ r x (T-1) -> n x (T-1)
    # print("the shape of Y_dmd_recons inside dmdc is: ", Y_dmd_recons.shape)
    return Lamda_a, Phi, Y_dmd_recons, bar_A, bar_B, bar_V