import os
import sys
from enum import Enum, auto
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


def load_img(path: str) -> np.ndarray:
    if os.path.exists(path):
        img = Image.open(path).convert("RGB")
        img_array = np.array(img) / 255.0
        return img_array
    else:
        print("Err")
        sys.exit(1)


class NoiseType(Enum):
    GAUSSIAN = auto()
    ADDITIVE = auto()
    MULTIPLICATIVE = auto()
    SALT_N_PEPPER = auto()


def gausian_noise(shape: tuple[int, int], varience: float) -> np.ndarray:
    stddev = np.sqrt(varience)
    noise = np.random.normal(0, stddev, shape)

    return noise


def additive_uniform_noise(shape: tuple[int, int], variance: float) -> np.ndarray:
    """Additive uniform noise. Spread is derived from variance: range = ±sqrt(3 * variance)."""
    spread = np.sqrt(3.0 * variance)
    return np.random.uniform(-spread, spread, shape)


def multiplicative_noise(channel: np.ndarray, variance: float) -> np.ndarray:
    """
    Speckle / sensor noise: noise amplitude scales with pixel brightness.
    noisy = channel * (1 + N(0, variance))
    noisy = I + N * I (where I is the original image/ channel)
    Bright pixels get more absolute noise than dark ones.
    """
    noise = np.random.normal(0.0, np.sqrt(variance), channel.shape)
    return np.clip(channel * (1.0 + noise), 0.0, 1.0)


def poisson_noise(channel: np.ndarray, scale: float = 255.0) -> np.ndarray:
    """
    Shot noise: models discrete photon arrivals. Variance = mean intensity.
    We scale up to integer counts, sample Poisson, then scale back.
    Higher scale = finer-grained / less visible noise.
    """
    counts = np.random.poisson(channel * scale)
    return np.clip(counts / scale, 0.0, 1.0)


def add_noise_to_channel(
    channel: np.ndarray,
    noise_type: NoiseType,
    variance: float,
    snp_density: float,
) -> npt.NDArray[np.floating[Any]]:
    if noise_type == NoiseType.GAUSSIAN:
        noise = np.random.normal(0.0, np.sqrt(variance), channel.shape)
        return np.clip(channel + noise, 0.0, 1.0)

    elif noise_type == NoiseType.ADDITIVE:
        spread = np.sqrt(3.0 * variance)
        noise = np.random.uniform(-spread, spread, channel.shape)
        return np.clip(channel + noise, 0.0, 1.0)

    elif noise_type == NoiseType.MULTIPLICATIVE:
        noise = np.random.normal(0.0, np.sqrt(variance), channel.shape)
        return np.clip(channel * (1.0 + noise), 0.0, 1.0)

    elif noise_type == NoiseType.SALT_N_PEPPER:
        noisy = channel.copy()
        n_corrupt = int(snp_density * channel.size)
        indices = np.random.choice(channel.size, size=n_corrupt, replace=False)
        salt_idx = indices[: n_corrupt // 2]
        pepper_idx = indices[n_corrupt // 2 :]
        noisy.ravel()[salt_idx] = 1.0
        noisy.ravel()[pepper_idx] = 0.0
        return noisy

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def add_noise_to_img(
    image: np.ndarray,
    noise_type: NoiseType,
    variance: float,
    snp_density: float,
    color_noise: bool,
) -> np.ndarray:
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    if color_noise:
        nr = add_noise_to_channel(r, noise_type, variance, snp_density)
        ng = add_noise_to_channel(g, noise_type, variance, snp_density)
        nb = add_noise_to_channel(b, noise_type, variance, snp_density)
    else:
        # Same noise pattern on all channels (greyscale grain)
        nr = add_noise_to_channel(r, noise_type, variance, snp_density)
        ng = np.clip(g + (nr - r), 0.0, 1.0)
        nb = np.clip(b + (nr - r), 0.0, 1.0)

    return np.dstack([nr, ng, nb])


def gaussian_kernel_2d(sigma: float) -> np.ndarray:
    """
    Build a 2D Gaussian kernel manually.

    A Gaussian kernel is a grid of weights shaped like a bell curve:
        K(i, j) = exp(-(i^2 + j^2) / (2 * sigma^2))

    We then normalise so all weights sum to 1, so pixel intensities
    are preserved on average (it's a weighted average of neighbours).
    Kernel size is chosen as 6*sigma + 1 to capture ~99% of the distribution.
    """
    radius = int(3 * sigma)
    size = 2 * radius + 1
    coords = np.arange(size) - radius  # [-r, ..., 0, ..., r]
    kernel_1d = np.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d)  # 2D = outer product of 1D with itself


def convolve_2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2D channel with a kernel using numpy's stride tricks.
    Pads with edge-replication (Neumann BC) before convolving.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

    # Use np.lib.stride_tricks to create a sliding window view — no Python loop

    windows = sliding_window_view(padded, (kh, kw))  # shape: (h, w, kh, kw)
    return (windows * kernel).sum(axis=(-2, -1))


def smooth_channel(channel: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_2d(sigma)
    return convolve_2d(channel, kernel)


# def diffusivity(grad_sq: np.ndarray, lam: float) -> np.ndarray:
#     """
#     1 / (1 + s/λ)
#     """
#     return 1.0 / (1.0 + grad_sq / lam)
#
#
# def common_pixels_by_shift(
#     img: np.ndarray, row_shift: int, col_shift: int
# ) -> np.ndarray:
#     """
#      Returns a copy of img where every pixel (i, j) holds the value
#     that was at (i + row_shift, j + col_shift) in the original.
#
#     Out-of-bounds indices are clamped to the border (Neumann BC).
#     """
#
#     h, w = img.shape
#
#     row_indices = np.clip(np.arange(h) + row_shift, 0, h - 1)
#     col_indices = np.clip(np.arange(w) + col_shift, 0, w - 1)
#     return img[np.ix_(row_indices, col_indices)]
#
#
# def gradient_mag_sq(channel: np.ndarray) -> np.ndarray:
#     """
#     Finds | grad(U) |^2 for each direction
#
#     dU/di ≈ (U[i+1, j] - U[i-1, j]) / 2      (vertical)
#     dU/dj ≈ (U[i, j+1] - U[i, j-1]) / 2      (horizontal)
#
#     |grad U|^2 = (dU/di)^2 + (dU/dj)^2
#
#     """
#     south_neighbour = common_pixels_by_shift(channel, 1, 0)  # U[i+1, j]
#     north_neighbour = common_pixels_by_shift(channel, -1, 0)  # U[i-1, j]
#     east_neighbour = common_pixels_by_shift(channel, 0, 1)  # U[i, j+1]
#     west_neighbour = common_pixels_by_shift(channel, 0, -1)  # U[i, j-1]
#
#     vertical_diff = (south_neighbour - north_neighbour) / 2.0  # dU/di
#     horizontal_diff = (east_neighbour - west_neighbour) / 2.0  # dU/dj
#
#     return vertical_diff**2 + horizontal_diff**2


def get_structure_tensor(
    channel: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    J = | grad_x^2        grad_x * grad_y |
        | grad_x * grad_y grad_y^2        |

    - Structure Tensor J has the gradient drection of each pixel

    The gradient of the smoothed image gives us:
        grad_x = du/dx   (horizontal rate of change)
        grad_y = du/dy   (vertical rate of change)

    """
    # channel = smooth_channel(channel, sigma=sigma) # Smooth the image with gausian kernal, might help?

    grad_x = (np.roll(channel, -1, axis=1) - np.roll(channel, 1, axis=1)) / 2.0
    grad_y = (np.roll(channel, -1, axis=0) - np.roll(channel, 1, axis=0)) / 2.0

    J_xx = grad_x * grad_x
    J_xy = grad_x * grad_y
    J_yy = grad_y * grad_y

    return J_xx, J_xy, J_yy


def build_diffusion_tensor(
    J_xx: np.ndarray, J_xy: np.ndarray, J_yy: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
      We want a diffusion tensor D whose eigenvectors are the same as J,
      but with carefully chosen eigenvalues that control how much we diffuse in each direction.

    SLOP -> Figure this part out

    Step 1 — Eigenvalues of J (closed form for a 2x2 symmetric matrix):
          mu_large = (J_xx + J_yy)/2 + sqrt(((J_xx - J_yy)/2)^2 + J_xy^2)
          mu_small = (J_xx + J_yy)/2 - sqrt(...)

          mu_large corresponds to the across-edge direction (max gradient)
          mu_small corresponds to the along-edge direction

      Step 2 — Assign new eigenvalues for D:
          d_across = g(mu_large)   small at edges — block diffusion across edges
          d_along  = 1.0           always 1 — freely diffuse along edges

          where g(s) = 1 / (1 + s/lambda)  (same Perona-Malik edge stopping)

      Step 3 — Eigenvector of J for mu_large (the across-edge direction):
          For a 2x2 symmetric matrix, if mu is an eigenvalue then
          (J - mu*I)v = 0, which gives v = (J_xy, mu - J_xx) (unnormalised).
          We normalise it and call it eigenvec_across.
          The along-edge eigenvector is just perpendicular to it.

      Step 4 — Reconstruct D from its eigendecomposition (P Λ P^T):
          D = d_across * eigenvec_across * eigenvec_across^T
            + d_along  * eigenvec_along  * eigenvec_along^T

      Since D is symmetric we again only need 3 values: D_xx, D_xy, D_yy.
    """
    # Step 1 — eigenvalues of J
    half_trace = (J_xx + J_yy) / 2.0
    discriminant = np.sqrt(np.maximum(((J_xx - J_yy) / 2.0) ** 2 + J_xy**2, 0.0))

    mu_large = half_trace + discriminant  # across-edge eigenvalue
    # mu_small = half_trace - discriminant  # along-edge eigenvalue (not needed explicitly)

    # Step 2 — diffusivity eigenvalues for D
    d_across = 1.0 / (1.0 + mu_large / lam)  # small at strong edges
    d_along = np.ones_like(d_across)  # always 1

    # Step 3 — eigenvector for the across-edge direction
    eigenvec_across_unnorm_x = J_xy
    eigenvec_across_unnorm_y = mu_large - J_xx

    norm = np.sqrt(eigenvec_across_unnorm_x**2 + eigenvec_across_unnorm_y**2)
    is_flat = norm < 1e-10  # avoid divide-by-zero in uniform regions

    eigenvec_across_x = np.where(is_flat, 1.0, eigenvec_across_unnorm_x / norm)
    eigenvec_across_y = np.where(is_flat, 0.0, eigenvec_across_unnorm_y / norm)

    # Along-edge eigenvector is perpendicular to across-edge
    eigenvec_along_x = -eigenvec_across_y
    eigenvec_along_y = eigenvec_across_x

    # Step 4 — reconstruct D = d_across * v_across v_across^T + d_along * v_along v_along^T
    D_xx = (
        d_across * eigenvec_across_x * eigenvec_across_x
        + d_along * eigenvec_along_x * eigenvec_along_x
    )
    D_xy = (
        d_across * eigenvec_across_x * eigenvec_across_y
        + d_along * eigenvec_along_x * eigenvec_along_y
    )
    D_yy = (
        d_across * eigenvec_across_y * eigenvec_across_y
        + d_along * eigenvec_along_y * eigenvec_along_y
    )

    return D_xx, D_xy, D_yy


def eed_step(channel: np.ndarray, dt: float, lam: float, sigma: float) -> np.ndarray:
    """
    The divergence of D*∇u expands as:
        div(D ∇u) = d/dx (D_xx * u_x + D_xy * u_y)
                  + d/dy (D_xy * u_x + D_yy * u_y)

    We compute D from the structure tensor of the image, but we apply it to the gradients of the actual current channel (that's what we're actually diffusing).

    SLOP -> Figure this out? kainda get it but the part above is weird, read paper
    """

    J_xx, J_xy, J_yy = get_structure_tensor(channel, sigma)
    D_xx, D_xy, D_yy = build_diffusion_tensor(J_xx, J_xy, J_yy, lam)

    # Gradients of the actual (unsmoothed) channel
    grad_x = (np.roll(channel, -1, axis=1) - np.roll(channel, 1, axis=1)) / 2.0
    grad_y = (np.roll(channel, -1, axis=0) - np.roll(channel, 1, axis=0)) / 2.0

    # Flux:  F = D * ∇u
    flux_x = D_xx * grad_x + D_xy * grad_y
    flux_y = D_xy * grad_x + D_yy * grad_y

    # Divergence of flux
    div_flux = (np.roll(flux_x, -1, axis=1) - np.roll(flux_x, 1, axis=1)) / 2.0 + (
        np.roll(flux_y, -1, axis=0) - np.roll(flux_y, 1, axis=0)
    ) / 2.0

    return np.clip(channel + dt * div_flux, 0.0, 1.0)


def denoise_channel(
    channel: np.ndarray, dt: float, num_iterations: int, lam: float, sigma: float
) -> np.ndarray:
    result = channel.copy()
    for _ in range(num_iterations):
        result = eed_step(result, dt, lam, sigma)
    return result


def main(  # noqa
    path: str,
    variance: float = 0.05,
    dt: float = 0.1,
    num_iterations: int = 100,
    lam: float = 0.003,
    sigma: float = 1.0,
    color_noise: bool = False,
    save: bool = False,
    output_path: str = "anisotropic_diffusion.png",
    snp_density: float = 0.05,
) -> None:
    original = load_img(path)

    r_orig, g_orig, b_orig = original[:, :, 0], original[:, :, 1], original[:, :, 2]

    n_types = len(NoiseType)
    fig, axes = plt.subplots(2, n_types + 1, figsize=(4 * (n_types + 1), 8))
    fig.suptitle(
        f"Edge Enhansing Diffusion — iter={num_iterations}  dt={dt}  λ={lam}",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for col, noise_type in enumerate(NoiseType, start=1):
        noisy = add_noise_to_img(
            original, noise_type, variance, snp_density, color_noise
        )

        red, green, blue = noisy[:, :, 0], noisy[:, :, 1], noisy[:, :, 2]
        denoised = np.dstack(
            [
                denoise_channel(red, dt, num_iterations, lam, sigma),
                denoise_channel(green, dt, num_iterations, lam, sigma),
                denoise_channel(blue, dt, num_iterations, lam, sigma),
            ]
        )

        axes[0, col].imshow(noisy)
        axes[0, col].set_title(f"Noisy\n({noise_type.name})")
        axes[0, col].axis("off")

        axes[1, col].imshow(denoised)
        axes[1, col].set_title(f"Denoised\n({noise_type.name})")
        axes[1, col].axis("off")

    plt.tight_layout()

    if save:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # params
    VARIANCE = 0.05
    DT = 0.067
    MAX_ITER = 67
    LAM = 0.03
    SAVE = True
    SNP_DENSITY = 0.05
    SIGMA = 1.0

    # ┌─────────────────┬────────────┬───────────────────────────────────────────────────────────────┐
    # │ Parameter       │ Good Range │ Effect                                                        │
    # ├─────────────────┼────────────┼───────────────────────────────────────────────────────────────┤
    # │ VARIANCE        │ 0.01–0.10  │ How much Gaussian noise is added to the image before          │
    # │                 │            │ denoising. Higher = noisier input, harder problem.            │
    # │                 │            │ ↑ more noise   ↓ cleaner input                                │
    # ├─────────────────┼────────────┼───────────────────────────────────────────────────────────────┤
    # │ DT              │ 0.05–0.20  │ Time step size. Controls how large each diffusion update is.  │
    # │                 │            │ MUST stay <= 0.25 or the simulation blows up (goes unstable). │
    # │                 │            │ ↑ faster convergence but risk of instability                  │
    # │                 │            │ ↓ slower but safer                                            │
    # ├─────────────────┼────────────┼───────────────────────────────────────────────────────────────┤
    # │ MAX_ITER        │ 50–300     │ How many diffusion steps to run. More iterations = more       │
    # │                 │            │ smoothing applied overall.                                    │
    # │                 │            │ ↑ smoother result (can over-smooth / lose detail)             │
    # │                 │            │ ↓ less smoothing (noise may remain)                           │
    # ├─────────────────┼────────────┼───────────────────────────────────────────────────────────────┤
    # │ LAM             │ 0.01–0.05  │ Edge sensitivity threshold (lambda in Perona-Malik).          │
    # │                 │            │ Gradients larger than LAM are treated as edges and protected. │
    # │                 │            │ Gradients smaller than LAM are treated as noise and smoothed. │
    # │                 │            │ ↑ more smoothing everywhere, edges less protected             │
    # │                 │            │ ↓ only very flat regions smoothed, edges strongly preserved   │
    # ├─────────────────┼────────────┼───────────────────────────────────────────────────────────────┤
    # │ SNP_DENSITY     │ 0.01–0.10  │ Salt-and-pepper only. Fraction of pixels corrupted.           │
    # │                 │            │ ↑ more corrupted pixels  ↓ fewer                              │
    # ├─────────────────┼────────────┼───────────────────────────────────────────────────────────────┤
    # │ SIGMA           │ 0.5–2.0    │ Gaussian pre-smooth scale for structure tensor estimation.    │
    # │                 │            │ ↑ smoother gradient estimate  ↓ more sensitive to fine detail │
    # └─────────────────┴────────────┴───────────────────────────────────────────────────────────────┘

    max_stable = 0.25
    assert max_stable >= DT, "Stability condition: dt must be <= 0.25"

    img_path = "../../imgs/Lenna_(test_image).png"
    out_path = "out_eed.png"

    main(
        img_path,
        variance=VARIANCE,
        dt=DT,
        num_iterations=MAX_ITER,
        lam=LAM,
        color_noise=True,
        save=SAVE,
        output_path=out_path,
        snp_density=SNP_DENSITY,
        sigma=SIGMA,
    )
