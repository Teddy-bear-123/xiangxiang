import os
import sys
from enum import Enum, auto
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# from matplotlib.animation import FFMpegWriter, FuncAnimation
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
    GAUSSIAN = auto()  # Normal distribution added to pixel values
    ADDITIVE = auto()  # Uniform distribution added to pixel values
    MULTIPLICATIVE = auto()  # Noise scaled by pixel intensity (speckle/sensor noise)
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
        # Bright pixels get more absolute noise than dark ones
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


def diffusivity(grad_sq: np.ndarray, lam: float) -> np.ndarray:
    """
    1 / (1 + s/λ)
    """
    return 1.0 / (1.0 + grad_sq / lam)


def common_pixels_by_shift(
    img: np.ndarray, row_shift: int, col_shift: int
) -> np.ndarray:
    """
     Returns a copy of img where every pixel (i, j) holds the value
    that was at (i + row_shift, j + col_shift) in the original.

    Out-of-bounds indices are clamped to the border (Neumann BC).
    """

    h, w = img.shape

    row_indices = np.clip(np.arange(h) + row_shift, 0, h - 1)
    col_indices = np.clip(np.arange(w) + col_shift, 0, w - 1)
    return img[np.ix_(row_indices, col_indices)]


def gradient_mag_sq(channel: np.ndarray) -> np.ndarray:
    """
    Finds | grad(U) |^2 for each direction

    dU/di ≈ (U[i+1, j] - U[i-1, j]) / 2      (vertical)
    dU/dj ≈ (U[i, j+1] - U[i, j-1]) / 2      (horizontal)

    |grad U|^2 = (dU/di)^2 + (dU/dj)^2

    """
    south_neighbour = common_pixels_by_shift(channel, 1, 0)  # U[i+1, j]
    north_neighbour = common_pixels_by_shift(channel, -1, 0)  # U[i-1, j]
    east_neighbour = common_pixels_by_shift(channel, 0, 1)  # U[i, j+1]
    west_neighbour = common_pixels_by_shift(channel, 0, -1)  # U[i, j-1]

    vertical_diff = (south_neighbour - north_neighbour) / 2.0  # dU/di
    horizontal_diff = (east_neighbour - west_neighbour) / 2.0  # dU/dj

    return vertical_diff**2 + horizontal_diff**2


def anisotropic_diffusion_step(
    channel: np.ndarray, dt: float, edge_threshold: float
) -> np.ndarray:
    """
    One explicit time step of:
      dU/dt = div( c(|grad U|^2) * grad U )

    Discretised using the neighbour-based update from Perona & Malik:
      U_new = U + 0.5 * dt * [
          (c + c_south) * (south - U)
        - (c + c_north) * (U - north)
        + (c + c_east)  * (east  - U)
        - (c + c_west)  * (U - west)
      ]

    The (c + c_neighbour) / 2 terms are edge weights on each "face"
    between the pixel and its neighbour — if either side is an edge,
    diffusion is reduced.
    """
    south = common_pixels_by_shift(channel, 1, 0)
    north = common_pixels_by_shift(channel, -1, 0)
    east = common_pixels_by_shift(channel, 0, 1)
    west = common_pixels_by_shift(channel, 0, -1)

    grad_sq = gradient_mag_sq(channel)
    c = diffusivity(grad_sq, edge_threshold)

    c_south = common_pixels_by_shift(c, 1, 0)
    c_north = common_pixels_by_shift(c, -1, 0)
    c_east = common_pixels_by_shift(c, 0, 1)
    c_west = common_pixels_by_shift(c, 0, -1)

    diffusion = (
        (c + c_south) * (south - channel)
        - (c + c_north) * (channel - north)
        + (c + c_east) * (east - channel)
        - (c + c_west) * (channel - west)
    )

    return np.clip(channel + 0.5 * dt * diffusion, 0.0, 1.0)


def denoise_channel(
    channel: np.ndarray, dt: float, num_iterations: int, edge_threshold: float
) -> np.ndarray:
    result = channel.copy()
    for _ in range(num_iterations):
        result = anisotropic_diffusion_step(result, dt, edge_threshold)
    return result


def main(  # noqa
    path: str,
    variance: float = 0.05,
    dt: float = 0.1,
    num_iterations: int = 100,
    lam: float = 0.003,
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
        f"Perona-Malik — iter={num_iterations}  dt={dt}  λ={lam}",
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

        nr, ng, nb = noisy[:, :, 0], noisy[:, :, 1], noisy[:, :, 2]
        denoised = np.dstack(
            [
                denoise_channel(nr, dt, num_iterations, lam),
                denoise_channel(ng, dt, num_iterations, lam),
                denoise_channel(nb, dt, num_iterations, lam),
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
    # └─────────────────┴────────────┴───────────────────────────────────────────────────────────────┘

    max_stable = 0.25
    assert max_stable >= DT, "Stability condition: dt must be <= 0.25"

    img_path = "../../imgs/Lenna_(test_image).png"
    out_path = "out_perona_malik.png"

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
    )
