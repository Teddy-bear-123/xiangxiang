import os
import sys
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.image import AxesImage
from PIL import Image


def load_img(path: str) -> np.ndarray:
    if os.path.exists(path):
        img = Image.open(path).convert("RGB")
        img_array = np.array(img) / 255.0
        return img_array
    else:
        print("Err")
        sys.exit(1)


def gen_noise_map(shape: tuple[int, int], varience: float) -> np.ndarray:
    stddev = np.sqrt(varience)
    noise = np.random.normal(0, stddev, shape)

    return noise


def add_noise_to_img(
    noise_map: npt.NDArray[np.floating[Any]], img: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:  # weird way to fix bugs
    noisy_img = img + noise_map
    clipped: npt.NDArray[np.floating[Any]] = np.clip(noisy_img, 0, 1)
    return clipped


def heat_eq(
    U: np.ndarray, U_new: np.ndarray, img_h: int, img_w: int, dt: float
) -> None:
    """
    U: U(x, y, z, t) -> I.e The image or the "Heat" at point x, y, z at time t that we want to diffuse

    M, N => img_h, img_w

    Returns: Unew -> The "diffused" "Heat" at time t+dt
    """

    r = dt
    np.copyto(U_new, U)

    # Neumans
    # U[0, j] = U[1, j]
    # U[-1, j] = U[-2, j]
    # We want to make the top, bottom, left and right become "perpendicual" i.e boundary point = interior neigbour and edges are pependicualr

    for j in range(img_w):
        U_new[0 * img_w + j] = U[1 * img_w + j]
        U_new[(img_h - 1) * img_w + j] = U[(img_h - 2) * img_w + j]

    for i in range(img_h):
        U_new[i * img_w + 0] = U[i * img_w + 1]
        U_new[i * img_w + (img_w - 1)] = U[i * img_w + (img_w - 2)]

    # dirichlet

    # for j in range(img_w):
    #     U_new[0*img_w + j] = 0.0
    #     U_new[(img_h - 1) * img_w + j] = 0.0
    #
    # for i in range(img_h):
    #     U_new[i * img_w + 0] = 0.0
    #     U_new[i * img_w + (img_w - 1)] = 0.0

    for i in range(1, img_h - 1):
        for j in range(1, img_w - 1):
            idx = i * img_w + j  # row order unrwaping
            down = (i + 1) * img_w + j
            up = (i - 1) * img_w + j
            left = i * img_w + (j - 1)
            right = i * img_w + (j + 1)

            U_new[idx] = U[idx] + r * (
                U[up] + U[down] + U[right] + U[left] - 4 * U[idx]
            )


class AnimationState:
    """Class to hold animation state"""

    def __init__(  # noqa
        self,
        UR: np.ndarray,
        UG: np.ndarray,
        UB: np.ndarray,
        U_noise: np.ndarray,
        img_h: int,
        img_w: int,
        dt: float,
        max_iter: int,
        tick: int,
    ):
        self.UR = UR
        self.UG = UG
        self.UB = UB
        self.U_noise = U_noise

        self.UR_new = np.zeros_like(UR)
        self.UG_new = np.zeros_like(UG)
        self.UB_new = np.zeros_like(UB)
        self.U_noise_new = np.zeros_like(U_noise)

        self.img_h = img_h
        self.img_w = img_w
        self.dt = dt
        self.max_iter = max_iter
        self.tick = tick
        self.iteration = 0


def update_animation(
    frame: int,  # noqa
    state: AnimationState,
    im_denoised: AxesImage,
    im_noise_denoised: AxesImage,
    axes: np.ndarray,
) -> tuple[AxesImage, AxesImage]:

    if state.iteration >= state.max_iter:
        return im_denoised, im_noise_denoised

    for _ in range(state.tick):
        if state.iteration >= state.max_iter:
            break

        heat_eq(state.UR, state.UR_new, state.img_h, state.img_w, state.dt)
        heat_eq(state.UG, state.UG_new, state.img_h, state.img_w, state.dt)
        heat_eq(state.UB, state.UB_new, state.img_h, state.img_w, state.dt)
        heat_eq(state.U_noise, state.U_noise_new, state.img_h, state.img_w, state.dt)

        state.UR, state.UR_new = state.UR_new, state.UR
        state.UG, state.UG_new = state.UG_new, state.UG
        state.UB, state.UB_new = state.UB_new, state.UB
        state.U_noise, state.U_noise_new = state.U_noise_new, state.U_noise

        state.iteration += 1

    denoised_R = state.UR.reshape(state.img_h, state.img_w)
    denoised_G = state.UG.reshape(state.img_h, state.img_w)
    denoised_B = state.UB.reshape(state.img_h, state.img_w)
    denoised_img = np.dstack([denoised_R, denoised_G, denoised_B])
    denoised_img = np.clip(denoised_img, 0, 1)

    noise_denoised = state.U_noise.reshape(state.img_h, state.img_w)

    im_denoised.set_data(denoised_img)
    axes[2].set_title(f"Denoised (iter={state.iteration})")

    im_noise_denoised.set_data(noise_denoised)

    return im_denoised, im_noise_denoised


def main(  # noqa
    path: str,
    varience: float,
    dt: float,
    max_iter: int,
    color_noise: bool = False,
    tick: int = 1,
    interval: int = 50,
) -> None:
    image = load_img(path)
    img_h, img_w = image.shape[:2]

    orgiginal_R = image[:, :, 0].copy()
    orgiginal_G = image[:, :, 1].copy()
    orgiginal_B = image[:, :, 2].copy()

    if color_noise:
        noise_map_R = gen_noise_map((img_h, img_w), varience)
        noise_map_G = gen_noise_map((img_h, img_w), varience)
        noise_map_B = gen_noise_map((img_h, img_w), varience)
        noise_map_vis = (noise_map_R + noise_map_G + noise_map_B) / 3.0
    else:
        noise_map = gen_noise_map((img_h, img_w), varience)
        noise_map_R = noise_map_G = noise_map_B = noise_map
        noise_map_vis = noise_map

    noisyR = add_noise_to_img(noise_map_R, orgiginal_R)
    noisyG = add_noise_to_img(noise_map_G, orgiginal_G)
    noisyB = add_noise_to_img(noise_map_B, orgiginal_B)

    # U(x, y, t)? or is it (x, y, z ,t)
    UR = noisyR.flatten().copy()
    UG = noisyG.flatten().copy()
    UB = noisyB.flatten().copy()

    noise_display = noise_map_vis.copy()
    den = noise_display.max() - noise_display.min()
    if den > 0:
        noise_display = (noise_display - noise_display.min()) / (
            noise_display.max() - noise_display.min()
        )
    else:
        noise_display.fill(0.0)

    U_noise = noise_display.flatten().copy()

    # UR_new = np.zeros_like(UR)
    # UG_new = np.zeros_like(UG)
    # UB_new = np.zeros_like(UB)
    # noise_U_new = np.zeros_like(U_noise)  # vicual

    original_img = np.dstack([orgiginal_R, orgiginal_G, orgiginal_B])
    noisy_img = np.dstack([noisyR, noisyG, noisyB])

    state = AnimationState(UR, UG, UB, U_noise, img_h, img_w, dt, max_iter, tick)
    try:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes = cast(np.ndarray, axes)
    except Exception:
        sys.exit(1)

    fig.suptitle("Diffusion of 'Heat' ", fontsize=16, fontweight="bold")

    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy_img)
    axes[1].set_title("Noisy Imag")
    axes[1].axis("off")

    denoised_img = noisy_img.copy()
    im_denoised = axes[2].imshow(denoised_img)
    axes[2].set_title("Denoised image")
    axes[2].axis("off")

    axes[3].imshow(noise_display, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Noise Map")
    axes[3].axis("off")

    im_noise_denoised = axes[4].imshow(noise_display, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("Diffused Noise")
    axes[4].axis("off")

    plt.tight_layout()

    num_frames = (max_iter + tick - 1) // tick

    anim = FuncAnimation(
        fig,
        update_animation,
        fargs=(state, im_denoised, im_noise_denoised, axes),
        frames=num_frames,
        interval=interval,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )

    # plt.show()
    writer = FFMpegWriter(
        fps=1000 // interval,
        metadata=dict(artist="Heat Diffusion"),  # noqa
        bitrate=2400,
    )
    anim.save("heat_dif.mp4", writer=writer)


if __name__ == "__main__":
    # params
    VARIENCE = 0.05
    DT = 0.1
    MAX_ITER = 100
    TICK = 2  # number of steps per frame
    INTERVAL = 50

    max_stable = 0.25
    assert max_stable >= DT
    img_path = "../../imgs/Lenna_(test_image).png"

    main(
        img_path,
        color_noise=True,
        dt=DT,
        interval=50,
        varience=VARIENCE,
        max_iter=MAX_ITER,
        tick=TICK,
    )
