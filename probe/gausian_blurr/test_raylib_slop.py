import os

import numpy as np
from PIL import Image as PILImage
from raylib import *


def load_img(path: str) -> np.ndarray:
    """Load image and convert to normalized float array"""
    if os.path.exists(path):
        img = PILImage.open(path).convert("RGB")
        img_array = np.array(img) / 255.0
        return img_array
    else:
        print(f"Error: Image not found at {path}")
        exit(1)


def gen_noise_map(shape: tuple[int, int], variance: float) -> np.ndarray:
    """Generate Gaussian noise map with given variance"""
    stddev = np.sqrt(variance)
    noise = np.random.normal(0, stddev, shape)
    return noise


def add_noise_to_img(noise_map: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Add noise to image and clip to valid range"""
    noisy_img = img + noise_map
    return np.clip(noisy_img, 0, 1)


def create_texture_from_rgb(
    dataR: np.ndarray, dataG: np.ndarray, dataB: np.ndarray, width: int, height: int
) -> Texture2D:
    """Create raylib texture from separate RGB channel data"""
    # Create empty texture
    image = GenImageColor(width, height, WHITE)
    texture = LoadTextureFromImage(image)
    UnloadImage(image)

    # Now update it with our data
    update_texture_rgb(texture, dataR, dataG, dataB, width, height)
    return texture


def update_texture_rgb(
    tex: Texture2D,
    dataR: np.ndarray,
    dataG: np.ndarray,
    dataB: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Update raylib texture from RGB channel data"""
    pixels = np.zeros(width * height * 4, dtype=np.uint8)

    for i in range(width * height):
        pixels[i * 4 + 0] = int(np.clip(dataR[i] * 255.0, 0, 255))  # R
        pixels[i * 4 + 1] = int(np.clip(dataG[i] * 255.0, 0, 255))  # G
        pixels[i * 4 + 2] = int(np.clip(dataB[i] * 255.0, 0, 255))  # B
        pixels[i * 4 + 3] = 255  # A

    # Use ffi.from_buffer to get proper pointer for raylib
    from raylib import ffi

    UpdateTexture(tex, ffi.from_buffer("unsigned char[]", pixels))


def heat_eq(
    U: np.ndarray, U_new: np.ndarray, img_h: int, img_w: int, dt: float
) -> None:
    """
    Apply one step of heat equation diffusion (in-place update of U_new)
    U: current state (flattened image)
    U_new: next state (will be modified)
    img_h, img_w: image dimensions
    dt: time step
    """
    r = dt

    # Copy current state to new state first
    U_new[:] = U[:]

    # Update interior points using finite difference
    for i in range(1, img_h - 1):
        for j in range(1, img_w - 1):
            idx = i * img_w + j
            up = (i + 1) * img_w + j
            down = (i - 1) * img_w + j
            left = i * img_w + (j - 1)
            right = i * img_w + (j + 1)

            U_new[idx] = U[idx] + r * (
                U[up] + U[down] + U[right] + U[left] - 4 * U[idx]
            )

    # Apply Neumann boundary conditions (zero derivative at boundaries)
    # This makes boundaries "perpendicular" - no heat flow across boundary
    for j in range(img_w):
        U_new[0 * img_w + j] = U_new[1 * img_w + j]  # Top
        U_new[(img_h - 1) * img_w + j] = U_new[(img_h - 2) * img_w + j]  # Bottom

    for i in range(img_h):
        U_new[i * img_w + 0] = U_new[i * img_w + 1]  # Left
        U_new[i * img_w + (img_w - 1)] = U_new[i * img_w + (img_w - 2)]  # Right


def main(
    path: str, variance: float, dt: float, max_iter: int, use_color_noise: bool = True
):
    """
    Main denoising application

    Args:
        path: Path to input image
        variance: Noise variance
        dt: Time step for heat equation
        max_iter: Maximum iterations
        use_color_noise: If True, use separate noise for each channel (color noise)
                        If False, use same noise for all channels (grayscale noise)
    """
    # Load image
    image = load_img(path)
    img_h, img_w = image.shape[:2]

    # Extract RGB channels
    original_R = image[:, :, 0].copy()
    original_G = image[:, :, 1].copy()
    original_B = image[:, :, 2].copy()

    # Generate noise
    if use_color_noise:
        # Separate noise for each channel (like C version)
        noise_map_R = gen_noise_map((img_h, img_w), variance)
        noise_map_G = gen_noise_map((img_h, img_w), variance)
        noise_map_B = gen_noise_map((img_h, img_w), variance)
    else:
        # Same noise for all channels (grayscale noise)
        noise_map = gen_noise_map((img_h, img_w), variance)
        noise_map_R = noise_map_G = noise_map_B = noise_map

    # Add noise to each channel
    noisy_R = add_noise_to_img(noise_map_R, original_R)
    noisy_G = add_noise_to_img(noise_map_G, original_G)
    noisy_B = add_noise_to_img(noise_map_B, original_B)

    # Flatten for processing (U is the "heat" we want to diffuse)
    UR = noisy_R.flatten().copy()
    UG = noisy_G.flatten().copy()
    UB = noisy_B.flatten().copy()

    # Also prepare noise map for diffusion (use same noise for visualization)
    if use_color_noise:
        # Use average of RGB noise for visualization
        noise_map_vis = (noise_map_R + noise_map_G + noise_map_B) / 3.0
    else:
        noise_map_vis = noise_map

    # Normalize noise to [0, 1] for display (noise is centered at 0)
    noise_display = noise_map_vis.copy()
    noise_display = (noise_display - noise_display.min()) / (
        noise_display.max() - noise_display.min()
    )

    # Flatten noise for diffusion
    U_noise = noise_display.flatten().copy()

    # Allocate buffers for next state
    UR_new = np.zeros_like(UR)
    UG_new = np.zeros_like(UG)
    UB_new = np.zeros_like(UB)
    U_noise_new = np.zeros_like(U_noise)

    # Initialize window
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"Heat Equation Image Denoising")

    # Create textures for all 5 images
    tex_original = create_texture_from_rgb(
        original_R.flatten(), original_G.flatten(), original_B.flatten(), img_w, img_h
    )
    tex_noisy = create_texture_from_rgb(
        noisy_R.flatten(), noisy_G.flatten(), noisy_B.flatten(), img_w, img_h
    )
    tex_denoised = create_texture_from_rgb(UR, UG, UB, img_w, img_h)

    # Noise map texture (grayscale)
    tex_noise = create_texture_from_rgb(U_noise, U_noise, U_noise, img_w, img_h)

    # Denoised noise texture (grayscale)
    tex_noise_denoised = create_texture_from_rgb(
        U_noise, U_noise, U_noise, img_w, img_h
    )

    iteration = 0
    SetTargetFPS(144)

    while not WindowShouldClose():
        # Controls
        if IsKeyPressed(KEY_SPACE):
            if iteration < max_iter:
                # Apply one step of heat equation to each channel
                heat_eq(UR, UR_new, img_h, img_w, dt)
                heat_eq(UG, UG_new, img_h, img_w, dt)
                heat_eq(UB, UB_new, img_h, img_w, dt)

                # Also diffuse the noise map
                heat_eq(U_noise, U_noise_new, img_h, img_w, dt)

                # Swap buffers
                UR, UR_new = UR_new, UR
                UG, UG_new = UG_new, UG
                UB, UB_new = UB_new, UB
                U_noise, U_noise_new = U_noise_new, U_noise

                # Update textures
                update_texture_rgb(tex_denoised, UR, UG, UB, img_w, img_h)
                update_texture_rgb(
                    tex_noise_denoised, U_noise, U_noise, U_noise, img_w, img_h
                )
                iteration += 1

        if IsKeyPressed(KEY_R):
            # Reset to noisy image
            UR[:] = noisy_R.flatten()
            UG[:] = noisy_G.flatten()
            UB[:] = noisy_B.flatten()
            U_noise[:] = noise_display.flatten()

            update_texture_rgb(tex_denoised, UR, UG, UB, img_w, img_h)
            update_texture_rgb(
                tex_noise_denoised, U_noise, U_noise, U_noise, img_w, img_h
            )
            iteration = 0

        # Drawing
        BeginDrawing()
        ClearBackground(DARKPURPLE)

        # Calculate layout for 5 images
        spacing = 15
        img_display_w = (SCREEN_WIDTH - 6 * spacing) // 5
        scale = max(1, img_display_w // img_w)

        scaled_w = img_w * scale
        scaled_h = img_h * scale
        start_y = (SCREEN_HEIGHT - scaled_h) // 2

        # Draw five images side by side
        x_pos = spacing

        # 1. Original
        DrawTextureEx(tex_original, (x_pos, start_y), 0.0, scale, WHITE)
        DrawText(b"Original", x_pos, start_y - 30, 20, WHITE)
        x_pos += scaled_w + spacing

        # 2. Noisy
        DrawTextureEx(tex_noisy, (x_pos, start_y), 0.0, scale, WHITE)
        DrawText(f"Noisy (var={variance:.3f})".encode(), x_pos, start_y - 30, 20, WHITE)
        x_pos += scaled_w + spacing

        # 3. Denoised
        DrawTextureEx(tex_denoised, (x_pos, start_y), 0.0, scale, WHITE)
        DrawText(
            f"Denoised (iter={iteration})".encode(), x_pos, start_y - 30, 20, WHITE
        )
        x_pos += scaled_w + spacing

        # 4. Noise map
        DrawTextureEx(tex_noise, (x_pos, start_y), 0.0, scale, WHITE)
        DrawText(b"Noise Map", x_pos, start_y - 30, 20, WHITE)
        x_pos += scaled_w + spacing

        # 5. Diffused noise
        DrawTextureEx(tex_noise_denoised, (x_pos, start_y), 0.0, scale, WHITE)
        DrawText(b"Diffused Noise", x_pos, start_y - 30, 20, WHITE)

        # Instructions
        DrawText(b"Press SPACE to iterate | Press R to reset", 10, 10, 20, YELLOW)
        DrawText(f"dt = {dt:.2f} | max_iter = {max_iter}".encode(), 10, 40, 20, WHITE)
        DrawText(
            f"Noise type: {'Color (RGB)' if use_color_noise else 'Grayscale'}".encode(),
            10,
            70,
            20,
            WHITE,
        )

        EndDrawing()

    # Cleanup
    UnloadTexture(tex_original)
    UnloadTexture(tex_noisy)
    UnloadTexture(tex_denoised)
    UnloadTexture(tex_noise)
    UnloadTexture(tex_noise_denoised)
    CloseWindow()


if __name__ == "__main__":
    # Parameters
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    VARIANCE = 0.05
    DT = 0.1
    MAX_ITER = 100

    # Path to your image
    img_path = "../../imgs/Lenna_(test_image).png"

    # Run with color noise (separate for each channel, like C version)
    # Set use_color_noise=False for grayscale noise (same for all channels)
    main(img_path, VARIANCE, DT, MAX_ITER, use_color_noise=True)
