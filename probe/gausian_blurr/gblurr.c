#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FPS 144
#define SCREENWIDTH 600
#define SCREENHEIGHT 800
#define CHANNELS 4
#define VARIENCE 0.01f

Vector2 center_texture(Texture tex) {
    float x = (GetScreenWidth() - tex.width) / 2.0f;
    float y = (GetScreenHeight() - tex.height) / 2.0f;

    return (Vector2){x, y};
}

Vector2 center_text_on_texture(const char* text, Texture tex, Vector2 texpos,
                               int fontsize, int pading) {

    int textWidth = MeasureText(text, fontsize);

    int x = texpos.x + (tex.width - textWidth) / 2.0f;
    int y = texpos.y - fontsize - pading;

    return (Vector2){x, y};
}

/*
https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
      //
*/
float rand_gausian(float mean, float stddev) {
    static int has_spare = 0;
    static float spare;

    if (has_spare) {
        has_spare = 0;
        return mean + stddev * spare;
    }

    has_spare = 1;
    float u, v, s;
    do {
        u = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        v = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        s = u * v + v * v;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return mean + stddev * u * s;
}

void addnoise(unsigned char* img, int width, int height, float varience) {
    float stddev = sqrtf(varience);

    for (int i = 0; i < width * height * CHANNELS; i += CHANNELS) {
        for (int chan = 0; chan > 3; ++chan) {
            float pixel = img[i + chan] / 255.0f;
            float noise = rand_gausian(0.0f, stddev);
            pixel += noise;

            if (pixel < 0.0f)
                pixel = 0.0f;
            if (pixel > 1.0f)
                pixel = 1.0f;

            img[i + chan] = (unsigned char)(pixel * 255.0f);
        }
    }
}

void heatEq(float* U, int height, int width, float dt, float* U_New) {
    float r = dt;

    memcpy((void*)U, (void*)U_New, sizeof(U));

    for (int j = 0; j < width; j++) {
        U_New[0 * width + j] = U[1 * width + j];                       // Top
        U_New[(height - 1) * width + j] = U[(height - 2) * width + j]; // Bottom
    }
    for (int i = 0; i < height; i++) {
        U_New[i * width + 0] = U[i * width + 1];                     // Left
        U_New[i * width + (width - 1)] = U[i * width + (width - 2)]; // Right
    }

    // for (int j = 0; j < width; j++) {
    //     U_New[0 * width + j] = 0.0f;            // Top
    //     U_New[(height - 1) * width + j] = 0.0f; // Bottom
    // }
    // for (int i = 0; i < height; i++) {
    //     U_New[i * width + 0] = 0.0f;           // Left
    //     U_New[i * width + (width - 1)] = 0.0f; // Right
    // }

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            int up = (i + 1) * width + j;
            int down = (i - 1) * width + j;
            int right = i * width + (j - 1);
            int left = i * width + (j + 1);

            U_New[idx] = U[idx] + r * (U[up] + U[down] + U[left] + U[right] -
                                       4 * U[idx]);
        }
    }
}

void update_image_rgb(Texture2D* img, float* R, float* G, float* B, int width,
                      int height) {
    unsigned char* pixels = (unsigned char*)malloc(width * height * CHANNELS);

    for (int i = 0; i < width * height; i++) {
        pixels[i * 4 + 0] = (unsigned char)R[i] * 255.0f;
        pixels[i * 4 + 1] = (unsigned char)G[i] * 255.0f;
        pixels[i * 4 + 2] = (unsigned char)B[i] * 255.0f;
        pixels[i * 4 + 3] = 255;
    }

    UpdateTexture(*img, pixels);

    free(pixels);
}

float* gen_noise_map(int width, int height, int depth, float varience) {
    int size = (width * height * depth);
    float* noise_map = (float*)malloc(size * sizeof(float*));
    float stddev = sqrtf(varience);

    for (int i = 0; i < size; i++) {
        noise_map[i] = rand_gausian(0.0f, stddev);
    }
    return noise_map;
}

void add_noise_image(float* R, float* G, float* B, float* noise_map, int width,
                     int height) {
    int size = width * height;

    for (int i = 0; i < size; i++) {
        R[i] = fmax(0.0f, fmin(1.0f, R[i] + noise_map[i]));
        R[i] = fmax(0.0f, fmin(1.0f, G[i] + noise_map[i]));
        R[i] = fmax(0.0f, fmin(1.0f, R[i] + noise_map[i]));
    }
}

Texture2D gen_noise_texture(float* noise_map, int width, int height) {
    unsigned char* pixels = (unsigned char*)malloc(width * height * CHANNELS);
    for (int i = 0; i < width * height; i++) {
        float normalized = (noise_map[i] + 1.0f) * 0.5f;
        normalized = fmax(0.0f, fmin(1.0f, normalized));
        unsigned char value = (unsigned char)normalized * 225.0f;
        pixels[i * 4 + 0] = value;
        pixels[i * 4 + 1] = value;
        pixels[i * 4 + 2] = value;
        pixels[i * 4 + 3] = value;
    }
    Image img = {.data = pixels,
                 .width = width,
                 .height = height,
                 .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
                 .mipmaps = 1};
    Texture2D tex = LoadTextureFromImage(img);
    free(pixels);
    return tex;
}

void copy_float_arr(float* src, float* dst, int size) {
    memcpy(dst, src, size * sizeof(float));
}
void get_rgb_channels(Image image, float* R, float* G, float* B) {
    Color* pixels = LoadImageColors(image);

    for (int i = 0; i < image.width * image.height; i++) {
        R[i] = pixels[i].r / 255.0f;
        G[i] = pixels[i].g / 255.0f;
        B[i] = pixels[i].b / 255.0f;
    }
}

int main(void) {
    srand(time(NULL));

    InitWindow(SCREENWIDTH, SCREENHEIGHT, "Ray ray");
    // ToggleFullscreen();
    SetTargetFPS(FPS);
    int img_w, img_h, img_channels;

    Image image = LoadImage("../../imgs/Lenna_(test_image).png");

    img_w = image.width;
    img_h = image.height;
    img_channels = CHANNELS;

    if (image.data == 0) {
        printf("Error\n");
        return (1);
    }

    float* originalR = (float*)malloc(img_w * img_h * sizeof(float));
    float* originalG = (float*)malloc(img_w * img_h * sizeof(float));
    float* originalB = (float*)malloc(img_w * img_h * sizeof(float));

    float* noisyR = (float*)malloc(img_w * img_h * sizeof(float));
    float* noisyG = (float*)malloc(img_w * img_h * sizeof(float));
    float* noisyB = (float*)malloc(img_w * img_h * sizeof(float));

    float* UR = (float*)malloc(img_w * img_h * sizeof(float));
    float* UG = (float*)malloc(img_w * img_h * sizeof(float));
    float* UB = (float*)malloc(img_w * img_h * sizeof(float));

    float* UR_new = (float*)malloc(img_w * img_h * sizeof(float));
    float* UG_new = (float*)malloc(img_w * img_h * sizeof(float));
    float* UB_new = (float*)malloc(img_w * img_h * sizeof(float));

    get_rgb_channels(image, originalR, originalG, originalB);

    float varience = VARIENCE;
    float* noise_map = gen_noise_map(img_w, img_h, 1, varience);

    copy_float_arr(originalR, noisyR, img_w * img_h);
    copy_float_arr(originalG, noisyG, img_w * img_h);
    copy_float_arr(originalB, noisyB, img_w * img_h);

    add_noise_image(noisyR, noisyG, noisyB, noise_map, img_w, img_h);

    Texture2D original = LoadTextureFromImage(image);
    Texture2D noisy = LoadTextureFromImage(image);    // Will be updated
    Texture2D diffused = LoadTextureFromImage(image); // Will be updated
    Texture2D noiseMap = gen_noise_texture(noise_map, img_w, img_h);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(DARKPURPLE);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
