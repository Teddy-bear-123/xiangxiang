#ifndef UTILS_H
#define UTILS_H

#include "raylib.h"

typedef struct {
    float* R;
    float* G;
    float* B;
    int width;
    int height;
    int channels;
    int is_grayscale;
} ImageData;

ImageData create_image(int width, int height, int is_grayscale);
void free_image(ImageData* img);
void extract_channels(Image img, ImageData* out);
void copy_image(ImageData* src, ImageData* dst);

Texture2D create_texture_from_image(ImageData* img);
Texture2D create_texture_from_noise(float* noise_map, int width, int height);
void update_texture_from_image(Texture2D* tex, ImageData* img);

float clamp_float(float value, float min, float max);
void copy_float_array(float* src, float* dst, int size);

float* generate_noise_map(int width, int height, int depth, float variance);
void add_bg_noise_image(ImageData* img, float* noise_map);
void add_color_noise_image(ImageData* img, float* noise_R, float* noise_G, float* noise_B);



typedef struct {
    Rectangle dest_rect;
    Vector2 label_pos;
} ImageSlot;

typedef struct {
    ImageSlot* slots;
    int num_slots;
    int cols;
    int rows;
    float scale;
    int scaled_width;
    int scaled_height;
    int padding;
    int label_height;
} GridLayout;

GridLayout calculate_grid_layout(int screen_width, int screen_height,
                                 int img_width, int img_height, int num_images);
void free_grid_layout(GridLayout* layout);
void draw_image_grid(Texture2D* textures, const char** labels,
                     GridLayout* layout, int img_width, int img_height);

#endif // UTILS_H
