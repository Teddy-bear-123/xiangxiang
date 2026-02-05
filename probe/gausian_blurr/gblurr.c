#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#define FPS 144
#define SCREENWIDTH 600
#define SCREENHEIGHT 800

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
float rand_gausian(float mean, float stddev){
    static int has_spare = 0;
    static float spare;
}

int main(void) {
    InitWindow(SCREENWIDTH, SCREENHEIGHT, "Ray ray");
    // ToggleFullscreen();
    SetTargetFPS(FPS);

    Texture2D image = LoadTexture("../../imgs/Lenna_(test_image).png");

    if (image.id == 0) {
        printf("Error\n");
        return (1);
    }

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(DARKPURPLE);

        DrawTexture(image, center_texture(image).x, center_texture(image).y,
                    WHITE);

        const char* label = "Original";
        int fontSize = 20;

        Vector2 textPos = center_text_on_texture(
            label, image, center_texture(image), fontSize, 10);

        DrawText(label, textPos.x, textPos.y, fontSize, WHITE);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
