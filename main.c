#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gaussian_blur.h"

Image *read_ppm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Erro ao abrir imagem");
        exit(1);
    }

    Image *img = (Image *) malloc(sizeof(Image));
    char format[3];
    fscanf(fp, "%s", format);
    if (strcmp(format, "P6") != 0) {
        fprintf(stderr, "Formato não suportado\n");
        fclose(fp);
        exit(1);
    }

    fscanf(fp, "%d %d %d", &img->width, &img->height, &img->max);
    fgetc(fp);
    img->data = (unsigned char *) malloc(3 * img->width * img->height);
    fread(img->data, 3, img->width * img->height, fp);
    fclose(fp);
    return img;
}

void write_ppm(const char *filename, Image *img) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Erro ao salvar imagem");
        exit(1);
    }
    fprintf(fp, "P6\n%d %d\n%d\n", img->width, img->height, img->max);
    fwrite(img->data, 3, img->width * img->height, fp);
    fclose(fp);
}

int main() {
    const char *input_path = "../input/image.ppm";
    int kernel_size = 61; // maior kernel == blur mais forte
    double sigma = 100.0; // intensidade do efeito

    Image *img = read_ppm(input_path);

    Image *seq = (Image *) malloc(sizeof(Image));
    seq->width = img->width;
    seq->height = img->height;
    seq->max = img->max;
    seq->data = (unsigned char *) malloc(3 * img->width * img->height);

    struct timespec start, finish;
    double time_seq;

    printf("Aplicando desfoque Gaussiano sequencial...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    gaussian_blur_sequential(img, seq, kernel_size, sigma);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    double elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("Tempo decorrido: %.9f segundos\n", elapsed);

    free(img->data);
    free(seq->data);
    free(img);
    free(seq);

    return 0;
}
