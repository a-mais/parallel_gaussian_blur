#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "gaussian_blur.h"
#include "gaussian_blur_parallel.h"

Image *readPPM(const char *filename) {
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
        exit(1);
    }

    fscanf(fp, "%d %d %d", &img->width, &img->height, &img->max);
    fgetc(fp);
    img->data = (unsigned char *) malloc(3 * img->width * img->height);
    fread(img->data, 3, img->width * img->height, fp);
    fclose(fp);
    return img;
}

void writePPM(const char *filename, Image *img) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n%d\n", img->width, img->height, img->max);
    fwrite(img->data, 3, img->width * img->height, fp);
    fclose(fp);
}

int main() {
    const char *inputPath = "../input/image.ppm";
    const char *outputSeq = "../output/output_sequential.ppm";
    const char *outputPar = "../output/output_parallel.ppm";

    int kernelSize = 31;     // maior kernel == blur mais forte
    double sigma = 46.0;      // intensidade do efeito

    Image *img = readPPM(inputPath);

    Image *seq = (Image *) malloc(sizeof(Image));
    Image *par = (Image *) malloc(sizeof(Image));
    seq->width = par->width = img->width;
    seq->height = par->height = img->height;
    seq->max = par->max = img->max;
    seq->data = (unsigned char *) malloc(3 * img->width * img->height);
    par->data = (unsigned char *) malloc(3 * img->width * img->height);

    double start, end, time_seq, time_par;

    printf("Aplicando desfoque Gaussiano sequencial...\n");
    start = omp_get_wtime();
    gaussianBlurSequential(img, seq, kernelSize, sigma);
    end = omp_get_wtime();
    time_seq = end - start;

    printf("Aplicando desfoque Gaussiano paralelo...\n");
    start = omp_get_wtime();
    gaussianBlurParallel(img, par, kernelSize, sigma);
    end = omp_get_wtime();
    time_par = end - start;

    double speedup = time_seq / time_par;

    printf("\nTempo Sequencial: %.4f s\n", time_seq);
    printf("Tempo Paralelo:   %.4f s\n", time_par);
    printf("Speedup: %.2fx\n", speedup);

    writePPM(outputSeq, seq);
    writePPM(outputPar, par);

    free(img->data);
    free(seq->data);
    free(par->data);
    free(img);
    free(seq);
    free(par);

    return 0;
}
