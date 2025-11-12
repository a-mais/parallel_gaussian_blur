#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include <math.h>
#include <stdlib.h>

typedef struct {
    int width;
    int height;
    int max;
    unsigned char *data;
} Image;

// Função que gera kernel Gaussiano com tamanho e sigma configuráveis
static double **generateGaussianKernel(int size, double sigma) {
    int half = size / 2;
    double **kernel = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
        kernel[i] = (double *)malloc(size * sizeof(double));

    double sum = 0.0;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            double exponent = -(x * x + y * y) / (2 * sigma * sigma);
            kernel[y + half][x + half] = exp(exponent);
            sum += kernel[y + half][x + half];
        }
    }

    // Normalização
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            kernel[i][j] /= sum;

    return kernel;
}

void gaussianBlurSequential(Image *input, Image *output, int kernelSize, double sigma) {
    int width = input->width;
    int height = input->height;
    int half = kernelSize / 2;
    double **kernel = generateGaussianKernel(kernelSize, sigma);

    for (int y = half; y < height - half; y++) {
        for (int x = half; x < width - half; x++) {
            double r = 0, g = 0, b = 0;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int pos = ((y + ky) * width + (x + kx)) * 3;
                    double k = kernel[ky + half][kx + half];
                    r += input->data[pos] * k;
                    g += input->data[pos + 1] * k;
                    b += input->data[pos + 2] * k;
                }
            }
            int outPos = (y * width + x) * 3;
            output->data[outPos] = (unsigned char)r;
            output->data[outPos + 1] = (unsigned char)g;
            output->data[outPos + 2] = (unsigned char)b;
        }
    }

    for (int i = 0; i < kernelSize; i++)
        free(kernel[i]);
    free(kernel);
}

#endif
