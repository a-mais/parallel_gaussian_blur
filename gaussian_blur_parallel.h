#ifndef GAUSSIAN_BLUR_PARALLEL_H
#define GAUSSIAN_BLUR_PARALLEL_H

#include <omp.h>
#include "gaussian_blur.h"

void gaussianBlurParallel(Image *input, Image *output, int kernelSize, double sigma) {
    int width = input->width;
    int height = input->height;
    int half = kernelSize / 2;
    double **kernel = generateGaussianKernel(kernelSize, sigma);

#pragma omp parallel for collapse(2)
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
