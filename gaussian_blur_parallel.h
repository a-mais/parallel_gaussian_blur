#include "gaussian_blur.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void gaussianBlurMPI(Image *input, Image *output, int kernelSize, double sigma, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size == 1) {
        gaussianBlurParallel(input, output, kernelSize, sigma);
        return;
    }

    int half = kernelSize / 2;
    double **kernel = generateGaussianKernel(kernelSize, sigma);

    // Broadcast dos parâmetros
    int width = input->width;
    int height = input->height;
    int max = input->max;

    MPI_Bcast(&width, 1, MPI_INT, 0, comm);
    MPI_Bcast(&height, 1, MPI_INT, 0, comm);
    MPI_Bcast(&max, 1, MPI_INT, 0, comm);
    MPI_Bcast(&kernelSize, 1, MPI_INT, 0, comm);
    MPI_Bcast(&sigma, 1, MPI_DOUBLE, 0, comm);

    // Distribuição de linhas
    int *local_rows = (int *)malloc(size * sizeof(int));
    MPI_Gather(&height, 1, MPI_INT, local_rows, 1, MPI_INT, 0, comm);
    if (rank != 0) {
        free(local_rows);
        local_rows = NULL;
    }

    int my_rows = height / size + (rank < height % size);
    int local_input_size = (my_rows + 2 * half) * width * 3;

    unsigned char *local_input = (unsigned char *)malloc(local_input_size);
    unsigned char *local_output = (unsigned char *)malloc(my_rows * width * 3);

    // Cálculo scatter/gather arrays (apenas rank 0)
    int *sendcounts = NULL, *displs = NULL;
    int *recvcounts = NULL, *recvdispls = NULL;

    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        recvcounts = (int *)malloc(size * sizeof(int));
        recvdispls = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = height / size + (i < height % size);
            sendcounts[i] = (rows + 2 * half) * width * 3;
            displs[i] = offset;
            offset += sendcounts[i];

            recvcounts[i] = rows * width * 3;
            recvdispls[i] = 0;
            if (i > 0) recvdispls[i] = recvdispls[i-1] + recvcounts[i-1];
        }
    }

    // Scatter com halo
    MPI_Scatterv(input->data, sendcounts, displs, MPI_UNSIGNED_CHAR,
                 local_input, local_input_size, MPI_UNSIGNED_CHAR, 0, comm);

    // Processamento híbrido MPI+OpenMP
#pragma omp parallel for collapse(2)
    for (int ly = half; ly < my_rows + half; ly++) {
        for (int x = half; x < width - half; x++) {
            double r = 0, g = 0, b = 0;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int y = ly + ky;
                    int px = x + kx;
                    if (px < 0 || px >= width) continue;

                    int pos = (y * width + px) * 3;
                    double k = kernel[ky + half][kx + half];
                    r += local_input[pos] * k;
                    g += local_input[pos + 1] * k;
                    b += local_input[pos + 2] * k;
                }
            }
            int outPos = ((ly - half) * width + x) * 3;
            local_output[outPos] = (unsigned char)r;
            local_output[outPos + 1] = (unsigned char)g;
            local_output[outPos + 2] = (unsigned char)b;
        }
    }

    // Gather das linhas válidas
    MPI_Gatherv(local_output, my_rows * width * 3, MPI_UNSIGNED_CHAR,
                output->data, recvcounts, recvdispls, MPI_UNSIGNED_CHAR, 0, comm);

    // Cleanup
    free(local_input);
    free(local_output);
    if (local_rows) free(local_rows);

    if (rank == 0) {
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(recvdispls);
    }

    for (int i = 0; i < kernelSize; i++)
        free(kernel[i]);
    free(kernel);
}
