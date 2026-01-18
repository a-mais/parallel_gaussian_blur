#ifndef GAUSSIAN_BLUR_PARALLEL_H
#define GAUSSIAN_BLUR_PARALLEL_H

#include <omp.h>
#include <mpi.h>
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

void gaussianBlurMPI(Image *input, Image *output, int kernelSize, double sigma, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Se só tiver 1 processo, usa versão sequencial
    if (size == 1) {
        if (rank == 0 && input != NULL) {
            gaussianBlurSequential(input, output, kernelSize, sigma);
        }
        return;
    }

    int half = kernelSize / 2;
    int width, height, max;

    // Rank 0 tem os dados
    if (rank == 0) {
        width = input->width;
        height = input->height;
        max = input->max;
    }

    // Broadcast dos parâmetros para todos os processos
    MPI_Bcast(&width, 1, MPI_INT, 0, comm);
    MPI_Bcast(&height, 1, MPI_INT, 0, comm);
    MPI_Bcast(&max, 1, MPI_INT, 0, comm);

    // Gerar kernel em todos os processos
    double **kernel = generateGaussianKernel(kernelSize, sigma);

    // Calcular distribuição de linhas
    int rows_per_proc = height / size;
    int remainder = height % size;
    int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int my_start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    // Alocar buffers locais (com halo para bordas)
    int local_height = my_rows + 2 * half;
    unsigned char *local_input = (unsigned char *)malloc(local_height * width * 3);
    unsigned char *local_output = (unsigned char *)malloc(my_rows * width * 3);

    // Workers precisam alocar estrutura de output também
    if (rank != 0) {
        output = (Image *)malloc(sizeof(Image));
        output->width = width;
        output->height = height;
        output->max = max;
        output->data = (unsigned char *)malloc(height * width * 3);
    }

    // Preparar scatter/gather arrays
    int *sendcounts = NULL, *displs = NULL;
    int *recvcounts = NULL, *recvdispls = NULL;

    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        recvcounts = (int *)malloc(size * sizeof(int));
        recvdispls = (int *)malloc(size * sizeof(int));

        int current_row = 0;
        for (int i = 0; i < size; i++) {
            int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);

            // Calcular linhas com halo
            int start_with_halo = (current_row - half < 0) ? 0 : current_row - half;
            int end_with_halo = (current_row + proc_rows + half > height) ? height : current_row + proc_rows + half;
            int rows_with_halo = end_with_halo - start_with_halo;

            sendcounts[i] = rows_with_halo * width * 3;
            displs[i] = start_with_halo * width * 3;

            recvcounts[i] = proc_rows * width * 3;
            recvdispls[i] = current_row * width * 3;

            current_row += proc_rows;
        }
    }

    // Scatter das linhas (com halo)
    MPI_Scatterv(
        rank == 0 ? input->data : NULL, sendcounts, displs, MPI_UNSIGNED_CHAR,
        local_input, local_height * width * 3, MPI_UNSIGNED_CHAR,
        0, comm
    );

    for (int ly = 0; ly < my_rows; ly++) {
        for (int x = half; x < width - half; x++) {
            double r = 0, g = 0, b = 0;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int input_row = ly + half + ky;
                    int input_col = x + kx;

                    int pos = (input_row * width + input_col) * 3;
                    double k = kernel[ky + half][kx + half];

                    r += local_input[pos] * k;
                    g += local_input[pos + 1] * k;
                    b += local_input[pos + 2] * k;
                }
            }

            int outPos = (ly * width + x) * 3;
            local_output[outPos] = (unsigned char)r;
            local_output[outPos + 1] = (unsigned char)g;
            local_output[outPos + 2] = (unsigned char)b;
        }
    }

    // Gather dos resultados
    MPI_Gatherv(
        local_output, my_rows * width * 3, MPI_UNSIGNED_CHAR,
        output->data, recvcounts, recvdispls, MPI_UNSIGNED_CHAR,
        0, comm
    );

    // Cleanup
    free(local_input);
    free(local_output);

    if (rank != 0) {
        free(output->data);
        free(output);
    }

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

#endif
