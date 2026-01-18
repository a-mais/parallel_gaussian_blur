#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *inputPath  = "../input/image.ppm";
    const char *outputSeq  = "../output/output_sequential.ppm";
    const char *outputPar  = "../output/output_parallel.ppm";
    const char *outputMPI  = "../output/output_mpi.ppm";

    int kernelSize = 31;
    double sigma = 46.0;

    Image *img = NULL;
    Image *seq = NULL;
    Image *par = NULL;
    Image *mpi_img = NULL;

    double time_seq = 0.0, time_par = 0.0, time_mpi = 0.0;
    double start, end;

    if (rank == 0) {
        img = readPPM(inputPath);

        // Alocar imagens de saída
        seq = malloc(sizeof(Image));
        par = malloc(sizeof(Image));
        mpi_img = malloc(sizeof(Image));

        *seq = *img;
        *par = *img;
        *mpi_img = *img;

        seq->data = malloc(3 * img->width * img->height);
        par->data = malloc(3 * img->width * img->height);
        mpi_img->data = malloc(3 * img->width * img->height);

        printf("=== Benchmark Gaussian Blur ===\n");
        printf("Imagem: %dx%d pixels\n", img->width, img->height);
        printf("Kernel: %dx%d, sigma=%.1f\n", kernelSize, kernelSize, sigma);
        printf("Processos MPI: %d\n\n", size);

        // Sequencial
        printf("1. Sequencial...\n");
        start = omp_get_wtime();
        gaussianBlurSequential(img, seq, kernelSize, sigma);
        end = omp_get_wtime();
        time_seq = end - start;
    }

    // OpenMP (apenas rank 0)
    if (rank == 0) {
        printf("2. OpenMP...\n");
        start = omp_get_wtime();
        gaussianBlurParallel(img, par, kernelSize, sigma);
        end = omp_get_wtime();
        time_par = end - start;
    }

    // MPI Híbrido - TODOS os processos participam
    if (rank == 0) {
        printf("3. MPI Hibrido...\n");
        start = omp_get_wtime();
    }

    gaussianBlurMPI(img, mpi_img, kernelSize, sigma, MPI_COMM_WORLD);

    if (rank == 0) {
        end = omp_get_wtime();
        time_mpi = end - start;
    }

    if (rank == 0) {
        double speedup_par = time_seq / time_par;
        double speedup_mpi = time_seq / time_mpi;

        printf("\n=== RESULTADOS ===\n");
        printf("Sequencial: %.4f s\n", time_seq);
        printf("OpenMP:     %.4f s (%.2fx)\n", time_par, speedup_par);
        printf("MPI:        %.4f s (%.2fx)\n", time_mpi, speedup_mpi);

        writePPM(outputSeq, seq);
        writePPM(outputPar, par);
        writePPM(outputMPI, mpi_img);

        // Cleanup
        free(img->data); free(seq->data); free(par->data); free(mpi_img->data);
        free(img); free(seq); free(par); free(mpi_img);
    }

    MPI_Finalize();
    return 0;
}
