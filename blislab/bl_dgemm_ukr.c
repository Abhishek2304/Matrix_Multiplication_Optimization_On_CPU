#include "bl_config.h"
#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k, //Kc
		           int    m, //Mr
                   int    n, //Nr
                   const double * restrict a, //packA 
                   const double * restrict b, //packB
                   double *c, // C - MrxNr section
                   unsigned long long ldc, // Move to next row in c
                   aux_t* data )
{
    int l, j, i;
    //printf("%d, %d, %d, %d \n", (k, DGEMM_MR, n, ldc));
    for (l = 0; l<k; ++l){
        // const double * restrict ak = &a[4*l];
        // const double * restrict bk = &b[4*l];
        // c[0*ldc + 0] += ak[0]*bk[0];
        // c[0*ldc + 1] += ak[0]*bk[1];
        // c[0*ldc + 2] += ak[0]*bk[2];
        // c[0*ldc + 3] += ak[0]*bk[3];
        // c[1*ldc + 0] += ak[1]*bk[0];
        // c[1*ldc + 1] += ak[1]*bk[1];
        // c[1*ldc + 2] += ak[1]*bk[2];
        // c[1*ldc + 3] += ak[1]*bk[3];
        // c[2*ldc + 0] += ak[2]*bk[0];
        // c[2*ldc + 1] += ak[2]*bk[1];
        // c[2*ldc + 2] += ak[2]*bk[2];
        // c[2*ldc + 3] += ak[2]*bk[3];
        // c[3*ldc + 0] += ak[3]*bk[0];
        // c[3*ldc + 1] += ak[3]*bk[1];
        // c[3*ldc + 2] += ak[3]*bk[2];
        // c[3*ldc + 3] += ak[3]*bk[3];
        for (i = 0; i < m; ++i){
            for (j = 0; j < n; ++j){
                //c[ldc*i + j] += a[l*m + i] * b[l*n + j];
                c(i,j, ldc) += a(m, i, l) * b(n, j, l);
            }
        }
    }

    // for ( l = 0; l < k; ++l )
    // {                 
    //     for ( j = 0; j < n; ++j )
    //     { 
    //         for ( i = 0; i < m; ++i )
    //         { 
    //             // ldc is used here because a[] and b[] are not packed by the
    //             // starter code
    //             // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
    //             //
    //             c( i, j, ldc ) += a( i, l, ldc) * b( l, j, ldc );   
    //         }
    //     }
    // }

}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//

// C = A*B
// A = m x k, B = k x n, C = m x n
void gemm_avx( int k,
               int m,
               int n,
               const double * restrict a,
               const double * restrict b,
               double *c,
               unsigned long long ldc,
               aux_t* data)
{
    __m256d ymm[4];
    double result[4];
    double temp[4];
    __m256d matrix2[4];

    unsigned long long ldm = 4;
    double m1[16];
    double m2[16];
    double res[16];

    // padding matrices less than 4x4
    int ind = 0;
    for(int i = 0; i < sizeof(m1)/sizeof(m1[0]); i++){
        if(i%ldm < ldc && i < ldc*ldm){
            m1[i] = a[ind];
            m2[i] = b[ind];
            ind++;
        } else {
            m1[i] = 0;
            m2[i] = 0;
        }
    }


    for(int i = 0; i < 4; i++){
        // ymm[0] = _mm256_broadcast_sd(a+i*ldc+0);
        // ymm[1] = _mm256_broadcast_sd(a+i*ldc+1);
        // ymm[2] = _mm256_broadcast_sd(a+i*ldc+2);
        // ymm[3] = _mm256_broadcast_sd(a+i*ldc+3);

        ymm[0] = _mm256_broadcast_sd(&m1[i*ldm+0]);
        ymm[1] = _mm256_broadcast_sd(&m1[i*ldm+1]);
        ymm[2] = _mm256_broadcast_sd(&m1[i*ldm+2]);
        ymm[3] = _mm256_broadcast_sd(&m1[i*ldm+3]);

        temp[0] = m2[i*ldm + 0];
        temp[1] = m2[i*ldm + 1];
        temp[2] = m2[i*ldm + 2];
        temp[3] = m2[i*ldm + 3];
        matrix2[0] = _mm256_loadu_pd(&temp[0]);
        ymm[0] = _mm256_mul_pd(ymm[0], matrix2[0]);
        ymm[1] = _mm256_mul_pd(ymm[1], matrix2[1]);

        //ymm[0] = _mm256_add_pd(ymm[0], ymm[1]);

        ymm[2] = _mm256_mul_pd(ymm[2], matrix2[2]);
        ymm[3] = _mm256_mul_pd(ymm[3], matrix2[3]);

        //ymm[2] = _mm256_add_pd(ymm[2], ymm[3]);
        //_mm256_storeu_pd(result, _mm256_add_pd(ymm[0], ymm[2]));

        res[0] += ymm[0][0];
        res[1] += ymm[0][1];
        res[2] += ymm[0][2];
        res[3] += ymm[0][3];
        res[4] += ymm[1][0];
        res[5] += ymm[1][1];
        res[6] += ymm[1][2];
        res[7] += ymm[1][3];
        res[8] += ymm[2][0];
        res[9] += ymm[2][1];
        res[10] += ymm[2][2];
        res[11] += ymm[2][3];
        res[12] += ymm[3][0];
        res[13] += ymm[3][1];
        res[14] += ymm[3][2];
        res[15] += ymm[3][3];

        // res[i*ldm+0] = result[0];
        // res[i*ldm+1] = result[1];
        // res[i*ldm+2] = result[2];
        // res[i*ldm+3] = result[3];

        // c(i, 0, ldc) = result[0];
        // c(i, 1, ldc) = result[1];
        // c(i, 2, ldc) = result[2];
        // c(i, 3, ldc) = result[3];


    }
    //ind = 0;
    for(int i = 0; i < sizeof(res)/sizeof(res[0]); i++){
        if(i%ldm<ldc && i < ldc*ldm){
            int row = i/4;
            int col = i%4;
            c[row*ldc + col] = res[i];
            //ind++;
        }
    }

}


// static inline void gemm_avx (
//     t_m4d *restrict dst,
//     const t_m4d *restrict matrix1,
//     const t_m4d *restrict matrix2)
// {
//     __m256d ymm[4];

//     for (int i = 0; i < 4; i++){
//         ymm[0] = _mm256_broadcast_sd(&matrix1->d[i][0]);
//         ymm[1] = _mm256_broadcast_sd(&matrix1->d[i][1]);
//         ymm[2] = _mm256_broadcast_sd(&matrix1->d[i][2]);
//         ymm[3] = _mm256_broadcast_sd(&matrix1->d[i][3]);
//         ymm[0] = _mm256_mul_pd(ymm[0], matrix2->m256d[0]);
//         ymm[1] = _mm256_mul_pd(ymm[1], matrix2->m256d[1]);
//         ymm[0] = _mm256_add_pd(ymm[0], ymm[1]);
//         ymm[2] = _mm256_mul_pd(ymm[2], matrix2->m256d[2]);
//         ymm[3] = _mm256_mul_pd(ymm[3], matrix2->m256d[3]);
//         ymm[2] = _mm256_add_pd(ymm[2], ymm[3]);
//         dst->m256d[i] = _mm256_add_pd(ymm[0], ymm[2]);
//     }

// }

