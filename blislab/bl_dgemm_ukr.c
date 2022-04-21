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
    for (l = 0; l<k; ++l){
        for (i = 0; i < m; ++i){
            for (j = 0; j < n; ++j){
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
    // double result[16];
    double temp[8];

    // unsigned long long ldm = 4;
    // double m1[16];
    // double m2[16];
    // double res[4];

    // TODO!!!: Need to implement this padding after the code works.

    // padding matrices less than 4x4
    // int ind = 0;
    // for(int i = 0; i < sizeof(m1)/sizeof(m1[0]); i++){
    //     if(i%ldm < ldc && i < ldc*ldm){
    //         m1[i] = a[ind];
    //         m2[i] = b[ind];
    //         ind++;
    //     } else {
    //         m1[i] = 0;
    //         m2[i] = 0;
    //     }
    // }

    register __m256d c00_c01_c02_c03 = _mm256_load_pd(c + 4*0 + ldc*4*0);
    register __m256d c10_c11_c12_c13 = _mm256_load_pd(c + ldc + 4*0 + ldc*4*0);
    register __m256d c20_c21_c22_c23 = _mm256_load_pd(c + 2*ldc + 4*0 + ldc*4*0);
    register __m256d c30_c31_c32_c33 = _mm256_load_pd(c + 3*ldc + 4*0 + ldc*4*0);
    register __m256d c04_c05_c06_c07 = _mm256_load_pd(c + 4*1 + ldc*4*0);
    register __m256d c14_c15_c16_c17 = _mm256_load_pd(c + ldc + 4*1 + ldc*4*0);
    register __m256d c24_c25_c26_c27 = _mm256_load_pd(c + 2*ldc + 4*1 + ldc*4*0);
    register __m256d c34_c35_c36_c37 = _mm256_load_pd(c + 3*ldc + 4*1 + ldc*4*0);

    for(int i = 0; i < k; i++){

        register __m256d a0x = _mm256_broadcast_sd(&a[i*m + 4*0 + 0]);
        register __m256d a1x = _mm256_broadcast_sd(&a[i*m + 4*0 + 1]);
        register __m256d a2x = _mm256_broadcast_sd(&a[i*m + 4*0 + 2]);
        register __m256d a3x = _mm256_broadcast_sd(&a[i*m + 4*0 + 3]);
        
        temp[0] = b[i*n + 4*0 + 0];
        temp[1] = b[i*n + 4*0 + 1];
        temp[2] = b[i*n + 4*0 + 2];
        temp[3] = b[i*n + 4*0 + 3];
        temp[4] = b[i*n + 4*1 + 0];
        temp[5] = b[i*n + 4*1 + 1];
        temp[6] = b[i*n + 4*1 + 2];
        temp[7] = b[i*n + 4*1 + 3];
        register __m256d matrix1 = _mm256_loadu_pd(&temp[0]);
        register __m256d matrix2 = _mm256_loadu_pd(&temp[4]);

        c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, matrix1, c00_c01_c02_c03);
        c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, matrix1, c10_c11_c12_c13);
        c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, matrix1, c20_c21_c22_c23);
        c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, matrix1, c30_c31_c32_c33);
        c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, matrix2, c04_c05_c06_c07);
        c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, matrix2, c14_c15_c16_c17);
        c24_c25_c26_c27 = _mm256_fmadd_pd(a2x, matrix2, c24_c25_c26_c27);
        c34_c35_c36_c37 = _mm256_fmadd_pd(a3x, matrix2, c34_c35_c36_c37);
    }

    _mm256_store_pd(c + 4*0 + ldc*4*0, c00_c01_c02_c03);
    _mm256_store_pd(c + ldc + 4*0 + ldc*4*0, c10_c11_c12_c13);
    _mm256_store_pd(c + 2*ldc + 4*0+ ldc*4*0, c20_c21_c22_c23);
    _mm256_store_pd(c + 3*ldc + 4*0 + ldc*4*0, c30_c31_c32_c33);
    _mm256_store_pd(c + 4*1 + ldc*4*0, c04_c05_c06_c07);
    _mm256_store_pd(c + ldc + 4*1 + ldc*4*0, c14_c15_c16_c17);
    _mm256_store_pd(c + 2*ldc + 4*1 + ldc*4*0, c24_c25_c26_c27);
    _mm256_store_pd(c + 3*ldc + 4*1 + ldc*4*0, c34_c35_c36_c37);

    register __m256d c40_c41_c42_c43 = _mm256_load_pd(c + 4*0 + ldc*4*1);
    register __m256d c50_c51_c52_c53 = _mm256_load_pd(c + ldc + 4*0 + ldc*4*1);
    register __m256d c60_c61_c62_c63 = _mm256_load_pd(c + 2*ldc + 4*0 + ldc*4*1);
    register __m256d c70_c71_c72_c73 = _mm256_load_pd(c + 3*ldc + 4*0 + ldc*4*1);
    register __m256d c44_c45_c46_c47 = _mm256_load_pd(c + 4*1 + ldc*4*1);
    register __m256d c54_c55_c56_c57 = _mm256_load_pd(c + ldc + 4*1 + ldc*4*1);
    register __m256d c64_c65_c66_c67 = _mm256_load_pd(c + 2*ldc + 4*1 + ldc*4*1);
    register __m256d c74_c75_c76_c77 = _mm256_load_pd(c + 3*ldc + 4*1 + ldc*4*1);

    for(int i = 0; i < k; i++){

        register __m256d a0x = _mm256_broadcast_sd(&a[i*m + 4*1 + 0]);
        register __m256d a1x = _mm256_broadcast_sd(&a[i*m + 4*1 + 1]);
        register __m256d a2x = _mm256_broadcast_sd(&a[i*m + 4*1 + 2]);
        register __m256d a3x = _mm256_broadcast_sd(&a[i*m + 4*1 + 3]);
        
        temp[0] = b[i*n + 4*0 + 0];
        temp[1] = b[i*n + 4*0 + 1];
        temp[2] = b[i*n + 4*0 + 2];
        temp[3] = b[i*n + 4*0 + 3];
        temp[4] = b[i*n + 4*1 + 0];
        temp[5] = b[i*n + 4*1 + 1];
        temp[6] = b[i*n + 4*1 + 2];
        temp[7] = b[i*n + 4*1 + 3];
        register __m256d matrix1 = _mm256_loadu_pd(&temp[0]);
        register __m256d matrix2 = _mm256_loadu_pd(&temp[4]);

        c40_c41_c42_c43 = _mm256_fmadd_pd(a0x, matrix1, c40_c41_c42_c43);
        c50_c51_c52_c53 = _mm256_fmadd_pd(a1x, matrix1, c50_c51_c52_c53);
        c60_c61_c62_c63 = _mm256_fmadd_pd(a2x, matrix1, c60_c61_c62_c63);
        c70_c71_c72_c73 = _mm256_fmadd_pd(a3x, matrix1, c70_c71_c72_c73);
        c44_c45_c46_c47 = _mm256_fmadd_pd(a0x, matrix2, c44_c45_c46_c47);
        c54_c55_c56_c57 = _mm256_fmadd_pd(a1x, matrix2, c54_c55_c56_c57);
        c64_c65_c66_c67 = _mm256_fmadd_pd(a2x, matrix2, c64_c65_c66_c67);
        c74_c75_c76_c77 = _mm256_fmadd_pd(a3x, matrix2, c74_c75_c76_c77);
    }

    _mm256_store_pd(c + 4*0 + ldc*4*1, c40_c41_c42_c43);
    _mm256_store_pd(c + ldc + 4*0 + ldc*4*1, c50_c51_c52_c53);
    _mm256_store_pd(c + 2*ldc + 4*0+ ldc*4*1, c60_c61_c62_c63);
    _mm256_store_pd(c + 3*ldc + 4*0 + ldc*4*1, c70_c71_c72_c73);
    _mm256_store_pd(c + 4*1 + ldc*4*1, c44_c45_c46_c47);
    _mm256_store_pd(c + ldc + 4*1 + ldc*4*1, c54_c55_c56_c57);
    _mm256_store_pd(c + 2*ldc + 4*1 + ldc*4*1, c64_c65_c66_c67);
    _mm256_store_pd(c + 3*ldc + 4*1 + ldc*4*1, c74_c75_c76_c77);

    // TODO!!!: Do the 0 unpacking
    //ind = 0;
    // for(int i = 0; i < sizeof(res)/sizeof(res[0]); i++){
    //     if(i%ldm<ldc && i < ldc*ldm){
    //         int row = i/4;
    //         int col = i%4;
    //         c[row*ldc + col] = res[i];
    //         //ind++;
    //     }
    // }

}

