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
    // double result[16];
    double temp[4];

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

            
    // 0,0
    register __m256d c00_c01_c02_c03 = _mm256_load_pd(c + 4*0 + ldc*4*0);
    register __m256d c10_c11_c12_c13 = _mm256_load_pd(c + ldc + 4*0 + ldc*4*0);
    register __m256d c20_c21_c22_c23 = _mm256_load_pd(c + 2*ldc + 4*0 + ldc*4*0);
    register __m256d c30_c31_c32_c33 = _mm256_load_pd(c + 3*ldc + 4*0 + ldc*4*0);

    for(int i = 0; i < k; i++){

        register __m256d a0x = _mm256_broadcast_sd(&a[i*m + 4*0 + 0]);
        register __m256d a1x = _mm256_broadcast_sd(&a[i*m + 4*0 + 1]);
        register __m256d a2x = _mm256_broadcast_sd(&a[i*m + 4*0 + 2]);
        register __m256d a3x = _mm256_broadcast_sd(&a[i*m + 4*0 + 3]);
        
        temp[0] = b[i*n + 4*0 + 0];
        temp[1] = b[i*n + 4*0 + 1];
        temp[2] = b[i*n + 4*0 + 2];
        temp[3] = b[i*n + 4*0 + 3];
        register __m256d matrix2 = _mm256_loadu_pd(&temp[0]);

        c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, matrix2, c00_c01_c02_c03);
        c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, matrix2, c10_c11_c12_c13);
        c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, matrix2, c20_c21_c22_c23);
        c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, matrix2, c30_c31_c32_c33);
    }

    _mm256_store_pd(c + 4*0 + ldc*4*0, c00_c01_c02_c03);
    _mm256_store_pd(c + ldc + 4*0 + ldc*4*0, c10_c11_c12_c13);
    _mm256_store_pd(c + 2*ldc + 4*0+ ldc*4*0, c20_c21_c22_c23);
    _mm256_store_pd(c + 3*ldc + 4*0 + ldc*4*0, c30_c31_c32_c33);

    // //0,1
    // c00_c01_c02_c03 = _mm256_load_pd(c + 4*1 + ldc*4*0);
    // c10_c11_c12_c13 = _mm256_load_pd(c + ldc + 4*1 + ldc*4*0);
    // c20_c21_c22_c23 = _mm256_load_pd(c + 2*ldc + 4*1 + ldc*4*0);
    // c30_c31_c32_c33 = _mm256_load_pd(c + 3*ldc + 4*1 + ldc*4*0);

    // for(int i = 0; i < k; i++){

    //     register __m256d a0x = _mm256_broadcast_sd(&a[i*m + 4*0 + 0]);
    //     register __m256d a1x = _mm256_broadcast_sd(&a[i*m + 4*0+ 1]);
    //     register __m256d a2x = _mm256_broadcast_sd(&a[i*m + 4*0 + 2]);
    //     register __m256d a3x = _mm256_broadcast_sd(&a[i*m + 4*0+ 3]);
        
    //     temp[0] = b[i*n + 4*1 + 0];
    //     temp[1] = b[i*n + 4*1 + 1];
    //     temp[2] = b[i*n + 4*1 + 2];
    //     temp[3] = b[i*n + 4*1 + 3];
    //     register __m256d matrix2 = _mm256_loadu_pd(&temp[0]);

    //     c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, matrix2, c00_c01_c02_c03);
    //     c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, matrix2, c10_c11_c12_c13);
    //     c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, matrix2, c20_c21_c22_c23);
    //     c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, matrix2, c30_c31_c32_c33);
    // }

    // _mm256_store_pd(c + 4*1 + ldc*4*0, c00_c01_c02_c03);
    // _mm256_store_pd(c + ldc + 4*1 + ldc*4*0, c10_c11_c12_c13);
    // _mm256_store_pd(c + 2*ldc + 4*1 + ldc*4*0, c20_c21_c22_c23);
    // _mm256_store_pd(c + 3*ldc + 4*1 + ldc*4*0, c30_c31_c32_c33);

    //1,0
    c00_c01_c02_c03 = _mm256_load_pd(c + 4*0 + ldc*4*1);
    c10_c11_c12_c13 = _mm256_load_pd(c + ldc + 4*0 + ldc*4*1);
    c20_c21_c22_c23 = _mm256_load_pd(c + 2*ldc + 4*0 + ldc*4*1);
    c30_c31_c32_c33 = _mm256_load_pd(c + 3*ldc + 4*0 + ldc*4*1);

    for(int i = 0; i < k; i++){

        register __m256d a0x = _mm256_broadcast_sd(&a[i*m + 4*1 + 0]);
        register __m256d a1x = _mm256_broadcast_sd(&a[i*m + 4*1 + 1]);
        register __m256d a2x = _mm256_broadcast_sd(&a[i*m + 4*1 + 2]);
        register __m256d a3x = _mm256_broadcast_sd(&a[i*m + 4*1 + 3]);
        
        temp[0] = b[i*n + 4*0 + 0];
        temp[1] = b[i*n + 4*0 + 1];
        temp[2] = b[i*n + 4*0 + 2];
        temp[3] = b[i*n + 4*0+ 3];
        register __m256d matrix2 = _mm256_loadu_pd(&temp[0]);

        c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, matrix2, c00_c01_c02_c03);
        c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, matrix2, c10_c11_c12_c13);
        c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, matrix2, c20_c21_c22_c23);
        c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, matrix2, c30_c31_c32_c33);
    }

    _mm256_store_pd(c + 4*0 + ldc*4*1, c00_c01_c02_c03);
    _mm256_store_pd(c + ldc + 4*0 + ldc*4*1, c10_c11_c12_c13);
    _mm256_store_pd(c + 2*ldc + 4*0 + ldc*4*1, c20_c21_c22_c23);
    _mm256_store_pd(c + 3*ldc + 4*0 + ldc*4*1, c30_c31_c32_c33);

    // //1,1

    // c00_c01_c02_c03 = _mm256_load_pd(c + 4*1 + ldc*4*1);
    // c10_c11_c12_c13 = _mm256_load_pd(c + ldc + 4*1 + ldc*4*1);
    // c20_c21_c22_c23 = _mm256_load_pd(c + 2*ldc + 4*1 + ldc*4*1);
    // c30_c31_c32_c33 = _mm256_load_pd(c + 3*ldc + 4*1 + ldc*4*1);

    // for(int i = 0; i < k; i++){

    //     register __m256d a0x = _mm256_broadcast_sd(&a[i*m + 4*1 + 0]);
    //     register __m256d a1x = _mm256_broadcast_sd(&a[i*m + 4*1 + 1]);
    //     register __m256d a2x = _mm256_broadcast_sd(&a[i*m + 4*1 + 2]);
    //     register __m256d a3x = _mm256_broadcast_sd(&a[i*m + 4*1 + 3]);
        
    //     temp[0] = b[i*n + 4*1 + 0];
    //     temp[1] = b[i*n + 4*1 + 1];
    //     temp[2] = b[i*n + 4*1 + 2];
    //     temp[3] = b[i*n + 4*1 + 3];
    //     register __m256d matrix2 = _mm256_loadu_pd(&temp[0]);

    //     c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, matrix2, c00_c01_c02_c03);
    //     c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, matrix2, c10_c11_c12_c13);
    //     c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, matrix2, c20_c21_c22_c23);
    //     c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, matrix2, c30_c31_c32_c33);
    // }

    // _mm256_store_pd(c + 4*1 + ldc*4*1, c00_c01_c02_c03);
    // _mm256_store_pd(c + ldc + 4*1 + ldc*4*1, c10_c11_c12_c13);
    // _mm256_store_pd(c + 2*ldc + 4*1 + ldc*4*1, c20_c21_c22_c23);
    // _mm256_store_pd(c + 3*ldc + 4*1 + ldc*4*1, c30_c31_c32_c33);

                
                // ymm[4] = _mm256_mul_pd(ymm[0], matrix2);
                // ymm[5] = _mm256_mul_pd(ymm[1], matrix2);
                // ymm[6] = _mm256_mul_pd(ymm[2], matrix2);
                // ymm[7] = _mm256_mul_pd(ymm[3], matrix2);

                // _mm256_storeu_pd(&result[0], ymm[4]);
                // _mm256_storeu_pd(&result[4], ymm[5]);
                // _mm256_storeu_pd(&result[8], ymm[6]);
                // _mm256_storeu_pd(&result[12], ymm[7]);

                // c(4*e + 0,4*f + 0,ldc) += result[0];
                // c(4*e + 0,4*f + 1,ldc) += result[1];
                // c(4*e + 0,4*f + 2,ldc) += result[2];
                // c(4*e + 0,4*f + 3,ldc) += result[3];
                // c(4*e + 1,4*f + 0,ldc) += result[4];
                // c(4*e + 1,4*f + 1,ldc) += result[5];
                // c(4*e + 1,4*f + 2,ldc) += result[6];
                // c(4*e + 1,4*f + 3,ldc) += result[7];
                // c(4*e + 2,4*f + 0,ldc) += result[8];
                // c(4*e + 2,4*f + 1,ldc) += result[9];
                // c(4*e + 2,4*f + 2,ldc) += result[10];
                // c(4*e + 2,4*f + 3,ldc) += result[11];
                // c(4*e + 3,4*f + 0,ldc) += result[12];
                // c(4*e + 3,4*f + 1,ldc) += result[13];
                // c(4*e + 3,4*f + 2,ldc) += result[14];
                // c(4*e + 3,4*f + 3,ldc) += result[15];

        // ymm[0] = _mm256_broadcast_sd(&a[i*m+0]);
        //         ymm[1] = _mm256_broadcast_sd(&a[i*m+1]);
        //         ymm[2] = _mm256_broadcast_sd(&a[i*m+2]);
        //         ymm[3] = _mm256_broadcast_sd(&a[i*m+3]);
        //         ymm[4] = _mm256_broadcast_sd(&a[i*m+4]);
        //         ymm[5] = _mm256_broadcast_sd(&a[i*m+5]);
        //         ymm[6] = _mm256_broadcast_sd(&a[i*m+6]);
        //         ymm[7] = _mm256_broadcast_sd(&a[i*m+7]);
                
        //         temp[0] = b[i*n + 0];
        //         temp[1] = b[i*n + 1];
        //         temp[2] = b[i*n + 2];
        //         temp[3] = b[i*n + 3];
        //         temp[4] = b[i*n + 4];
        //         temp[5] = b[i*n + 5];
        //         temp[6] = b[i*n + 6];
        //         temp[7] = b[i*n + 7];
        //         matrix2[0] = _mm256_loadu_pd(&temp[0]);
        //         matrix2[1] = _mm256_loadu_pd(&temp[4]);
                
        //         ymm[8] = _mm256_mul_pd(ymm[0], matrix2[0]);
        //         ymm[9] = _mm256_mul_pd(ymm[1], matrix2[0]);
        //         ymm[10] = _mm256_mul_pd(ymm[2], matrix2[0]);
        //         ymm[11] = _mm256_mul_pd(ymm[3], matrix2[0]);
        //         ymm[12] = _mm256_mul_pd(ymm[4], matrix2[0]);
        //         ymm[13] = _mm256_mul_pd(ymm[5], matrix2[0]);
        //         ymm[14] = _mm256_mul_pd(ymm[6], matrix2[0]);
        //         ymm[15] = _mm256_mul_pd(ymm[7], matrix2[0]);

        //         _mm256_storeu_pd(&result[0], ymm[8]);
        //         _mm256_storeu_pd(&result[4], ymm[9]);
        //         _mm256_storeu_pd(&result[8], ymm[10]);
        //         _mm256_storeu_pd(&result[12], ymm[11]);
        //         _mm256_storeu_pd(&result[16], ymm[12]);
        //         _mm256_storeu_pd(&result[20], ymm[13]);
        //         _mm256_storeu_pd(&result[24], ymm[14]);
        //         _mm256_storeu_pd(&result[28], ymm[15]);

        //         c(0,0,ldc) += result[0];
        //         c(0,1,ldc) += result[1];
        //         c(0,2,ldc) += result[2];
        //         c(0,3,ldc) += result[3];
        //         c(1,0,ldc) += result[4];
        //         c(1,1,ldc) += result[5];
        //         c(1,2,ldc) += result[6];
        //         c(1,3,ldc) += result[7];
        //         c(2,0,ldc) += result[8];
        //         c(2,1,ldc) += result[9];
        //         c(2,2,ldc) += result[10];
        //         c(2,3,ldc) += result[11];
        //         c(3,0,ldc) += result[12];
        //         c(3,1,ldc) += result[13];
        //         c(3,2,ldc) += result[14];
        //         c(3,3,ldc) += result[15];
        //         c(4,0,ldc) += result[16];
        //         c(4,1,ldc) += result[17];
        //         c(4,2,ldc) += result[18];
        //         c(4,3,ldc) += result[19];
        //         c(5,0,ldc) += result[20];
        //         c(5,1,ldc) += result[21];
        //         c(5,2,ldc) += result[22];
        //         c(5,3,ldc) += result[23];
        //         c(6,0,ldc) += result[24];
        //         c(6,1,ldc) += result[25];
        //         c(6,2,ldc) += result[26];
        //         c(6,3,ldc) += result[27];
        //         c(7,0,ldc) += result[28];
        //         c(7,1,ldc) += result[29];
        //         c(7,2,ldc) += result[30];
        //         c(7,3,ldc) += result[31];
                
        //         ymm[8] = _mm256_mul_pd(ymm[0], matrix2[1]);
        //         ymm[9] = _mm256_mul_pd(ymm[1], matrix2[1]);
        //         ymm[10] = _mm256_mul_pd(ymm[2], matrix2[1]);
        //         ymm[11] = _mm256_mul_pd(ymm[3], matrix2[1]);
        //         ymm[12] = _mm256_mul_pd(ymm[4], matrix2[1]);
        //         ymm[13] = _mm256_mul_pd(ymm[5], matrix2[1]);
        //         ymm[14] = _mm256_mul_pd(ymm[6], matrix2[1]);
        //         ymm[15] = _mm256_mul_pd(ymm[7], matrix2[1]);

        //         _mm256_storeu_pd(&result[0], ymm[8]);
        //         _mm256_storeu_pd(&result[4], ymm[9]);
        //         _mm256_storeu_pd(&result[8], ymm[10]);
        //         _mm256_storeu_pd(&result[12], ymm[11]);
        //         _mm256_storeu_pd(&result[16], ymm[12]);
        //         _mm256_storeu_pd(&result[20], ymm[13]);
        //         _mm256_storeu_pd(&result[24], ymm[14]);
        //         _mm256_storeu_pd(&result[28], ymm[15]);

        //         c(0,4,ldc) += result[0];
        //         c(0,5,ldc) += result[1];
        //         c(0,6,ldc) += result[2];
        //         c(0,7,ldc) += result[3];
        //         c(1,4,ldc) += result[4];
        //         c(1,5,ldc) += result[5];
        //         c(1,6,ldc) += result[6];
        //         c(1,7,ldc) += result[7];
        //         c(2,4,ldc) += result[8];
        //         c(2,5,ldc) += result[9];
        //         c(2,6,ldc) += result[10];
        //         c(2,7,ldc) += result[11];
        //         c(3,4,ldc) += result[12];
        //         c(3,5,ldc) += result[13];
        //         c(3,6,ldc) += result[14];
        //         c(3,7,ldc) += result[15];
        //         c(4,4,ldc) += result[16];
        //         c(4,5,ldc) += result[17];
        //         c(4,6,ldc) += result[18];
        //         c(4,7,ldc) += result[19];
        //         c(5,4,ldc) += result[20];
        //         c(5,5,ldc) += result[21];
        //         c(5,6,ldc) += result[22];
        //         c(5,7,ldc) += result[23];
        //         c(6,4,ldc) += result[24];
        //         c(6,5,ldc) += result[25];
        //         c(6,6,ldc) += result[26];
        //         c(6,7,ldc) += result[27];
        //         c(7,4,ldc) += result[28];
        //         c(7,5,ldc) += result[29];
        //         c(7,6,ldc) += result[30];
        //         c(7,7,ldc) += result[31];

    // for (int i = 0; i < 4; i++){
    //     for (int j = 0; j < 4; j++){
    //         printf("%d ", c(i,j,ldc));
    //     }
    //     printf("\n");
    // }

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

