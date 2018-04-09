/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems Spring 2017
 * Instructor Bryan Mills, PhD
 * Student: 
 * Implement Pthreads version of Strassen algorithm for matrix multiplication.
 */

#include "timer.h"
#include "io.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NUM_THREADS 11
#include <semaphore.h>
// Make these globals so threads can operate on them. You will need to
// add additional matrixes for all the M and C values in the Strassen
// algorithms.
int **A;
int **B;
int **C;
int **C_p;
// Reference matrix, call simpleMM to populate.
int **R;
//R_1 R_2 for temp result
int **R_1;
int **R_2;

int **M_1;
int **M_2;
int **M_3;
int **M_4;
int **M_5;
int **M_6;
int **M_7;

int sub_size;
int **zero;


// Stupid simple Matrix Multiplication, meant as example.
void simpleMM(int N) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k++) {
        R[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// WRITE CODE HERE, you will need to also add functions for each
// of the sub-matrixes you will need to calculate but you can create your
// threads in this fucntion.
void product(int** first, int** second, int** result) {
    for (int i = 0; i < sub_size; i++) {
        for (int j = 0; j < sub_size; j++) {
            result[i][j] = 0;
            for (int k = 0; k < sub_size; k++) {
                result[i][j] += first[i][k] * second[k][j];
            }
        }
    }
}
void Add(int** first, int** second, int** result,int i_1,int j_1,int i_2,int j_2){
    for(int i = 0; i < sub_size; i++) {
        for(int j = 0; j < sub_size; j++){
            result[i][j] = first[i_1 + i][j_1 + j] + second[i_2 + i][j_2 + j];
        }
    }
}
void minus(int** first, int** second, int** result,int i_1,int j_1,int i_2,int j_2){
    for(int i = 0; i < sub_size; i++) {
        for(int j = 0; j < sub_size; j++){
            result[i][j] = first[i_1 + i][j_1 + j] - second[i_2 + i][j_2 + j];
        }
    }
}

void assign(int** first, int** second,int i_1,int j_1){
    for(int i = 0; i < sub_size; i++){
        for(int j = 0; j < sub_size; j++){
            first[i + i_1][j + j_1] = second[i][j];
        }
    }
}
void strassenMM(int N) {
  // Do something here... seriously.
  //M1 = (A11 + A22)(B11 + B22) 
    Add(A,A,R_1,0,0,sub_size,sub_size);
    Add(B,B,R_2,0,0,sub_size,sub_size);
    product(R_1,R_2,M_1);
  //M2 = (A21 + A22)B11
    Add(A,A,R_1,sub_size,0,sub_size,sub_size);
    Add(B,zero,R_2,0,0,0,0);
    product(R_1,R_2,M_2);
  //M3 = A11 (B12 - B22)
    Add(A,zero,R_1,0,0,0,0);
    minus(B,B,R_2,0,sub_size,sub_size,sub_size);
    product(R_1,R_2,M_3);
  //M4 = A22 (B21 - B11)  
    Add(A,zero,R_1,sub_size,sub_size,0,0);
    minus(B,B,R_2,sub_size,0,0,0);
    product(R_1,R_2,M_4);
  //M5 = (A11 + A12) B22
    Add(A,A,R_1,0,0,0,sub_size);
    Add(B,zero,R_2,sub_size,sub_size,0,0);
    product(R_1,R_2,M_5);
  //M6 = (A21 - A11)(B11 + B12)  
    minus(A,A,R_1,sub_size,0,0,0);
    Add(B,B,R_2,0,0,0,sub_size);
    product(R_1,R_2,M_6);
  //M7 = (A12 - A22)(B21 + B22)  
    minus(A,A,R_1,0,sub_size,sub_size,sub_size);
    Add(B,B,R_2,sub_size,0,sub_size,sub_size);
    product(R_1,R_2,M_7);
    
    //C11 = M1+M4-M5+M7;
    Add(M_1,M_4,R_1,0,0,0,0);
    minus(R_1,M_5,R_1,0,0,0,0);
    Add(R_1,M_7,R_1,0,0,0,0);
    assign(C,R_1,0,0);

    //C12 = M3 + M5;
    Add(M_3,M_5,R_1,0,0,0,0);
    assign(C,R_1,0,sub_size);

    //C21 = M2 + M4;
    Add(M_2,M_4,R_1,0,0,0,0);
    assign(C,R_1,sub_size,0);
    
    //C22 = M1 - M2 + M3 + M6
    minus(M_1,M_2,R_1,0,0,0,0);
    Add(R_1,M_3,R_1,0,0,0,0);
    Add(R_1,M_6,R_1,0,0,0,0);
    assign(C,R_1,sub_size,sub_size);
}

void* paralle_strassenMM(void *rank) {
  // Do something here... seriously.
  //M1 = (A11 + A22)(B11 + B22) 
    int *rank_int_ptr = (int*)rank;
    int my_rank = *rank_int_ptr;
    switch(my_rank){
    case 0:{
      
      Add(A,A,R_1,0,0,sub_size,sub_size);
      Add(B,B,R_2,0,0,sub_size,sub_size);
      product(R_1,R_2,M_1);
      break;
    }
    //M2 = (A21 + A22)B11
    case 1:{
      Add(A,A,R_1,sub_size,0,sub_size,sub_size);
      Add(B,zero,R_2,0,0,0,0);
      product(R_1,R_2,M_2);
      break;
   }
  //M3 = A11 (B12 - B22)
    case 2:{
      Add(A,zero,R_1,0,0,0,0);
      minus(B,B,R_2,0,sub_size,sub_size,sub_size);
      product(R_1,R_2,M_3);
      break;
    }
  //M4 = A22 (B21 - B11)  
    case 3:{
      Add(A,zero,R_1,sub_size,sub_size,0,0);
      minus(B,B,R_2,sub_size,0,0,0);
      product(R_1,R_2,M_4);
      break;
    }
  //M5 = (A11 + A12) B22
    case 4:{
      Add(A,A,R_1,0,0,0,sub_size);
      Add(B,zero,R_2,sub_size,sub_size,0,0);
      product(R_1,R_2,M_5);
      break;
    }
  //M6 = (A21 - A11)(B11 + B12)  
    case 5:{
      minus(A,A,R_1,sub_size,0,0,0);
      Add(B,B,R_2,0,0,0,sub_size);
      product(R_1,R_2,M_6);
      break;
    }
  //M7 = (A12 - A22)(B21 + B22)  
    case 6:{
      minus(A,A,R_1,0,sub_size,sub_size,sub_size);
      Add(B,B,R_2,sub_size,0,sub_size,sub_size);
      product(R_1,R_2,M_7);
      break;
    }
    //C11 = M1+M4-M5+M7;
    case 7:{
      Add(M_1,M_4,R_1,0,0,0,0);
      minus(R_1,M_5,R_1,0,0,0,0);
      Add(R_1,M_7,R_1,0,0,0,0);
      assign(C_p,R_1,0,0);
      break;
    }
    //C12 = M3 + M5;
    case 8:{
      Add(M_3,M_5,R_1,0,0,0,0);
      assign(C_p,R_1,0,sub_size);
      break;
    }
    //C21 = M2 + M4;
    case 9:{
      Add(M_2,M_4,R_1,0,0,0,0);
      assign(C_p,R_1,sub_size,0);
      break;
    }  
    //C22 = M1 - M2 + M3 + M6
    case 10:{
      minus(M_1,M_2,R_1,0,0,0,0);
      Add(R_1,M_3,R_1,0,0,0,0);
      Add(R_1,M_6,R_1,0,0,0,0);
      assign(C_p,R_1,sub_size,sub_size);
      break;
    }
  }
}
// Allocate square matrix.
int **allocMatrix(int size) {
    size = size % 2 == 0 ? size : size + 1;
  int **matrix;
  matrix = (int **)malloc(size * sizeof(int *));
  for (int row = 0; row < size; row++) {
    matrix[row] = (int *)malloc(size * sizeof(int));
  }
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      matrix[i][j] = 0;
    }
  }
  return matrix;
}

// Allocate memory for all the matrixes, you will need to add code
// here to initialize any matrixes that you need.
void initMatrixes(int N) {
  A = allocMatrix(N); B = allocMatrix(N); C = allocMatrix(N); R = allocMatrix(N);
    sub_size = N % 2 == 0 ? N / 2 : (N + 1) / 2;
    M_1 = allocMatrix(sub_size);
    M_2 = allocMatrix(sub_size);
    M_3 = allocMatrix(sub_size);
    M_4 = allocMatrix(sub_size);
    M_5 = allocMatrix(sub_size);
    M_6 = allocMatrix(sub_size);
    M_7 = allocMatrix(sub_size);
    R_1 = allocMatrix(sub_size);
    R_2 = allocMatrix(sub_size);
    zero = allocMatrix(sub_size);
}

// Free up matrixes.
void cleanup() {
  free(A);
  free(B);
  free(C);
  free(R);
}

// Main method
int main(int argc, char* argv[]) {
  int N;
  double elapsedTime;
  pthread_t ids[NUM_THREADS];
  int ranks[NUM_THREADS];
  // checking parameters
  if (argc != 2 && argc != 4) {
    printf("Parameters: <N> [<fileA> <fileB>]\n");
    return 1;
  }
  // N = atoi(argv[1]);
  N = 2000;
  initMatrixes(N);

  // reading files (optional)
  // if(argc == 4){
  //   readMatrixFile(A,N,argv[2]);
  //   readMatrixFile(B,N,argv[3]);
  // } else {
    // Otherwise, generate two random matrix.
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
  A[i][j] = rand() % 5;
  B[i][j] = rand() % 5;
      }
    }
  // }
  printf("%d\n", N);
  // Do simple multiplication and time it.
  timerStart();
  simpleMM(N);
  printf("Simple MM took %ld ms\n", timerStop());

  // Do strassen multiplication and time it.
  timerStart();
  strassenMM(N);
  printf("Strassen MM took %ld ms\n", timerStop());

  timerStart();
  //paralle strassenMM
  for (int i=0; i < 7; i++) {
    ranks[i] = i;
    // HW3: Create the thread(s) here, calling nth_mat_vec.
    pthread_create(&ids[i], NULL, paralle_strassenMM, &ranks[i]);
  }
  for (int i=0; i < 7; i++) {
    // HW3: Join all the threads here.
    pthread_join(ids[i], NULL);
  }
  for (int i=7; i < 11; i++) {
    ranks[i] = i;
    // HW3: Create the thread(s) here, calling nth_mat_vec.
    pthread_create(&ids[i], NULL, paralle_strassenMM, &ranks[i]);
  }
  for (int i=7; i < 11; i++) {
    // HW3: Join all the threads here.
    pthread_join(ids[i], NULL);
  }
  printf("Parallel Strassen MM took %ld ms\n", timerStop());


  if (compareMatrix(C, R, N) != 0) {
    if (N < 20) {
      printf("\n\n------- MATRIX C\n");
      printMatrix(C,N);
      printf("\n------- MATRIX R\n");
      printMatrix(R,N);
    }
    printf("Matrix C doesn't match Matrix R, if N < 20 they will be printed above.\n");
  }
  if (compareMatrix(C, C_p, N) != 0) {
    if (N < 20) {
      printf("\n\n------- MATRIX C\n");
      printMatrix(C_p,N);
      printf("\n------- MATRIX R\n");
      printMatrix(C,N);
    }
    printf("Matrix C doesn't match Matrix R, if N < 20 they will be printed above.\n");
  }
  // stopping timer
  
  cleanup();
  return 0;
}
