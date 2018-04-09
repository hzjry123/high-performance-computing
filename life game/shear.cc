/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD (bmills@cs.pitt.edu)
 * STUDENTS: Implement OpenMP parallel shear sort.
 */

#include <math.h>
#include "timer.h"
#include "io.h"

#ifdef _OPENMP
# include <omp.h>
#endif 
// using namespace std;
#define MAX_VALUE 10000
void sort_row(int **A,int i,int M){
    #pragma omp parallel for 
    for(int j = 0; j < M - 1; j++){
        for(int k = 0; k < M - 1 - j; k++){
          //if row is even, sort in ascend order.
          //if row is odd, sort in descend order.
            int flag = !((i % 2 == 0)^(A[i][k] > A[i][k + 1]));
            if (flag == 1)
            {//swap element
              int temp = A[i][k];
              A[i][k] = A[i][k + 1];
              A[i][k + 1] = temp;
            }
        }
    }
}
//sort column in ascend order
void sort_column(int **A,int i,int M){
    #pragma omp parallel for 
    for(int j = 0; j < M - 1; j++){
        for(int k = 0; k < M - j - 1; k++){
            if (A[k][i] > A[k + 1][i])
            {
              int temp = A[k][i];
              A[k][i] = A[k + 1][i];
              A[k + 1][i] = temp;
            }
        }
    }
}
void shear_sort(int **A, int M) {
  // Students: Implement parallel shear sort here.
    int p = 0;
    while(p < M){
      //sort row first
        #pragma omp parallel for 
        for(int i = 0; i< M; i++){
            sort_row(A,i, M);
        }
        //sort column then
        #pragma omp parallel for
        for(int i = 0; i < M; i++)
            sort_column(A,i, M);
        p++;

    }
}

// Allocate square matrix.
int **allocMatrix(int size) {
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

// Main method

int main(int argc, char* argv[]) {
    int N, M;
    int **A;
    double elapsedTime;
   
  // checking parameters
  if (argc != 2 && argc != 3) {
    printf("Parameters: <N> [<file>]\n");
    return 1;
  }
  N = atoi(argv[1]);
  M = (int) sqrt(N);
  if(N != M*M){
    printf("N has to be a perfect square!\n");
    exit(1);
  }

  // allocating matrix A
  A = allocMatrix(M);

  // reading files (optional)
  if(argc == 3){
    readMatrixFile(A,M,argv[2]);
  } else {
    srand (time(NULL));
    // Otherwise, generate random matrix.
    for (int i=0; i<M; i++) {
      for (int j=0; j<M; j++) {
	A[i][j] = rand() % MAX_VALUE;
      }
    }
  }
  
  // starting timer
  timerStart();
  // calling shear sort function
  shear_sort(A,M);
  // stopping timer
  elapsedTime = timerStop();

  // print if reasonably small
  if (M <= 10) {
      printMatrix(A,M);
      printf("\n");
  }
  printf("Took %ld ms\n", timerStop());

  // releasing memory
  for (int i=0; i<M; i++) {
    delete [] A[i];
  }
  delete [] A;
     //print max threads
  # ifdef _OPENMP
  int total = omp_get_max_threads();
  # else
  int total = 0;
  # endif 
  printf("total thread %d \n", total);
  return 0;
 
}

