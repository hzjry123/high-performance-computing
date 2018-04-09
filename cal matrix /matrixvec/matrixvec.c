#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 3
const int M = 10;
const int N = 2;

int A [M][N] = { {1,4},
		 {2,5},
		 {3,6},
		 {1,2},
		 {1,2},
		 {1,2},
		 {3,4},
		 {9,5},
		 {9,3},
		 {1,2}};
int X [N] = { 3, 2 };
int Y [M];

// HW3: Implement this function.
void* nth_mat_vect(void *rank) {
  // Hint: Look in the slides and/or book.
  int *rank_int_ptr = (int*)rank;
  int my_rank = *rank_int_ptr;
  int local_m = (( M - 1) / NUM_THREADS) + 1;
  //local_m = 5, 
  int my_first_row = (my_rank * local_m);
  int my_last_row = (my_rank + 1) * local_m - 1;
  
  for(int i = my_first_row; i <= my_last_row && i < M; i++){
   	Y[i] = 0.0;
	for(int j = 0; j < N; j++){
		Y[i] += A[i][j] * X[j];
	}
  }
 return NULL;
}

// Helper method to print vector to standard out.
void printVector(int vector[], int n){
  for(int i=0; i<n; i++){
    printf("%d \n", vector[i]);
  }
}

// Helper method to prints matrix to standard out.
void printMatrix(int matrix[M][N], int m, int n){
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      printf("%d\t", matrix[i][j]);
    }
    printf("\n");
  }
}

int main() {
  pthread_t ids[NUM_THREADS];
  int ranks[NUM_THREADS];
  for (int i=0; i < NUM_THREADS; i++) {
    ranks[i] = i;
    // HW3: Create the thread(s) here, calling nth_mat_vec.
    pthread_create(&ids[i], NULL, nth_mat_vect, &ranks[i]);
  }
  for (int i=0; i < NUM_THREADS; i++) {
    // HW3: Join all the threads here.
    pthread_join(ids[i], NULL);
  }
  printf("MATRIX A=\n");
  printMatrix(A, N, M);
  printf("VECTOR X=\n");
  printVector(X, N);
  printf("VECTOR Y=\n");
  printVector(Y, M);
  return 0;
}

