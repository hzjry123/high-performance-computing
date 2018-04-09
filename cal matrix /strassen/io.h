/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD 
 * Original Code Esteban Meneses (converted to c)
 * Input/output operations for matrices.
 */

#include<stdio.h>

// Reads a matrix from text file
int readMatrixFile(int **matrix, int N, char *filename){

  FILE *ifp, *ofp;
  char *mode = "r";
  ifp = fopen(filename, mode);

  if (ifp == NULL) {
    fprintf(stderr, "Can't open input file in.list!\n");
    return 1;
  }

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      if (!fscanf(ifp, "%d", &matrix[i][j])) {
	fprintf(stderr, "Unable to read!\n");
	return 1;
      }
    }
  }
  return 0;
}

// Prints matrix to standard output
void printMatrix(int **matrix, int N){
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      printf("%d\t", matrix[i][j]);
    }
    printf("\n");
  }
}

// Compares the two matrixes and returns the number of cells which are
// not equal.
int compareMatrix(int **M1, int **M2, int N) {
  int c = 0;
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      if (M1[i][j] != M2[i][j]) {
	c++;
      }
    }
  }
  return c;
}
