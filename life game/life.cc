/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD (bmills@cs.pitt.edu)
 * Students:
 * Implement openmp verions of conway's game of life.
 */

#include "timer.h"
#include "io.h"
#ifdef _OPENMP
# include <omp.h>
#endif 

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
// Function implementing Conway's Game of Life
void conway(int **World, int N, int M){
    // STUDENT: IMPLEMENT THE GAME HERE, make it parallel!
    //iterate M times


    int** state = allocMatrix(N);
    for(int p = 0; p < M; p++){
        //cal elements' state
        #pragma omp parallel for 
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                //get corner case;
                int i_start = i - 1 >= 0 ? i-1 : 0;
                int j_start = j - 1 >= 0 ? j-1 : 0;
                int i_end = i + 1 < N ? i+1 : N - 1;
                int j_end = j + 1 < N ? j+1 : N - 1;
                for(;i_start <= i_end; i_start++){
                    for(int tmp = j_start; tmp <= j_end; tmp++){
                        if(World[i_start][tmp] == 1){
                            //cal neighbor
                            state[i][j]++;
                        }
                    }
                }
            }
        }

        //renew matrix
        #pragma omp parallel for 
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                //if element is alive
                if(World[i][j] == 1){
                    //if elment itself is alive then state-- as it consider itself as a live neighbor
                    state[i][j]--;
                    //state < 2 or state > 3 element die.
                    if (state[i][j] < 2 || state[i][j] > 3) {
                        World[i][j] = 0;
                    }
                }
                //if element is die and can be live
                else if(World[i][j] == 0 && state[i][j] == 3)
                {
                    World[i][j] = 1;
                    
                }
                //reset state matrix
                state[i][j] = 0;
            }
            
        }
    }
}


// Main method
int main(int argc, char* argv[]) {
    int N,M;
    int **World;
    double elapsedTime;
    
    // checking parameters
    if (argc != 3 && argc != 4) {
        printf("Parameters: <N> <M> [<file>]\n");
        return 1;
    }
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    
    
    // allocating matrices
    World = allocMatrix(N);
    
    // reading files (optional)
    if(argc == 4){
        readMatrixFile(World,N,argv[3]);
    } else {
        // Otherwise, generate two random matrix.
        srand (time(NULL));
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                World[i][j] = rand() % 2;
            }
        }
    }

    // starting timer
    timerStart();
    
    // calling conway's game of life
    conway(World,N,M);
    
    // stopping timer
    elapsedTime = timerStop();
    if(N < 16)
    printMatrix(World,N);
    
    printf("Took %ld ms\n", timerStop());
    
    // releasing memory
    for (int i=0; i<N; i++) {
        delete [] World[i];
    }
    delete [] World;
    //print max threads
    # ifdef _OPENMP
    int total = omp_get_max_threads();
    # else
    int total = 0;
    # endif 
    printf("total thread %d \n", total);
    return 0;
}


