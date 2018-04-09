/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems Spring of 2017
 * Instructor Bryan Mills, PhD
 * Student:
 * Implement Pthreads version of trapezoidal approximation.
 */

#include <stdio.h>
#include "timer.h"
#include <pthread.h>

// Update thread count here, currently implementation is not
// threadsafe. Go ahead see what happens if you change this > 1.
#define NUM_THREADS 100

// Global variables to make coverting to pthreads easier :)
double a;
double b;
int n;
double approx;
int flag = 0;
pthread_mutex_t mutex;
// Actual areas under the f(x) = x^2 curves, for you to check your
// values against.
double static NEG_1_TO_POS_1 = 0.66666666666667;
double static ZERO_TO_POS_10 = 333.333;

// f function is defined a x^2
double f(double a) {
    return a * a;
}

// Parallelize the loop in this function.
// * Which variable needs to be protected?
// * Do so using both a busy-wait first.
// * Then use a mutex and observe the time difference.
void* trap_loop(void *rank) {
    int *rank_int_ptr = (int*) rank;
    int my_rank = *rank_int_ptr;
    double local_x_i;
    double my_approx = 0.0;
    double h = (b-a) / n;
    int step = n / NUM_THREADS;
    int start = step * my_rank;
    int end = step * (my_rank + 1) -1;
    double local_approx = 0.0;
    
    for(int i = start; i < end; i++) {
        local_x_i = a + i*h;
        my_approx += f(local_x_i);
    }

    // busy-waiting version
    
    // int test = 0;
    // while (test != my_rank){
    //     //without mutex and test, it can run perfectly on my XCode
    //     //but always deadlock on comet.
    //     //then I tried many approaches, finally it worked out.
    //     //thanks god.
    //     pthread_mutex_lock(&mutex);
    //     test = flag;
    //     pthread_mutex_unlock(&mutex);
    // };
    // approx = approx + my_approx;
    // flag = (flag+1) % NUM_THREADS;

    // mutex version

    pthread_mutex_lock(&mutex);
    approx = approx + my_approx;
    pthread_mutex_unlock(&mutex);
    return NULL;

}

void trap() {
    double x_i;
    double h = (b-a) / n;
    
    approx = ( f(a) - f(b) ) / 2.0;
    
    pthread_t ids[NUM_THREADS];
    int ranks[NUM_THREADS];
    for (int i=0; i < NUM_THREADS; i++) {
        ranks[i] = i;
        pthread_create(&ids[i], NULL, trap_loop, &ranks[i]);
    }
    for (int i=0; i < NUM_THREADS; i++) {
        pthread_join(ids[i], NULL);
    }
    approx = h*approx;
    return;
}

int main() {
    // Example 1 [-1,1]
    a = -1.0;
    b = 1.0;
    n = 1000000000;
    timerStart();
    trap();
    printf("Took %ld ms\n", timerStop());
    printf("a:%f\t b:%f\t n:%d\t actual:%f\t approximation:%f\n", a, b, n, NEG_1_TO_POS_1, approx);
    
    // Example 2 [0,10]
    a = 0.0;
    b = 10.0;
    n = 1000000000;
    timerStart();
    trap();
    printf("Took %ld ms\n", timerStop());
    printf("a:%f\t b:%f\t n:%d\t actual:%f\t approximation:%f\n", a, b, n, ZERO_TO_POS_10, approx);
    
    return 0;
}
