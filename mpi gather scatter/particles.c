/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Student: 
 * Instructor: Bryan Mills, University of Pittsburgh
 * MPI particle-interaction code. 
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TAG 7
#define CONSTANT 777

// Particle-interaction constants
#define A 10250000.0
#define B 726515000.5
#define MASS 0.1
#define DELTA 1

// Random initialization constants
#define POSITION 0
#define VELOCITY 1

// Structure for shared properties of a particle (to be included in messages)
struct Particle{
  float x;
  float y;
  float mass;
  float fx;
  float fy;
};

// Headers for auxiliar functions
float random_value(int type);
void print_particles(struct Particle *particles, int n);
void interact(struct Particle *source, struct Particle *destination);
void compute_interaction(struct Particle *source, struct Particle *destination, int limit);
void compute_self_interaction(struct Particle *set, int size);
void merge(struct Particle *first, struct Particle *second, int limit);

// Main function
main(int argc, char** argv){
  
  int myRank;// Rank of process
  int p;// Number of processes
  int n;// Number of total particles
  int previous;// Previous rank in the ring
  int next;// Next rank in the ring
  int tag = TAG;// Tag for message
  int number;// Number of local particles
  int *sendcounts;  //the array of sendcounts.
  int *sendBuff;
  int *displs;
  struct Particle *globals;// Array of all particles in the system
  struct Particle *locals;// Array of local particles
  struct Particle *remotes;// Array of foreign particles
  int i;
  char *file_name;// File name
  MPI_Status status;// Return status for receive
  int j, rounds, initiator, sender;
  double start_time, end_time, send_start, send_end, send_duration = 0.0;
  MPI_Request send_request, recv_request;
  // checking the number of parameters
  // printf("p is %d\n", p);
  if(argc < 2){
    printf("ERROR: Not enough parameters\n");
    printf("Usage: %s <number of particles> [<file>]\n", argv[0]);
    exit(1);
  }
  
  // getting number of particles
  //asci to integer
  n = atoi(argv[1]);
  // printf("n is %d\n", n);
  // initializing MPI structures and checking p is odd
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  // printf("p is %d\n", p);
  if(p % 2 == 0){
    p = p - 1;
    if(myRank == p){
      MPI_Finalize();
      return 0;
    }
  }
  
  // printf("my Rank is %d\n", myRank);
  srand(myRank+myRank*CONSTANT);

  // acquiring memory for particle arrays
  number = n / p;
  int rem = n % p;
  

  int sum = 0;
  // printf("%d\n", number);
  //init send count
  sendBuff = (int *)malloc(sizeof(int) * p);
  sendcounts = (int *)malloc(sizeof(int) * p);
  displs = (int *)malloc(sizeof(int) * p);
  for (i = 0; i < p; i++) {
        sendcounts[i] = number;
        if (rem > 0) {
            sendcounts[i]++;
            rem--;
        }
        sendBuff[i] = sendcounts[i] * (sizeof(struct Particle)) / sizeof(float);
        displs[i] = sum * (sizeof(struct Particle)) / sizeof(float);
        sum += sendcounts[i];
  }
  // printf("%d\n", number);
  // for(i = 0; i < p; i++){
  //   sendBuff[i] = sendcounts[i] * (sizeof(struct Particle)) / sizeof(float);
  //   displs[i] = displs[i] * (sizeof(struct Particle)) / sizeof(float);
  // }
  //ensure numnber could be safe for every processor
  if(n % p) number++;
  locals = (struct Particle *) malloc(number * sizeof(struct Particle));
  remotes = (struct Particle *) malloc(number * sizeof(struct Particle));
    
  // checking for file information

  if(argc == 3){
    if(myRank == 0){
      globals = (struct Particle *) malloc(n * sizeof(struct Particle));
      // YOUR CODE GOES HERE (reading particles from file)
      read_file(globals,n,argv[2]);

    }
    

    // To send/recv (or scatter/gather) you will need to learn how to
    // transfer structs of floats, treat it as a contiguous block of
    // floats. Here is an example:

    // hint: because your nodes need to both send and receive you
    // might consider asyncronous send/recv.

    // YOUR CODE GOES HERE (distributing particles among processors)
    //rank 0 distribute the global particles to other processors.
// MPI_Scatterv(&data, sendcounts, displs, MPI_CHAR, &rec_buf, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Scatterv(globals, 
              sendBuff, 
              displs,
              MPI_FLOAT,
              locals,
              number * (sizeof (struct Particle)) / sizeof(float), 
              MPI_FLOAT,
              0,
              MPI_COMM_WORLD);
  } else {
    // random initialization of local particle array
    globals = (struct Particle *) malloc(n * sizeof(struct Particle));
    for(j = 0; j < sendcounts[myRank]; j++){
      locals[j].x = random_value(POSITION);
      locals[j].y = random_value(POSITION);
      locals[j].fx = 0.0;
      locals[j].fy = 0.0;
      locals[j].mass = MASS;
    }
  }
 
    

  // starting timer
  if(myRank == 0){
    printf("num processors %d\n", p);
    start_time = MPI_Wtime();
  }

  // YOUR CODE GOES HERE (ring algorithm)
  

  int next_rank = (myRank + 1) % p;
  int previous_rank = (myRank + p - 1) % p;
  //first send local to next remote
  MPI_Isend(locals,
             number * (sizeof (struct Particle)) / sizeof(float),
             MPI_FLOAT,
             next_rank,
             tag,
             MPI_COMM_WORLD,
             &send_request);
           
  MPI_Irecv(remotes,
             number * (sizeof (struct Particle)) / sizeof(float),
             MPI_FLOAT,
             previous_rank,
             tag,
             MPI_COMM_WORLD,
             &recv_request);
  MPI_Wait(&recv_request, 
                   &status);

  compute_interaction(locals,remotes,number);


  for(i = 1; i < (p - 1) / 2; i++){     
    if(myRank == 0) {      
      // print_particles(globals,n);
      send_start = MPI_Wtime();
    }
    MPI_Isend(remotes,
               number * (sizeof (struct Particle)) / sizeof(float),
               MPI_FLOAT,
               next_rank,
               tag,
               MPI_COMM_WORLD,
               &send_request);
                
    MPI_Irecv(remotes,
             number * (sizeof (struct Particle)) / sizeof(float),
             MPI_FLOAT,
             previous_rank,
             tag,
             MPI_COMM_WORLD,
             &recv_request);

    MPI_Wait(&recv_request, 
                   &status);
    if(myRank == 0) {      
      // print_particles(globals,n);
      send_end = MPI_Wtime();
      send_duration += send_end - send_start;
    }
    compute_interaction(locals,remotes,number);
  }
  if(myRank == 0){
    printf("all send: %f seconds\n", send_duration);
  }
  // printf("I recv 2nd my Rank is %d\n", myRank);
  //send back to original processor
  int start_process = (myRank + p - (p - 1) / 2) % p;
  int end_process = (myRank + (p - 1) / 2) % p;
  MPI_Isend(remotes,
               number * (sizeof (struct Particle)) / sizeof(float),
               MPI_FLOAT,
               start_process,
               tag,
               MPI_COMM_WORLD,
               &send_request);
                
  MPI_Irecv(remotes,
           number * (sizeof (struct Particle)) / sizeof(float),
           MPI_FLOAT,
           end_process,
           tag,
           MPI_COMM_WORLD,
           &recv_request);
  MPI_Wait(&recv_request, 
                   &status);

  //merge result
  merge(locals, remotes, number);
  //compute local
  compute_self_interaction(locals,number);

    //processors receive local processors 
    
  // stopping timer
  if(myRank == 0){
    end_time = MPI_Wtime();
    printf("Duration: %f seconds\n", (end_time-start_time));
  }
  
  // printing information on particles
  if(argc == 3){
    // YOUR CODE GOES HERE (collect particles at rank 0)

    //why sendBuff[myRank]? every processor send their own buff size!
    MPI_Gatherv(locals, 
                  sendBuff[myRank], 
                  MPI_FLOAT,
                  globals,
                  sendBuff, 
                  displs,
                  MPI_FLOAT,
                  0,
                  MPI_COMM_WORLD);

    if(myRank == 0) {      
      print_particles(globals,n);
      end_time = MPI_Wtime();
      printf("all done: %f seconds\n", (end_time-start_time));
    }
  }
  else{
    MPI_Gatherv(locals, 
                sendBuff[myRank], 
                MPI_FLOAT,
                globals,
                sendBuff, 
                displs,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);
    if(myRank == 0) {      
      // print_particles(globals,n);
      end_time = MPI_Wtime();
      printf("all done: %f seconds\n", (end_time-start_time));
    }
  }

  // finalizing MPI structures
  MPI_Finalize();
}

// Function for random value generation
float random_value(int type){
  float value;
  switch(type){
  case POSITION:
    value = (float)rand() / (float)RAND_MAX * 100.0;
    break;
  case VELOCITY:
    value = (float)rand() / (float)RAND_MAX * 10.0;
    break;
  default:
    value = 1.1;
  }
  return value;
}

// Function for printing out the particle array
void print_particles(struct Particle *particles, int n){
  int j;
  printf("Index\tx\ty\tmass\tfx\tfy\n");
  for(j = 0; j < n; j++){
    printf("%d\t%f\t%f\t%f\t%f\t%f\n",j,particles[j].x,particles[j].y,particles[j].mass,particles[j].fx,particles[j].fy);
  }
}

// Function for computing interaction among two particles
// There is an extra test for interaction of identical particles, in which case there is no effect over the destination
void interact(struct Particle *first, struct Particle *second){
  float rx,ry,r,fx,fy,f;
  if(first == NULL || second == NULL) return;
  // computing base values
  rx = first->x - second->x;
  ry = first->y - second->y;
  r = sqrt(rx*rx + ry*ry);

  if(r == 0.0)
    return;

  f = A / pow(r,6) - B / pow(r,12);
  fx = f * rx / r;
  fy = f * ry / r;

  // updating sources's structure
  first->fx = first->fx + fx;
  first->fy = first->fy + fy;
  
  // updating destination's structure
  second->fx = second->fx - fx;
  second->fy = second->fy - fy;

}

// Function for computing interaction between two sets of particles
void compute_interaction(struct Particle *first, struct Particle *second, int limit){
  int j,k;
  
  for(j = 0; j < limit; j++){
    for(k = 0; k < limit; k++){
      interact(&first[j],&second[k]);
    }
  }
}

// Function for computing interaction between two sets of particles
void compute_self_interaction(struct Particle *set, int size){
  int j,k;
  
  for(j = 0; j < size; j++){
    for(k = j+1; k < size; k++){
      interact(&set[j],&set[k]);
    }
  }
}

// Function to merge two particle arrays
// Permanent changes reside only in first array
void merge(struct Particle *first, struct Particle *second, int limit){
  int j;
  
  for(j = 0; j < limit; j++){
    first[j].fx += second[j].fx;
    first[j].fy += second[j].fy;
  }
}

// Reads particle information from a text file
int read_file(struct Particle *set, int size, char *file_name){
  FILE *ifp, *ofp;
  char *mode = "r";
  ifp = fopen(file_name, mode);

  if (ifp == NULL) {
    fprintf(stderr, "Can't open input file!\n");
    return 1;
  }

  int i=0;
  // reading particle values
  for(i=0; i<size; i++){
    fscanf(ifp, "%f\t%f\t%f", &set[i].x, &set[i].y, &set[i].mass);
    set[i].fx = 0.0;
    set[i].fy = 0.0;
  }
  
  // closing file
  fclose(ifp);

  return 0;
}

