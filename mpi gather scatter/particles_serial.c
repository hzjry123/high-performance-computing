/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Student: 
 * Instructor Bryan Mills, University of Pittsburgh
 * Serial particle-interaction code. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

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
void print_particles(struct Particle *set, int n);
void print_particles_input(struct Particle *particles, int n);
void interact(struct Particle *source, struct Particle *destination);
void compute_self_interaction(struct Particle *set, int size);
int read_file(struct Particle *set, int size, char *file_name);

// Main function
int main(int argc, char** argv){
  int n;// Number of total particles
  struct Particle *locals;// Array of local particles
  char *file_name;// File name

  // checking the number of parameters
  if(argc < 2){
    printf("ERROR: Not enough parameters\n");
    printf("Usage: %s <number of particles> [<file>]\n", argv[0]);
    exit(1);
  }
  
  // getting number of particles
  //asci to integer
  n = atoi(argv[1]);
  //random seed
  srand(CONSTANT);

  // acquiring memory for particle arrays
  locals = (struct Particle *) malloc(n * sizeof(struct Particle));

  // checking for file information
  if(argc == 3){
    read_file(locals,n,argv[2]);
  } else {
    // random initialization of local particle array
    for(int j = 0; j < n; j++){
      locals[j].x = random_value(POSITION);
      locals[j].y = random_value(POSITION);
      locals[j].mass = MASS;
      locals[j].fx = 0.0;
      locals[j].fy = 0.0;
    }
  }
  
  // starting timer
  timerStart();

  // particle interaction
  compute_self_interaction(locals,n);
  
  // stopping timer
  double duration = timerStop();
  
  // printing information on particles
  if(argc == 3) {
    print_particles(locals,n);
  }

  printf("Duration: %f seconds\n", duration);
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

// Function for printing out the particle array
void print_particles_input(struct Particle *particles, int n){
  int j;
  printf("x\ty\tmass\n");
  for(j = 0; j < n; j++){
    printf("%f\t%f\t%f\n",particles[j].x,particles[j].y,particles[j].mass);
  }
}

// Function for computing interaction among two particles
// There is an extra test for interaction of identical particles, in which case there is no effect over the destination
void interact(struct Particle *first, struct Particle *second){
  float rx,ry,r,fx,fy,f;

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
void compute_self_interaction(struct Particle *set, int size){
  int j,k;
  
  for(j = 0; j < size; j++){
    for(k = j+1; k < size; k++){
      interact(&set[j],&set[k]);
    }
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

  // reading particle values
  for(int i=0; i<size; i++){
    fscanf(ifp, "%f\t%f\t%f", &set[i].x, &set[i].y, &set[i].mass);
    set[i].fx = 0.0;
    set[i].fy = 0.0;
  }
  
  // closing file
  fclose(ifp);

  return 0;
}
