#include <stdio.h>
#include <stdlib.h>
#include "params.h"

#define MAX_KEYS 10
#define MAX_STR_LEN 128

void init_problem( 
    const char* filename)
{
  // TODO: CHECK THE BOUNDS OF THESE DODGEY MALLOCS
  double* values = (double*)malloc(sizeof(double)*MAX_KEYS);
  char* keys_space = (char*)malloc(sizeof(char)*MAX_KEYS*(MAX_STR_LEN+1));
  char** keys = (char**)malloc(sizeof(char*)*MAX_KEYS);
  for(int ii = 0; ii < MAX_KEYS; ++ii) {
    keys[ii] = &keys_space[ii*(MAX_STR_LEN+1)];
  }

  int nkeys = 0;
  get_problem_parameter(0, filename, keys, values, &nkeys);

  for(int ii = 0; ii < nkeys; ++ii) {
    printf("%s=%lf\n", keys[ii], values[ii]);
  }

#if 0
  int index = 0;
  while(get_problem_parameter(
    index++, filename, char** dependent_keys, 
    int* ndependent_keys, int* xpos, int* ypos, 
    int* width, int* height, double* value))
  {

  }

  for(int ii = 0; ii < MAX_STR_LEN
#endif // if 0
}

