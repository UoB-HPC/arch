#include <stdlib.h>

#define MAX_DEPENDENT_VARIABLES 10
#define MAX_STR_LEN 128

void init_problem(
    const char* filename
    )
{
  char* variables_space = 
    (char*)malloc(sizeof(char)*MAX_DEPENDENT_VARIABLES*MAX_STR_LEN);
  double* values_space = 
    (double*)malloc(sizeof(double)*MAX_DEPENDENT_VARIABLES);
  char** variables;
  double** values;
  for(int ii = 0; ii < MAX_DEPENDENT_VARIABLES; ++ii) {
    variables[ii] = &variables_space[ii*MAX_STR_LEN];
    values[ii] = &values_space[ii*MAX_STR_LEN];
  }

  int index = 0;
  while(get_problem_parameter(
    index++, filename, char** dependent_variables, 
    int* ndependent_variables, int* xpos, int* ypos, 
    int* width, int* height, double* value))
  {

  }

  for(int ii = 0; ii < MAX_STR_LEN
}

