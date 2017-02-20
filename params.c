#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include "shared.h"
#include "params.h"

#define MAX_STR_LEN 1024

int get_parameter_line(
    const char* param_name, const char* filename, char** param_line);

// Returns a parameter from the parameter file of type integer
int get_int_parameter(const char* param_name, const char* filename) 
{ 
  char line[MAX_STR_LEN];
  char* param_line = line;
  if(!get_parameter_line(param_name, filename, &param_line)) {
    TERMINATE("Could not find the parameter %s in file %s.\n", 
        param_name, filename);
  }

  int value;
  sscanf(param_line, "%d", &value);
  return value;
}

// Returns a parameter from the parameter file of type double
double get_double_parameter(const char* param_name, const char* filename) 
{ 
  char line[MAX_STR_LEN];
  char* param_line = line;
  if(!get_parameter_line(param_name, filename, &param_line)) {
    TERMINATE("Could not find the parameter %s in file %s.\n", 
        param_name, filename);
  }

  double value;
  sscanf(param_line, "%lf", &value);
  return value;
}

// Fetches all of the problem parameters
int get_problem_parameter(
    const int index, const char* filename, 
    char** keys, double* values, int* nkeys)
{
  char line[MAX_STR_LEN];
  char counter[MAX_STR_LEN];
  char* param_line = line;

  sprintf(counter, "problem_%d", index);
  if(!get_parameter_line(counter, filename, &param_line)) {
    return 0;
  }

  // Parse the kv pairs
  int key_index = 0;
  int parse_value = 0;
  for(size_t cc = 0; cc < strlen(param_line); ++cc) {
    if(param_line[cc] == '=') {
      // We are finished adding the key, time to get value
      parse_value = 1;
      key_index = 0;
    }
    else if(param_line[cc] != ' ') {
      if(parse_value) {
        sscanf(&param_line[cc], "%lf", &values[(*nkeys)++]);

        // Move the pointer to next space
        while(param_line[++cc] != ' ');
        parse_value = 0;
      }
      else {
        keys[*nkeys][key_index++] = param_line[cc];
      }
    }
  }

  return 1;
}

// Fetches a line from a parameter file with corresponding token
int get_parameter_line(
    const char* param_name, const char* filename, char** param_line)
{
  FILE* fp = fopen(filename, "r");
  if(!fp) {
    TERMINATE("Could not open the parameter file: %s.\n", filename);
  }

  char tok[MAX_STR_LEN];
  while (fgets(*param_line, MAX_STR_LEN, fp)) {
    skip_whitespace(param_line);

    // Read in the parameter name
    sscanf(*param_line, "%s", tok);
    if(strmatch(tok, param_name)) {
      *param_line += strlen(tok);
      skip_whitespace(param_line);
      fclose(fp);
      return 1;
    }
  }

  fclose(fp);
  return 0;
}

// Skips any leading whitespace
void skip_whitespace(char** param_line)
{
  for(unsigned long ii = 0 ; ii < strlen(*param_line); ++ii) {
    if(isspace((*param_line)[0])) {
      (*param_line)++;
    }
    else {
      return;
    }
  }
}

