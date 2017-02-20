#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include "shared.h"
#include "params.h"

#define MAX_STR_LEN 1024

int get_parameter_line(
    const char* param_name, const char* filename, char** param_line);

// Skips a token in the stream
void get_tok(char** param_line, char tok[MAX_STR_LEN]) 

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
    const int index, const int ndims, const char* filename, char** variables, 
    double** values, int* nvariables)
{
  char tok[MAX_STR_LEN];
  char line[MAX_STR_LEN];
  char counter[MAX_STR_LEN];
  char* param_line = line;
  int max_variables = *nvariables;

  sprintf(counter, "problem_%d", index);
  if(get_parameter_line(counter, filename, &param_line)) {
    nvariables = 0;

    // Parse the dependent variables
    for(int ii = 0; ii < max_variables; ++ii) {
      get_tok(&param_line, tok);

      if(!strmatch(tok, ":")) {
        size_t token_len = 0;
        size_t value_start = 0;
        for(size_t cc = 0; cc < strlen(tok); ++cc) {
          if(!token_len && (tok[cc] == ' ' || tok[cc] == '=')) {
            token_len = cc;
          }
          else if(token_len && (tok[cc] != ' ' && tok[cc] != '=')) {
            value_start = cc;
          }
        }
        strncpy(variables[(*nvariables)], tok, token_len);
        sscanf(tok+value_start, "%lf", values[(*nvariables)]);
        (*nvariables)++;
      }
      else {
        for(int ii = 0; ii < 2*ndims; ++ii) {
          // Parse the bounds
          skip_whitespace(&param_line);
          sscanf(param_line, "%d", &bounds[ii]);
        }
        return 1;
      }
    }

    TERMINATE(
        "Problem configuration is incorrect %s.\n", param_line);
  }

  return 0;
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
    get_tok(param_line, tok);
    if(strmatch(tok, param_name)) {
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

// Skips a token in the stream
void get_tok(char** param_line, char tok[MAX_STR_LEN]) 
{
  sscanf(*param_line, "%s", tok);
  *param_line += strlen(tok);
  skip_whitespace(param_line);
}

#if 0
// Make sure the user has entered sane values
assert((*xpos) >= 0 && (*xpos) <= global_nx);
assert((*ypos) >= 0 && (*ypos) <= global_ny);
assert((*width) > 0 && (*xpos)+(*width) <= global_nx);
assert((*height) > 0 && (*ypos)+(*height) <= global_ny);
#endif // if 0

