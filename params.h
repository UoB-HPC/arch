#pragma once

enum { ENERGY_KEY, DENSITY_KEY, TEMPERATURE_KEY };

// Returns a parameter from the parameter file of type integer
int get_int_parameter(const char* param_name, const char* filename);

// Returns a parameter from the parameter file of type double
double get_double_parameter(const char* param_name, const char* filename);

// Skips any leading whitespace
void skip_whitespace(char** line);

#ifdef __cplusplus
extern "C" {
#endif

// Fetches all of the problem parameters
int get_key_value_parameter(
    const char* specifier, const char* filename, 
    char* keys, double* values, int* nkeys);

#ifdef __cplusplus
}
#endif

