#pragma once

// Returns a parameter from the parameter file of type integer
int get_int_parameter(const char* param_name, const char* filename);

// Returns a parameter from the parameter file of type double
double get_double_parameter(const char* param_name, const char* filename);

// Fetches all of the problem parameters
int get_key_value_parameter(
    const char* specifier, const char* filename, 
    char** keys, double* values, int* nkeys);

// Skips any leading whitespace
void skip_whitespace(char** line);

