#pragma once

// Returns a parameter from the parameter file of type integer
int get_int_parameter(const char* param_name, const char* filename);

// Returns a parameter from the parameter file of type double
double get_double_parameter(const char* param_name, const char* filename);

// Skips any leading whitespace
void skip_whitespace(char** line);

