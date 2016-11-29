#pragma once 

#include "state.h"
#include "mesh.h"

typedef enum
{

} State;

// Initialise the state for the problem
void initialise_state(
    State* state, Mesh* mesh);

// Deallocate all of the state memory
void finalise_state(
    State* state);

