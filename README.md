# arch
A shared architectural code for a suite of mini-apps

# Purpose

The purpose of this code is to wrap up the shared architectural concerns of a suite of mini-apps. The mini-apps are all designed to support algorithmic optimisation, 

# Build

Although the architectural code is not intended to be built independently of the other applications, it acts as a shell that the applications reside within. The individual mini-apps cannot be built without the architectural code.

To build an application, clone the repository into the root of the arch project. E.g.

git clone git@github.com:uob-hpc/arch
cd arch
git clone git@github.com:uob-hpc/neutral
cd neutral

An example compilation string for the dependent applications

make -j DEBUG=<yes/no> KERNELS=omp3 COMPILER=INTEL

Please refer to the individual applications in order to determine the specific build and execution steps.

# Integration of new mini-apps

In many places the architectural code is generic, but it's not currently clear how well this will extend to future use cases. Over time this section will be updated with the steps that must be taken for future mini-app integrations to fit within the general practices of the code.
