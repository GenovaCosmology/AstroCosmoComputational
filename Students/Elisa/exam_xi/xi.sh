#!/bin/bash

source /cvmfs/euclid-dev.in2p3.fr/CentOS7/EDEN-3.0/bin/activate
source /cvmfs/euclid-dev.in2p3.fr/CentOS7/EDEN-3.0/etc/profile.d/elementsenv.sh
export OMP_NUM_THREADS=40

## pre rec
E-Run --debug LE3_GC_TwoPointCorrelation LE3_GC_ComputeTwoPointCorrelation --parfile xi/parameters_xi.ini --workdir ./

# span[hosts=1]