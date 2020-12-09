#!/bin/bash
#PBS -q workq
#PBS -l nodes=15:ppn=16
#PBS -l walltime=72:00:00
#PBS -o matrix_gen
#PBS -j oe
#PBS -N graphmmp
#PBS -A hpc_michal01

HOST=$(echo $LST | cut -d '.' -f1).host
LOG=$(echo $LST | cut -d '.' -f1).log

cd /work/jfeins1/mastro/

module load gnuparallel/20180222/INTEL-18.0.0

cat $PBS_NODEFILE | uniq > /work/jfeins1/maestro/$HOST

parallel --jobs $PBS_NUM_PPN --slf /work/jfeins1/maestro/$HOST --wd /work/jfeins1/maestro/ --joblog /work/jfeins1/maestro/$LOG --resume --colsep ' ' -a /work/jfeins1/maestro/$LST sh /work/jfeins1/maestro/python_wrapper_gen_encoding.sh {}

rm /work/jfeins1/maestro/$HOST

