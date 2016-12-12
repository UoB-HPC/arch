
OMP_NUM_THREADS=1 srun -n 1 ./multi.cuda 5000 5000 50 > strong_5000_50_storm_k40/OUT_gpu1
echo "Finished 1 GPU"
OMP_NUM_THREADS=1 srun -n 2 ./multi.cuda 5000 5000 50 > strong_5000_50_storm_k40/OUT_gpu2
echo "Finished 2 GPU"
OMP_NUM_THREADS=1 srun -n 3 ./multi.cuda 5000 5000 50 > strong_5000_50_storm_k40/OUT_gpu3
echo "Finished 3 GPU"
OMP_NUM_THREADS=1 srun -n 4 ./multi.cuda 5000 5000 50 > strong_5000_50_storm_k40/OUT_gpu4
echo "Finished 4 GPU"
OMP_NUM_THREADS=1 srun -n 5 ./multi.cuda 5000 5000 50 > strong_5000_50_storm_k40/OUT_gpu5
echo "Finished 5 GPU"
OMP_NUM_THREADS=1 srun -n 6 ./multi.cuda 5000 5000 50 > strong_5000_50_storm_k40/OUT_gpu6
echo "Finished 6 GPU"
OMP_NUM_THREADS=1 srun -n 7 ./multi.cuda 5000 5000 50 > strong_5000_50_storm_k40/OUT_gpu7
echo "Finished 7 GPU"
OMP_NUM_THREADS=1 srun -n 8 ./multi.cuda 5000 5000 50 > strong_5000_50_storm_k40/OUT_gpu8
echo "Finished 8 GPU"
