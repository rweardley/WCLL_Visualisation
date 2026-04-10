module load cray-python/3.11.7

set -x

sbatch -A fus166 -q normal -t 02:00:00 \
  -N60 --ntasks-per-node=56 \
  -J split_flds \
  -o log_%x-%j.out  \
  --wrap='module load cray-python/3.11.7; srun --cpu-bind=cores --threads-per-core=1 --label python3 split_m4_v2.py'
