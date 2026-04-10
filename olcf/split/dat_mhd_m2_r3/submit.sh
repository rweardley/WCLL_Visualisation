module load cray-python/3.11.7

set -x

#pip install --upgrade pip
#pip3 install pyvista

#srun -A nfu106 -t 00:30:00 -q debug -N1 -n56 -J split_fld -o %x-%j.log python3 split_m2.py
#srun -A nfu106 -t 00:30:00 -q normal -N1 -n56 -J split_fld -o log_%x-%j.out python3 split_m2.py

#sbatch -A fus166 -q normal -t 00:30:00 -N1 \
#       --ntasks-per-node=56 \
#       -J split_flds \
#       -o log_%x-%j.out \
#       --wrap="module load cray-python/3.11.7 && srun --cpu-bind=cores --threads-per-core=1 --label python3 split_m2_v2.py"
#       --wrap="module load cray-python/3.11.7 && srun --label python3 split_m2.py"

sbatch -A fus166 -q normal -t 00:30:00 \
  -N1 --ntasks-per-node=56 \
  -J split_flds \
  -o log_%x-%j.out  \
  --wrap='module load cray-python/3.11.7; srun --cpu-bind=cores --threads-per-core=1 --label python3 split_m2_v2.py'
