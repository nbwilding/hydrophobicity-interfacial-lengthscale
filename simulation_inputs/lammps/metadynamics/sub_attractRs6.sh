#!/bin/bash
#SBATCH --job-name=pmfRs6T300P0D0.01
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# --- Cray PE environment (run-time) ---
module reset
module load craype-network-ofi
module load PrgEnv-gnu

export LD_LIBRARY_PATH=$HOME/opt/plumed/2.9.0/lib:$LD_LIBRARY_PATH

# --- LAMMPS executable (built with PLUMED + MANYBODY) ---
LMP=$HOME/opt/lammps-plumed-sw/bin/lmp
echo "Using LAMMPS: $LMP"
$LMP -h | head -n 30

pwd

Rs=6.0
Bias=10
Temp=300.0
Press=0.0
D0water=0.01

# For 3x3 system
UW=30.0
rep=3

Ds=$(awk -v r="$Rs" 'BEGIN{printf "%.1f", 2*r}')
echo "Ds=$Ds"

folder=BIAS${Bias}-T${Temp}-P${Press}-Rs${Rs}-D0${D0water}
mkdir -p $folder

Rs_tag=$(awk -v r="$Rs" 'BEGIN{printf "%.1f", r}')
D0_tag=$(awk -v d="$D0water" 'BEGIN{printf "%.2f", d}')

echo "Rs_tag==$Rs_tag"
echo "D0_tag==$D0_tag"
cp "LR_pot_tables/Vext_Rs${Rs_tag}_eps${D0_tag}.table" "Vext${D0water}.table"

cp mW.sw onlyWater.start in.equil.mw.attract.lmp in.pmf.mw.attract.lmp "Vext${D0water}.table" "$folder"
rm -f "Vext${D0water}.table"

cd "$folder"

# PLUMED input (3x3x3 replication)
cat > plumed.dat <<EOF
RESTART
UNITS LENGTH=A ENERGY=kcal/mol
d1: DISTANCE ATOMS=7131,7132
uwall: UPPER_WALLS ARG=d1 AT=${UW} KAPPA=150 EXP=2 EPS=1 OFFSET=0
restraint: METAD ARG=d1 SIGMA=0.04 HEIGHT=1.0 PACE=100 BIASFACTOR=${Bias} TEMP=${Temp}
PRINT ARG=d1,restraint.bias STRIDE=100 FILE=COLVAR
FLUSH STRIDE=100
EOF

# Use Slurm's task count (don’t hardcode 64)
echo "SLURM_NTASKS=$SLURM_NTASKS  SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"

# --- Run equilibration ---
srun  $LMP -in in.equil.mw.attract.lmp  -v T "$Temp" -v P "$Press" -v solute_diam "$Ds" -v rep "$rep" -v D0water "$D0water"

# --- Run PMF / metadynamics ---
srun $LMP -in in.pmf.mw.attract.lmp  -v T "$Temp" -v P "$Press" -v solute_diam "$Ds" -v D0water "$D0water"
