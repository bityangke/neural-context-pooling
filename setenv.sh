# Script to setup enviroment variables
# Usage: . setenv.sh OR source setenv.sh
# Get project dir
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Add source to PYTHONPATH
export PYTHONPATH=$DIR:PYTHONPATH

# Load requirements with module-environment
module purge
module load compilers/gcc/4.9.4
module load compilers/cuda/7.5
module load libs/cudnn/v5
module load tools/conda

# Safe activation of conda enviroment (overcome racing conditions ;))
while true; do
  source activate neural-context-pooling
  if [ $? -eq 0 ]; then
    break;
  else
    sleep $[ ($RANDOM % 10 ) + 1]s
  fi
done
