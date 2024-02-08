set -eu
source ./tools/shell_utils.sh
load_project_config

if [ -e "/.dockerenv" ] || [ -d "/.singularity.d" ]; then
    echo "This environment is docker!"
    export ROOT_DIR="/work"
else
    echo "Expected docker environment..."
    exit 1
fi

export ROOT=$ROOT_DIR
export STDOUT_DIR=$ROOT_DIR/stdout_logs
mkdir -p $STDOUT_DIR
mkdir -p "${ROOT_DIR}/datasets"

wandb login

#必ず設定する
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d-%H%M%S`
export CURRENT_DIR=`pwd`
export TAG=`basename ${CURRENT_DIR}`


echo "Setup : ${TAG}"

### for debugging ###
date
uname -a
which python
python --version
