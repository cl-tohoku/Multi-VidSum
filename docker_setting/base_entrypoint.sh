set -eu

cwd=`dirname "${0}"`
BASE_DIR=`(cd "${cwd}/../" && pwd)`

LIB_DIR="${BASE_DIR}/lib"
DOCKER_SETTING_DIR="${BASE_DIR}/docker_setting"


cd ${DOCKER_SETTING_DIR}
wandb login `cat .wandb_api_key.txt`
cd

cd ${LIB_DIR}/transformers
echo 'Installing transformers...'
pip install --editable .
cd

cd $ENTER_DIR

