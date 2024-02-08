set -eu
source ./scripts/setup.sh

if [ $# -ge 2 ]; then
    config_file_path=$1
    GPU_ID_ARRAY="${@:2}"
    GPU_IDS=`python -c "print(\",\".join(\"${GPU_ID_ARRAY}\".split()))"`
else
    echo "Select config file path and gpu id"
    exit
fi

JSONNET_RESULTS=$(
    jsonnet $config_file_path \
	--ext-str TAG=${TAG} \
	--ext-str ROOT=${ROOT_DIR} \
	--ext-str CURRENT_DIR=${CURRENT_DIR} \
)


echo "Config file:\n${JSONNET_RESULTS}"

TEST_ARGS=`python ./tools/config2args.py ${JSONNET_RESULTS}`
echo $TEST_ARGS

cd $SOURCE_DIR
TEST_PROGRAM=`echo "python ./test.py ${TEST_ARGS} 2>&1 | tee ${STDOUT_DIR}/test_${TAG}_${DATE}.log"`
eval "CUDA_VISIBLE_DEVICES=$GPU_IDS $TEST_PROGRAM"
cd -
