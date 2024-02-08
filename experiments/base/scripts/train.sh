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

TRAIN_ARGS=`python ./tools/config2args.py ${JSONNET_RESULTS}`
echo $TRAIN_ARGS


cd $SOURCE_DIR
TRAIN_PROGRAM=`echo "python ./train.py --mode train ${TRAIN_ARGS} 2>&1 | tee ${STDOUT_DIR}/train_${TAG}_${DATE}.log"`
eval "CUDA_VISIBLE_DEVICES=$GPU_IDS $TRAIN_PROGRAM"
cd -

