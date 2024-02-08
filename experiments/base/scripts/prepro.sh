set -eu
source ./scripts/setup.sh

if [ $# = 1 ]; then
    config_file_path=$1
else
    echo "select prepro config file path"
    exit
fi

JSONNET_RESULTS=$(
    jsonnet $config_file_path \
	--ext-str TAG=${TAG} \
	--ext-str ROOT=${ROOT_DIR} \
	--ext-str CURRENT_DIR=${CURRENT_DIR} \
)

echo "Config file:\n${JSONNET_RESULTS}"

PREPRO_ARGS=`python ./tools/config2args.py ${JSONNET_RESULTS}`
echo $PREPRO_ARGS


cd $SOURCE_DIR
PREPRO_PROGRAM=`echo "python ./prepro.py ${PREPRO_ARGS}"`
eval $PREPRO_PROGRAM
cd -

