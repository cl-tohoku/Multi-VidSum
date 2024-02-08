THIS_SCRIPT_PATH=`dirname $0`

export BASE_DIR=`readlink -f $THIS_SCRIPT_PATH`
export PROJECT_NAME=`basename $BASE_DIR`


export WORK_DIR=""

# if WORK_DIR is not set, then throw error
if [ -z "$WORK_DIR" ]; then
    echo "WORK_DIR is not set"
    echo "Please set WORK_DIR in .project_config.sh"
    exit 1
fi

export CODE_DIR="${BASE_DIR}/src"
export SOURCE_DIR="${CODE_DIR}"

export EXPERIMENT_DIR="${BASE_DIR}/experiments"
export LIB_DIR="${BASE_DIR}/lib"
export DOCKER_SETTING_DIR="${BASE_DIR}/docker_setting"
export DEFAULT_CONFIG_DIR="${BASE_DIR}/default_configs"
