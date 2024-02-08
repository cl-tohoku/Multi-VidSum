set -eux
source ./tools/shell_utils.sh
load_project_config

ENTRY_POINT_SCRIPT_PATH="${DOCKER_SETTING_DIR}/entrypoint.sh"
    
docker build \
       -t $PROJECT_NAME \
       --build-arg entry_point_script_path="${ENTRY_POINT_SCRIPT_PATH}" \
       --build-arg home_directory="$(readlink -f $HOME)" \
       .
