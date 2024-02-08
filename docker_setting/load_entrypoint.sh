if [ -d "/.singularity.d" ] && [ ! -e "/.dockerenv" ]; then
    . /.singularity.d/env/10-docker2singularity.sh
    export PATH=$PATH:$HOME/.local/bin
fi

source $ENTRY_POINT_SCRIPT_PATH
