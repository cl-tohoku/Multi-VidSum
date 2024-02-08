function find-to-root() {
    if [ ! $# = 2 ]; then
	echo "Usage: find-to-root \$dir_path \$target_file_name"
	return 1
    fi

    find_output=`find $1 -maxdepth 1 -name $2`
    
    if [ -n "$find_output" ]; then
	echo `readlink -f $find_output`
	return 0
    fi
    
    abs_path=`readlink -f $1`
    
    if [ $abs_path = "/" ]; then
	return 0
    else
	find-to-root $1/.. $2
	return $?
    fi
}

project_config_path=`find-to-root . .project_config.sh`
echo "Load config from $project_config_path"
source $project_config_path


ln -sf $EXPERIMENT_DIR/base/scripts scripts
ln -sf $EXPERIMENT_DIR/base/tools tools
