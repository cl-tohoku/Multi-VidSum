{
  remove_null(input_array):
    if std.length(input_array) == 0 then
      []
    else
      local top = input_array[0];
      if top == null then
	self.remove_null(input_array[1:])
      else
	[top] + self.remove_null(input_array[1:]),

  permutation_sub(input_array, copyed_array):
 
    if std.length(input_array) == 0 then
      []
    else
      local top = input_array[0];
      self.remove_null(
	[
	  if top == elem then null else [top, elem]
	  for elem in copyed_array
	]
      )
      + self.permutation_sub(
	input_array[1:],
	copyed_array,
      ),

  permutation(input_array):
    self.permutation_sub(input_array, input_array),

  # https://github.com/google/jsonnet/issues/312
  objectPop(obj, keys): { 
    [k]: obj[k] for k in std.objectFieldsAll(obj) if !std.member(keys, k)
  },

  split_path(path): std.split(path, "/"),
  
  basename(path):
    local splited_path = self.split_path(path);
    local splited_path_length = std.length(splited_path);
    splited_path[splited_path_length - 1],

  dirname(path): 
    local splited_path = self.split_path(path);
    local splited_path_length = std.length(splited_path);
    std.join("/", splited_path[:splited_path_length - 1]),

  split_extension(path):
    std.split(path, "."),
}
