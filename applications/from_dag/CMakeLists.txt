set(CMAKE_CXX_STANDARD 20)

get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(from_dag ${app_dir}/main.cc
                        ${app_dir}/dag.cc
                        ${app_dir}/parse.cc
                        ${app_dir}/generate.cc
                        ${app_dir}/partition/partition_options.cc
                        ${app_dir}/partition/partition.cc
                        ${app_dir}/partition/run.cc)

# link the stuff we need
target_link_libraries(from_dag bbts-common)
target_link_libraries(from_dag
  -L/usr/local/lib
  -lgecodeflatzinc
  -lgecodedriver
  -lgecodeminimodel
  -lgecodeset
  -lgecodefloat
  -lgecodeint
  -lgecodekernel
  -lgecodesearch
  -lgecodesupport)

