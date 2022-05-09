set(CMAKE_CXX_STANDARD 20)

get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(from_dag ${app_dir}/main.cc
                        ${app_dir}/dag.cc
                        ${app_dir}/parse.cc
                        ${app_dir}/relation.cc
                        ${app_dir}/generate.cc
                        ${app_dir}/print_table.cc
                        ${app_dir}/partition/partition.cc
                        ${app_dir}/partition/run.cc)
#set_target_properties(from_dag PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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
  -lgecodesupport
  -lpthread)
target_link_libraries(from_dag ${MPI_LIBRARIES})

add_library(from_dag_ops SHARED ${app_dir}/ops/ops.cc)
target_link_libraries(from_dag_ops bbts-common)

target_compile_definitions(from_dag
  PRIVATE FROM_DAG_KERNEL_LIB=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfrom_dag_ops.so)


