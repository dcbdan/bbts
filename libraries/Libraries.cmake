# get the current directory
get_filename_component(path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# compile each .cc file into a shared library
file(GLOB files "${path}/*.cc")

#add_custom_target(libraries)
foreach(file ${files})
    # grab the name of the test without the extension
    get_filename_component(fileName "${file}" NAME_WE)
    if(EXISTS "${path}/${fileName}/") 
      file(GLOB fileNameCC "${path}/${fileName}/*.cc")
      file(GLOB fileNameCU "${path}/${fileName}/*.cc")
      add_library(${fileName} SHARED ${file} ${fileNameCC} ${fileNameCU})
    else()
      add_library(${fileName} SHARED ${file})
    endif()
  
    #target_link_libraries(${fileName} bbts-common)
  
    #add_dependencies(libraries ${fileName})
endforeach()
