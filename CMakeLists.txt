cmake_minimum_required(VERSION 3.1)
project( rosrect )
add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
  sensor_msgs
  image_transport
  cv_bridge
)

find_package( OpenCV REQUIRED )

find_library( LIBOPENCL OpenCL )

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs
)


include_directories( ${OPENCL_INCLUDE_DIRS} )

include_directories(
  ${catkin_INCLUDE_DIRS}
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(TARGET_CLTOHEADER "cltoheader")
add_executable(${TARGET_CLTOHEADER} cltoheader.c)

#

add_custom_command(OUTPUT oclimgutil_cl.h
  COMMENT "Generating oclimgutil_cl.h"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/oclimgutil.cl source > ${CMAKE_CURRENT_BINARY_DIR}/oclimgutil_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclimgutil_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclimgutil_cl.h)

add_custom_command(OUTPUT oclpolyline_cl.h
  COMMENT "Generating oclpolyline_cl.h"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/oclpolyline.cl source > ${CMAKE_CURRENT_BINARY_DIR}/oclpolyline_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclpolyline_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclpolyline_cl.h)

add_custom_command(OUTPUT oclrect_cl.h
  COMMENT "Generating oclrect_cl.h"
  COMMAND $<TARGET_FILE:${TARGET_CLTOHEADER}> ${CMAKE_CURRENT_SOURCE_DIR}/oclrect.cl source > ${CMAKE_CURRENT_BINARY_DIR}/oclrect_cl.h
  DEPENDS ${TARGET_CLTOHEADER}
  )
add_custom_target(oclrect_cl.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/oclrect_cl.h)

#

add_library(helper_obj OBJECT oclhelper.c helper.c)

add_library(oclimgutil_obj OBJECT oclimgutil.c oclimgutil.h ${CMAKE_CURRENT_BINARY_DIR}/oclimgutil_cl.h)
add_dependencies(oclimgutil_obj oclimgutil_cl.h_generated)

add_library(oclpolyline_obj OBJECT oclpolyline.c oclpolyline.h ${CMAKE_CURRENT_BINARY_DIR}/oclpolyline_cl.h)
add_dependencies(oclpolyline_obj oclpolyline_cl.h_generated)

add_library(oclrect_obj OBJECT oclrect.c oclrect.h ${CMAKE_CURRENT_BINARY_DIR}/oclrect_cl.h)
add_dependencies(oclrect_obj oclrect_cl.h_generated)

add_executable( vidrect vidrect.cpp $<TARGET_OBJECTS:oclrect_obj> $<TARGET_OBJECTS:oclpolyline_obj> $<TARGET_OBJECTS:oclimgutil_obj> $<TARGET_OBJECTS:helper_obj>)
target_link_libraries( vidrect ${catkin_LIBRARIES} ${OpenCV_LIBS} ${LIBOPENCL} )


