cmake_minimum_required(VERSION 2.8.3)
project(object_recognition_cnn)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    message_generation
    cv_bridge
    image_transport
    object_detection_common
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(PNG REQUIRED)
find_package(cmake_modules REQUIRED)
find_package(TBB REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Object.msg
)

## Generate services in the 'srv' folder
add_service_files(
   FILES
   LoadWeights.srv
   StoreWeights.srv
   Localize.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES object_recognition_cnn
  CATKIN_DEPENDS message_runtime # roscpp rospy std_msgs
#  DEPENDS OpenCV
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIR})
include_directories(include)
include_directories(include/object_recognition_cnn)


## Declare a cpp library
#add_library(nn_demo
#   src/${PROJECT_NAME}/object_recognition_cnn.cpp
# )

## Declare a cpp executable
add_executable(object_recognition_cnn src/object_recognition_cnn.cpp src/CNN.cpp)
add_executable(localization_tester src/test.cpp src/Tester.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
add_dependencies(object_recognition_cnn ${catkin_EXPORTED_TARGETS} object_recognition_cnn_generate_messages_cpp)
add_dependencies(localization_tester ${catkin_EXPORTED_TARGETS} object_recognition_cnn_generate_messages_cpp)

# -DHEADLESS
set(CMAKE_CXX_FLAGS "-DHEADLESS -DCNN_USE_TBB -DSAVE_KERNELS -DCNN_USE_SSE -std=c++11 -Wall -march=native -O3 ${CMAKE_CXX_FLAGS}") # network options: -DHEADLESS does not create debug windows, -DFULLY_CONNECTED, -DHSV_COLOR_SPACE, -DYUV_COLOR_SPACE, -DYCRCB_COLOR_SPACE, -DGRAYSCALE actually takes YCrCb and only first channel, -DCNN_USE_AVX (or -DCNN_USE_SSE), -DCNN_USE_TBB, -DSAVE_KERNELS to save learned kernels while training. There is no -DBGR_COLOR_SPACE because bgr is the default, -DCHANNELS=<vector initializer list> e.g. -DCHANNELS={0, 2} (default {0, 1, 2})

## Specify libraries to link a library or executable target against
target_link_libraries(object_recognition_cnn ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${PNG_LIBRARY})
target_link_libraries(localization_tester ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PNG_LIBRARY})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS nn_demo nn_demo_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_nn_demo.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
