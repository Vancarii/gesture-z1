cmake_minimum_required(VERSION 3.0.2)
project(z1_sdk)

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)

# Force use of system Python (not Anaconda) for pybind11
find_program(PYTHON3_EXECUTABLE
    NAMES python3
    PATHS /usr/bin /usr/local/bin
    NO_DEFAULT_PATH
)
if(NOT PYTHON3_EXECUTABLE)
    find_program(PYTHON3_EXECUTABLE python3)
endif()
if(PYTHON3_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON3_EXECUTABLE})
    set(PYTHON_EXECUTABLE ${PYTHON3_EXECUTABLE})
endif()

find_package(pybind11 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES unitree_arm_sdk
  CATKIN_DEPENDS rospy
  DEPENDS EIGEN3
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# Link against the pre-built library
add_library(unitree_arm_sdk SHARED IMPORTED)
set_target_properties(unitree_arm_sdk PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libunitree_arm_sdk.so
)

# Build the Python interface module
pybind11_add_module(unitree_arm_interface 
  src/arm_python_interface.cpp
  src/UdpPort.cpp
)

target_link_libraries(unitree_arm_interface 
  PRIVATE 
    unitree_arm_sdk
    ${EIGEN3_LIBRARIES}
)

# Install the Python module to the lib directory
install(TARGETS unitree_arm_interface
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

# Install the pre-built library
install(FILES lib/libunitree_arm_sdk.so
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

# Install header files
install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# Install Python scripts
install(PROGRAMS scripts/example_joint_ctrl.py scripts/example_lowcmd.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

