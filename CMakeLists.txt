cmake_minimum_required(VERSION 2.6)
project(cvcsh)
find_package(OpenCV REQUIRED )
find_package(Boost 1.55.0 COMPONENTS  filesystem system REQUIRED)

set(ROOT ~)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CVCSH_SOURCES	CSH_nn_example_usage.cpp CSH_nn.cpp )
	
include_directories(.)
include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(cvcsh    ${CVCSH_SOURCES})	
target_link_libraries(cvcsh  ${OpenCV_LIBS} ${Boost_LIBRARIES} )

#install(TARGETS cvcsh RUNTIME DESTINATION bin)

