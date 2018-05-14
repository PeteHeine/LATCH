cmake_minimum_required(VERSION 2.8)
project(latch)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()


#MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
if ( CMAKE_COMPILER_IS_GNUCC )
# -Wall -Wextra -Werror -Wshadow -pedantic -Ofast -std=gnu++17 -fomit-frame-pointer -mavx2 -march=native -mfma -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic -Ofast -std=gnu++17 -fomit-frame-pointer -march=native -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize")
	set(THREADS_PREFER_PTHREAD_FLAG ON)



	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -std=c++11")	
	if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
		message("CPU: x86_64")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -march=native ")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -march=native  -mavx2  -mfma")
	elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
		message("CPU: aarch64")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a ")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
	else()
		message(FATAL_ERROR "Specify flags for processor type: " ${CMAKE_SYSTEM_PROCESSOR} )
	endif()
endif()
if ( MSVC )
	# FLAGS
		# /W0 means no errors are published
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  /W0")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
endif()

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()


# EIGEN
if ( MSVC ) # FOR WINDOWS
	SET( EIGEN3_INCLUDE_DIR "${Eigen3_IN}" )
	if(NOT EIGEN3_INCLUDE_DIR)
		MESSAGE( FATAL_ERROR "Please point the environment variable Eigen3_IN to the include directory of your Eigen3 installation.")
	else(NOT EIGEN3_INCLUDE_DIR)
		MESSAGE("Found EIGEN3! ")
	endif()
else( MSVC ) # For ubuntu
	find_package(Eigen3)
endif()

# PYTHON
if ( MSVC ) # FOR WINDOWS
	SET(PYTHON_PATH ${Python_IN})
	if(NOT PYTHON_PATH)
		MESSAGE( FATAL_ERROR "Please point the environment variable Python_IN to the include directory of your Python installation.")
	else(NOT PYTHON_PATH)
		MESSAGE("Found Python! ")
		SET(PYTHON_INCLUDE_DIRS ${PYTHON_PATH}/include)
		SET(PYTHON_LIBRARIES ${PYTHON_PATH}/libs/python36.lib)
	endif()
else( MSVC ) # For ubuntu
	find_package(PythonLibs) #
	MESSAGE("Python version: " ${PYTHON_LIBRARY} "  " ${PYTHON_INCLUDE_DIR})
	#SET(PYTHON_PATH ${Python_IN})
	#if(NOT PYTHON_PATH)
	#	MESSAGE( FATAL_ERROR "Please point the environment variable Python_IN to the include directory of your Python installation.")
	#else(NOT PYTHON_PATH)
	#	MESSAGE("Found Python! ")
	#	SET(PYTHON_INCLUDE_DIR ${PYTHON_PATH}/include)
	#	SET(PYTHON_LIBRARY ${PYTHON_PATH}/libs/python36.lib)
	#endif()


endif()


# g2o could potentially be added as a subdirectory. It is however super annoying to wait for it to build, when the cmake-file have been changed!!!
# add_subdirectory(External/g2o)


#FIND_PACKAGE(Eigen3)
#IF(EIGEN3_FOUND)
#  SET(G2O_EIGEN3_INCLUDE ${EIGEN3_INCLUDE_DIR} CACHE PATH "Directory of Eigen3")
#ELSE(EIGEN3_FOUND)
#  SET(G2O_EIGEN3_INCLUDE "" CACHE PATH "Directory of Eigen3")
#ENDIF(EIGEN3_FOUND)

#find_package(pybind11 REQUIRED)

# Directories of .h-files
include_directories(
${PROJECT_SOURCE_DIR}
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)

# Build examples

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)




add_library( feature_extractor
feature_extractor.cpp
ORBextractor.cpp)

add_executable( main
main.cpp)

target_link_libraries( main
${OpenCV_LIBS}
Threads::Threads
)

add_executable( ${PROJECT_NAME}
feature_extractor.cpp
ORBextractor.cpp)

target_link_libraries( ${PROJECT_NAME}
${OpenCV_LIBS}
Threads::Threads)


