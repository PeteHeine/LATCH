cmake_minimum_required(VERSION 2.8)

if(DEFINED ENV{VCPKG_ROOT})
#NOTE has to be defined before project command
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()
project(latch)

if((MSVC) AND (NOT DEFINED ENV{VCPKG_ROOT}))
    message("Windows port depends on VCPKG.")
    message("VCPKG can be installed from https://github.com/Microsoft/vcpkg")
    message("Set  VCPKG_DEFAULT_TRIPLET=x64-windows")
    message(FATAL_ERROR "VCPKG_ROOT has point to root of VCPKG to ensure that all required packages can be found automatically")
endif()

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
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  /W3")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
endif()

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

find_package(GTest REQUIRED)

# Directories of .h-files
include_directories(
${PROJECT_SOURCE_DIR}
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)

add_library( feature_extractor
feature_extractor.h
feature_extractor.cpp
ORBextractor.h
ORBextractor.cpp)

add_executable( main
main.cpp)

target_link_libraries( main
${OpenCV_LIBS}
Threads::Threads
)



add_executable( ${PROJECT_NAME}
main_latch.cpp
feature_extractor.h
feature_extractor.cpp
ORBextractor.h
ORBextractor.cpp)

target_link_libraries( ${PROJECT_NAME}
${OpenCV_LIBS}
Threads::Threads)


enable_testing()
add_executable(test_feature_extractor test_feature_extractor.cpp feature_extractor.h
feature_extractor.cpp ORBextractor.h ORBextractor.cpp)

target_link_libraries(test_feature_extractor PRIVATE GTest::GTest GTest::Main)
target_link_libraries(test_feature_extractor ${OpenCV_LIBS})
#add_test(NAME that-test-I-made COMMAND runUnitTests)
