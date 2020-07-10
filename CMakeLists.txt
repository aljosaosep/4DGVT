cmake_minimum_required(VERSION 3.9.6)
project(GOT)

# Build Options
option(SHOW_PATH "Show path of libs and headers" ON)
set(OpenCV_DIR "/home/aljosa/local/share/OpenCV" CACHE PATH "Path to local OpenCV e.g. USER/local/share/OpenCV")
set(BOOST_ROOT "/home/aljosa/local/lib" CACHE PATH "Boost root dir")

# Configuration
# Compiler Flags
set(CMAKE_CXX_COMPILER_FLAGS "${CMAKE_CXX_COMPILER_FLAGS} -Wall") # Enables all compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")  # Generate position independent code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -DBOOST_MATH_DISABLE_FLOAT128")  # Enable c++17
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")  # Forcing SSE vectorization instructions may cause segmentfault
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

# Here we set *_LIBS, *_INCLUDE_DIRS, etc
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc calib3d)
find_package(PCL 1.9.0 REQUIRED COMPONENTS common io visualization filters sample_consensus segmentation kdtree)
link_directories(${PCL_LIBRARY_DIRS})

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.55 COMPONENTS program_options thread system REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIR})
    message(STATUS "Boost found!" ${Boost_LIBRARIES})
endif()

include_directories(include ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

# Include own files
file(GLOB PROJ_SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/src)

file(GLOB EXTERNAL_SRC_FILES ${CMAKE_SOURCE_DIR}/src/external/*.cpp)
set(EXTERNAL_SRC_DIR ${CMAKE_SOURCE_DIR}/src/external)
include_directories(${EXTERNAL_SRC_DIR})

file(GLOB SUN_UTILS_SRC_FILES ${CMAKE_SOURCE_DIR}/src/sun_utils/*.cpp)
set(SUN_UTILS_SRC_DIR ${CMAKE_SOURCE_DIR}/src/sun_utils)
include_directories(${SUN_UTILS_SRC_DIR})

file(GLOB CAMOT_SRC_FILES ${CMAKE_SOURCE_DIR}/src/CAMOT/*.cpp)
set(CAMOT_SRC_DIR ${CMAKE_SOURCE_DIR}/src/CAMOT)
include_directories(${CAMOT})

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Use OpenMP")
endif()

# Libraries
add_library (connected_components
        src/connected_components/CC.cpp
        src/connected_components/CPoint.cpp

        include/connected_components/CC.h
        include/connected_components/CPoint.h
        )

add_library (viso2
        src/libviso2/matrix.cpp
        src/libviso2/viso.cpp
        src/libviso2/viso_mono.cpp
        src/libviso2/viso_stereo.cpp
        src/libviso2/matcher.cpp
        src/libviso2/reconstruction.cpp
        src/libviso2/triangle.cpp
        src/libviso2/filter.cpp

        include/libviso2/viso.h
        include/libviso2/viso_mono.h
        include/libviso2/viso_stereo.h
        include/libviso2/matcher.h
        include/libviso2/reconstruction.h
        include/libviso2/triangle.h
        include/libviso2/filter.h
        )

add_library (elas
        src/libelas/matrix.cpp
        src/libelas/triangle.cpp
        src/libelas/filter.cpp
        src/libelas/elas.cpp
        src/libelas/descriptor.cpp

        include/libelas/elas.h
        include/libelas/image.h
        include/libelas/timer.h
        include/libelas/matrix.h
        include/libelas/triangle.h
        include/libelas/filter.h
        include/libelas/descriptor.h
        )

add_library (tracking
        src/tracking/data_queue.cpp
        src/tracking/observation.cpp
        src/tracking/multi_object_tracker_base.cpp
        src/tracking/hypothesis.cpp
        src/tracking/kalman_filter.cpp
        src/tracking/extended_kalman_filter.cpp
        src/tracking/visualization.cpp
        src/tracking/qpbo.cpp

        include/tracking/data_queue.h
        include/tracking/observation.h
        include/tracking/multi_object_tracker_base.h
        include/tracking/hypothesis.h
        include/tracking/kalman_filter.h
        include/tracking/extended_kalman_filter.h
        include/tracking/visualization.h
        include/tracking/qpbo.h
        include/tracking/dynamics_handler.h
        )

add_library (scene_segmentation
        src/scene_segmentation/ground_histogram.cpp
        src/scene_segmentation/object_proposal.cpp
        src/scene_segmentation/scene_segmentation.cpp
        src/scene_segmentation/utils_segmentation.cpp
        src/scene_segmentation/json_prop_tools.cpp
        src/scene_segmentation/scene_filtering.cpp

        include/scene_segmentation/ground_histogram.h
        include/scene_segmentation/object_proposal.h
        include/scene_segmentation/scene_segmentation.h
        include/scene_segmentation/utils_segmentation.h
        include/scene_segmentation/segmentation_visualization.h
        include/scene_segmentation/parameters_gop3D.h
        include/scene_segmentation/json_prop_tools.h
        include/scene_segmentation/scene_filtering.h)

target_link_libraries(connected_components)
target_link_libraries(elas)
target_link_libraries(viso2)
target_link_libraries(scene_segmentation pthread)
target_link_libraries(tracking viso2)

# Subdirectories
# Add subfolder that also contain CMakeLists.txt
add_subdirectory(apps)

# Debug infos
if(SHOW_PATH)
    message(STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES})
    message(STATUS ${OpenCV_LIBS})
    message(STATUS "Project Source Dirs: " ${PROJECT_SOURCE_DIR})
    message(STATUS "OpenCV Include Dirs: " ${OpenCV_INCLUDE_DIRS})
    message(STATUS "PCL Include Dirs: " ${PCL_INCLUDE_DIRS})
    message(STATUS "")
endif()