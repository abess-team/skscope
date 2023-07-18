cmake_minimum_required(VERSION 3.14)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(_scope LANGUAGES CXX)

# environment variables
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


if(MSVC)
    SET(CMAKE_CXX_FLAGS "/EHsc /arch:AVX2")
    SET(CMAKE_CXX_FLAGS_DEBUG "/W4")
elseif(DARWIN)
    SET(CMAKE_CXX_FLAGS_DEBUG "-Wall")
else()
    SET(CMAKE_CXX_FLAGS "-march=native")
    SET(CMAKE_CXX_FLAGS_DEBUG "-Wall")
    # This flags are used if cmake is called with -DCMAKE_BUILD_TYPE=PROFILE
    SET(CMAKE_CXX_FLAGS_PROFILE --coverage)
endif()

# dependencies
add_subdirectory(pybind11)
find_package(OpenMP)


# sources
aux_source_directory(src SOURCES)
pybind11_add_module(_scope MODULE ${SOURCES})

# links
target_include_directories(_scope PRIVATE include
                                                 src)
target_link_libraries(_scope PRIVATE pybind11::module 
                                            pybind11::opt_size 
                                            pybind11::lto
                                            pybind11::windows_extras)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_scope PRIVATE OpenMP::OpenMP_CXX)
endif()

# build extension
pybind11_extension(_scope)
if(NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(_scope)
endif()

set_target_properties(_scope PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                               CUDA_VISIBILITY_PRESET "hidden")
