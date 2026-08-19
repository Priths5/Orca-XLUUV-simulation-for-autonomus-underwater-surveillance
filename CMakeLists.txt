cmake_minimum_required(VERSION 3.10)
project(orca_xluuv_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Requires Stonefish to already be built & installed on this machine
# (https://github.com/patrykcieslak/stonefish). This project does not
# vendor or build Stonefish itself.
find_package(Stonefish REQUIRED)

add_executable(orca_sim src/main.cpp)
target_link_libraries(orca_sim Stonefish::Stonefish)

# Keep the resources next to the built binary so the relative
# "./resources/" path in main.cpp resolves correctly when you run it
# from the build directory.
add_custom_command(TARGET orca_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        $<TARGET_FILE_DIR:orca_sim>/resources
)
