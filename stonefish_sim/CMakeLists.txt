cmake_minimum_required(VERSION 3.10)
project(orca_xluuv_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Requires Stonefish to already be built & installed on this machine
# (https://github.com/patrykcieslak/stonefish). This project does not
# vendor or build Stonefish itself.
find_package(Stonefish REQUIRED)

add_executable(orca_sim src/main.cpp)
target_link_libraries(orca_sim Stonefish::Stonefish)

# Scenario 1: seabed survey with obstacle avoidance (rocks/fish).
add_executable(orca_survey_sim src/main_survey.cpp src/OrcaController.cpp)
target_link_libraries(orca_survey_sim Stonefish::Stonefish)

# Scenario 2: survey while avoiding other surveying submarines.
add_executable(orca_multisub_sim src/main_multisub.cpp src/OrcaController.cpp)
target_link_libraries(orca_multisub_sim Stonefish::Stonefish)

set(ORCA_TARGETS orca_sim orca_survey_sim orca_multisub_sim)

# Mass / buoyancy / CG / CB probe used to trim the vehicle models.
add_executable(orca_mass_probe tools/mass_probe.cpp)
target_link_libraries(orca_mass_probe Stonefish::Stonefish)

# Unit test for the RL command -> actuator force conversion (no ROS, no Stonefish).
add_executable(orca_rl_adapter_test test/test_command_adapter.cpp)
target_include_directories(orca_rl_adapter_test PRIVATE src)

# ---------------------------------------------------------------------------
# RL integration: ROS 2-aware Stonefish executable. Built only when a ROS 2
# environment is sourced (source /opt/ros/<distro>/setup.bash), so the three
# demo executables above still build on a machine without ROS 2.
# ---------------------------------------------------------------------------
find_package(rclcpp QUIET)
if(rclcpp_FOUND)
    find_package(std_msgs REQUIRED)
    find_package(std_srvs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(rosgraph_msgs REQUIRED)
    message(STATUS "ROS 2 found ($ENV{ROS_DISTRO}) - building orca_rl_sim")

    add_executable(orca_rl_sim
        src/main_rl.cpp
        src/OrcaStonefishRos.cpp
        src/OrcaController.cpp)
    target_link_libraries(orca_rl_sim
        Stonefish::Stonefish
        rclcpp::rclcpp
        ${std_msgs_TARGETS}
        ${std_srvs_TARGETS}
        ${sensor_msgs_TARGETS}
        ${nav_msgs_TARGETS}
        ${geometry_msgs_TARGETS}
        ${rosgraph_msgs_TARGETS})
    list(APPEND ORCA_TARGETS orca_rl_sim)
else()
    message(WARNING "rclcpp not found: orca_rl_sim (ROS 2 RL integration) will NOT be built. "
                    "Source your ROS 2 setup.bash and re-run cmake.")
endif()

# Keep the resources next to the built binaries so the relative
# "./resources/" path in main*.cpp resolves correctly when you run it
# from the build directory. A target (not POST_BUILD) so edited .scn files
# are copied on every `make`, even when no executable had to be relinked.
add_custom_target(copy_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        ${CMAKE_BINARY_DIR}/resources
    COMMENT "Copying scenario resources")
