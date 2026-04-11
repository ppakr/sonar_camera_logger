cmake_minimum_required(VERSION 3.8)
project(sonar_camera_logger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── ROS 2 dependencies ────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc objdetect calib3d)

# ── Third-party dependencies ──────────────────────────────────────────────────
find_package(Protobuf REQUIRED)
find_package(CURL REQUIRED)
find_package(ZLIB REQUIRED)

# Snappy has no official CMake module; find it manually
find_library(SNAPPY_LIBRARIES NAMES snappy REQUIRED)
find_path(SNAPPY_INCLUDE_DIRS NAMES snappy.h REQUIRED)

# ── Generate C++ from .proto ──────────────────────────────────────────────────
find_program(PROTOC_EXECUTABLE protoc REQUIRED)

set(PROTO_FILE   "${CMAKE_CURRENT_SOURCE_DIR}/proto/WaterLinkedSonarIntegrationProtocol.proto")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

set(PROTO_PB_CC "${PROTO_GEN_DIR}/WaterLinkedSonarIntegrationProtocol.pb.cc")
set(PROTO_PB_H  "${PROTO_GEN_DIR}/WaterLinkedSonarIntegrationProtocol.pb.h")

# Protobuf well-known types (.proto files) live alongside the C++ headers
add_custom_command(
  OUTPUT "${PROTO_PB_CC}" "${PROTO_PB_H}"
  COMMAND "${PROTOC_EXECUTABLE}"
          "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto"
          "--proto_path=${Protobuf_INCLUDE_DIRS}"
          "--cpp_out=${PROTO_GEN_DIR}"
          "${PROTO_FILE}"
  DEPENDS "${PROTO_FILE}"
  COMMENT "Running protoc on WaterLinkedSonarIntegrationProtocol.proto"
  VERBATIM
)

add_custom_target(wl_proto_gen DEPENDS "${PROTO_PB_CC}" "${PROTO_PB_H}")

# ── sonar_driver executable ───────────────────────────────────────────────────
add_executable(sonar_driver
  src/sonar_driver.cpp
  "${PROTO_PB_CC}"
)
add_dependencies(sonar_driver wl_proto_gen)

target_include_directories(sonar_driver PRIVATE
  "${PROTO_GEN_DIR}"               # generated .pb.h lives here
  ${Protobuf_INCLUDE_DIRS}
  ${SNAPPY_INCLUDE_DIRS}
)

target_link_libraries(sonar_driver
  protobuf::libprotobuf
  ${SNAPPY_LIBRARIES}
  CURL::libcurl
  ZLIB::ZLIB
)

ament_target_dependencies(sonar_driver
  rclcpp
  sensor_msgs
  std_msgs
)

install(TARGETS sonar_driver
  DESTINATION lib/${PROJECT_NAME}
)

# ── aruco_distance_node ───────────────────────────────────────────────────────
add_executable(aruco_distance_node src/aruco_detection.cpp)
target_include_directories(aruco_distance_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(aruco_distance_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  cv_bridge
)
target_link_libraries(aruco_distance_node ${OpenCV_LIBS})

install(TARGETS aruco_distance_node
  DESTINATION lib/${PROJECT_NAME}
)

# ── camera_calibration_node ───────────────────────────────────────────────────
add_executable(camera_calibration_node src/camera_calibration.cpp)
target_include_directories(camera_calibration_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(camera_calibration_node
  rclcpp
)
target_link_libraries(camera_calibration_node ${OpenCV_LIBS})

install(TARGETS camera_calibration_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# ── Python scripts ───────────────────────────────────────────────────────────
install(PROGRAMS
  scripts/sonar_driver.py
  scripts/sonar_camera_fusion.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)

  # Pure geometry unit tests — no ROS2 or hardware needed
  ament_add_pytest_test(test_geometry test/test_geometry.py)

  # Node unit tests — hardware mocked, ROS2 required
  # ament_add_pytest_test(test_sonar_driver test/test_sonar_driver.py)

  # Launch / integration tests — requires loopback alias for 192.168.1.96
  # (auto-skipped when alias is absent, see test file for setup instructions)
  # find_package(launch_testing_ament_cmake REQUIRED)
  # add_launch_test(test/test_sonar_driver_launch.py)
endif()

ament_package()
