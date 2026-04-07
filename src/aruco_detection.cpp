#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ArucoDistanceNode : public rclcpp::Node
{
public:
  ArucoDistanceNode()
  : Node("aruco_distance_node")
  {
    declare_parameter("camera_index", 0);
    declare_parameter("capture_fps", 30.0);
    declare_parameter("marker_size", 0.05);
    declare_parameter("camera_frame", "camera_optical_frame");
    declare_parameter("calibration_file", "");
    declare_parameter("detector_params_file", "");

    const int camera_index = get_parameter("camera_index").as_int();
    const double fps = get_parameter("capture_fps").as_double();
    marker_size_ = get_parameter("marker_size").as_double();
    camera_frame_ = get_parameter("camera_frame").as_string();
    const auto calib_file = get_parameter("calibration_file").as_string();
    const auto params_file = get_parameter("detector_params_file").as_string();

    // ── Camera intrinsics from YAML (same cv::FileStorage format as lab1) ─────
    cv::FileStorage fs_calib(calib_file, cv::FileStorage::READ);
    if (!fs_calib.isOpened()) {
      RCLCPP_FATAL(get_logger(), "Cannot open calibration file: %s", calib_file.c_str());
      throw std::runtime_error("Calibration file not found");
    }
    fs_calib["cameraMatrix"] >> camera_matrix_;
    fs_calib["distCoeffs"] >> dist_coeffs_;
    fs_calib.release();
    RCLCPP_INFO(get_logger(), "Calibration loaded from: %s", calib_file.c_str());

    // ── ArUco detector parameters from YAML (same format as lab1) ─────────────
    cv::aruco::DetectorParameters detector_params;
    cv::FileStorage fs_params(params_file, cv::FileStorage::READ);
    if (fs_params.isOpened()) {
      detector_params.readDetectorParameters(fs_params.root());
      fs_params.release();
      RCLCPP_INFO(get_logger(), "Detector params loaded from: %s", params_file.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "Detector params file not found, using defaults: %s",
                  params_file.c_str());
    }

    // ── DICT_4X4_50 fixed to match the printed markers (IDs 0–4) ──────────────
    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    detector_ = std::make_unique<cv::aruco::ArucoDetector>(dict, detector_params);

    // ── Open webcam ───────────────────────────────────────────────────────────
    cap_.open(camera_index);
    if (!cap_.isOpened()) {
      RCLCPP_FATAL(get_logger(), "Cannot open camera index %d", camera_index);
      throw std::runtime_error("Camera open failed");
    }

    // ── ROS publishers ────────────────────────────────────────────────────────
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/aruco/poses", 10);
    ids_pub_ = create_publisher<std_msgs::msg::Int32MultiArray>("/aruco/ids", 10);
    distance_pub_ =
      create_publisher<std_msgs::msg::Float64MultiArray>("/aruco/distances", 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/aruco/image_debug", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    cv::namedWindow("ArUco Distance Estimation", cv::WINDOW_AUTOSIZE);

    const auto period = std::chrono::duration<double>(1.0 / fps);
    timer_ = create_wall_timer(period, std::bind(&ArucoDistanceNode::timerCallback, this));

    RCLCPP_INFO(get_logger(),
      "ArUco distance node ready (camera=%d, DICT_4X4_50, marker_size=%.3f m)",
      camera_index, marker_size_);
  }

  ~ArucoDistanceNode()
  {
    cap_.release();
    cv::destroyAllWindows();
  }

private:
  void timerCallback()
  {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN(get_logger(), "Empty frame, skipping.");
      return;
    }

    // ── Detect markers ────────────────────────────────────────────────────────
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    detector_->detectMarkers(frame, corners, ids, rejected);

    if (ids.empty()) {
      cv::imshow("ArUco Distance Estimation", frame);
      cv::waitKey(1);
      return;
    }

    cv::aruco::drawDetectedMarkers(frame, corners, ids);

    const size_t N = ids.size();
    const float  half = static_cast<float>(marker_size_) / 2.f;

    // Marker object points in local frame — same layout as lab1
    cv::Mat obj_pts(4, 1, CV_32FC3);
    obj_pts.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-half,  half, 0.f);
    obj_pts.ptr<cv::Vec3f>(0)[1] = cv::Vec3f( half,  half, 0.f);
    obj_pts.ptr<cv::Vec3f>(0)[2] = cv::Vec3f( half, -half, 0.f);
    obj_pts.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-half, -half, 0.f);

    // ── Pose estimation for every detected marker ─────────────────────────────
    std::vector<cv::Vec3d> rvecs(N), tvecs(N);
    for (size_t i = 0; i < N; ++i) {
      cv::solvePnP(obj_pts, corners[i], camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i]);
      cv::drawFrameAxes(frame, camera_matrix_, dist_coeffs_,
                        rvecs[i], tvecs[i], half * 0.5f, 2);
    }

    // ── Project each marker centre to 2-D image space ─────────────────────────
    cv::Mat centre_3d(1, 1, CV_32FC3);
    centre_3d.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(0.f, 0.f, 0.f);

    std::vector<cv::Point2f> centres_2d(N);
    for (size_t i = 0; i < N; ++i) {
      std::vector<cv::Point2f> proj;
      cv::projectPoints(centre_3d, rvecs[i], tvecs[i], camera_matrix_, dist_coeffs_, proj);
      centres_2d[i] = proj[0];
    }

    // ── Draw all pairwise distances (lab1 style) ───────────────────────────────
    int text_y = 30;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        cv::Vec3d rel  = tvecs[i] - tvecs[j];
        float     dist = static_cast<float>(cv::norm(rel));

        // Green line between the two marker centres
        cv::line(frame, centres_2d[i], centres_2d[j], cv::Scalar(0, 255, 0), 2);

        // Distance label at midpoint of the line
        cv::Point2f mid = (centres_2d[i] + centres_2d[j]) * 0.5f;
        cv::putText(frame, cv::format("%.3f m", dist),
                    cv::Point(static_cast<int>(mid.x) + 5, static_cast<int>(mid.y)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        // Left-side summary block — same text layout as lab1
        cv::putText(frame,
                    "ID" + std::to_string(ids[i]) + " <-> ID" + std::to_string(ids[j]),
                    cv::Point(10, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);
        text_y += 28;
        cv::putText(frame, "  x: " + std::to_string(rel[0]),
                    cv::Point(10, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
        text_y += 25;
        cv::putText(frame, "  y: " + std::to_string(rel[1]),
                    cv::Point(10, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
        text_y += 25;
        cv::putText(frame, "  z: " + std::to_string(rel[2]),
                    cv::Point(10, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
        text_y += 25;
        cv::putText(frame, "  dist: " + std::to_string(dist) + " m",
                    cv::Point(10, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
        text_y += 35;
      }
    }

    cv::imshow("ArUco Distance Estimation", frame);
    cv::waitKey(1);

    // ── Publish ROS topics ────────────────────────────────────────────────────
    const auto stamp = now();

    std_msgs::msg::Int32MultiArray ids_msg;
    for (int id : ids) ids_msg.data.push_back(id);
    ids_pub_->publish(ids_msg);

    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp    = stamp;
    pose_array.header.frame_id = camera_frame_;

    for (size_t i = 0; i < N; ++i) {
      cv::Mat R;
      cv::Rodrigues(rvecs[i], R);
      const auto q = rotMatToQuat(R);

      geometry_msgs::msg::Pose pose;
      pose.position.x    = tvecs[i][0];
      pose.position.y    = tvecs[i][1];
      pose.position.z    = tvecs[i][2];
      pose.orientation.w = q[0];
      pose.orientation.x = q[1];
      pose.orientation.y = q[2];
      pose.orientation.z = q[3];
      pose_array.poses.push_back(pose);

      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp    = stamp;
      tf.header.frame_id = camera_frame_;
      tf.child_frame_id  = "aruco_marker_" + std::to_string(ids[i]);
      tf.transform.translation.x = tvecs[i][0];
      tf.transform.translation.y = tvecs[i][1];
      tf.transform.translation.z = tvecs[i][2];
      tf.transform.rotation.w = q[0];
      tf.transform.rotation.x = q[1];
      tf.transform.rotation.y = q[2];
      tf.transform.rotation.z = q[3];
      tf_broadcaster_->sendTransform(tf);
    }
    pose_pub_->publish(pose_array);

    // N×N symmetric distance matrix; data[i*N + j] = metres between marker i and j
    std_msgs::msg::Float64MultiArray dist_msg;
    dist_msg.layout.dim.resize(2);
    dist_msg.layout.dim[0].label  = "marker_i";
    dist_msg.layout.dim[0].size   = N;
    dist_msg.layout.dim[0].stride = N * N;
    dist_msg.layout.dim[1].label  = "marker_j";
    dist_msg.layout.dim[1].size   = N;
    dist_msg.layout.dim[1].stride = N;
    dist_msg.data.assign(N * N, 0.0);

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        const double d = cv::norm(tvecs[i] - tvecs[j]);
        dist_msg.data[i * N + j] = d;
        dist_msg.data[j * N + i] = d;
      }
    }
    distance_pub_->publish(dist_msg);

    std_msgs::msg::Header hdr;
    hdr.stamp    = stamp;
    hdr.frame_id = camera_frame_;
    debug_pub_->publish(*cv_bridge::CvImage(hdr, "bgr8", frame).toImageMsg());
  }

  // 3×3 rotation matrix -> quaternion [w, x, y, z]
  static std::array<double, 4> rotMatToQuat(const cv::Mat & R)
  {
    const double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
    double w = std::sqrt(std::max(0.0, 1.0 + trace)) / 2.0;
    double x = std::sqrt(std::max(0.0, 1.0 + R.at<double>(0, 0) - R.at<double>(1, 1) - R.at<double>(2, 2))) / 2.0;
    double y = std::sqrt(std::max(0.0, 1.0 - R.at<double>(0, 0) + R.at<double>(1, 1) - R.at<double>(2, 2))) / 2.0;
    double z = std::sqrt(std::max(0.0, 1.0 - R.at<double>(0, 0) - R.at<double>(1, 1) + R.at<double>(2, 2))) / 2.0;
    x = std::copysign(x, R.at<double>(2, 1) - R.at<double>(1, 2));
    y = std::copysign(y, R.at<double>(0, 2) - R.at<double>(2, 0));
    z = std::copysign(z, R.at<double>(1, 0) - R.at<double>(0, 1));
    return {w, x, y, z};
  }

  cv::VideoCapture cap_;
  std::unique_ptr<cv::aruco::ArucoDetector> detector_;
  cv::Mat camera_matrix_, dist_coeffs_;
  double marker_size_;
  std::string camera_frame_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr ids_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr distance_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArucoDistanceNode>());
  rclcpp::shutdown();
  return 0;
}