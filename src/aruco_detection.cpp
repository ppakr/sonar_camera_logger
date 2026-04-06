#include <cmath>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

class ArucoDistanceNode : public rclcpp::Node
{
public:
  ArucoDistanceNode() : Node("aruco_distance_node")
  {
    declare_parameter("camera_index",   0);
    declare_parameter("capture_fps",    30.0);
    declare_parameter("marker_size",    0.05);
    declare_parameter("dictionary_id",  10);
    declare_parameter("camera_frame",   "camera_optical_frame");
    declare_parameter("camera_fx",      600.0);
    declare_parameter("camera_fy",      600.0);
    declare_parameter("camera_cx",      320.0);
    declare_parameter("camera_cy",      240.0);
    declare_parameter("dist_coeffs",    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});

    marker_size_  = get_parameter("marker_size").as_double();
    camera_frame_ = get_parameter("camera_frame").as_string();
    const int dict_id      = get_parameter("dictionary_id").as_int();
    const int camera_index = get_parameter("camera_index").as_int();
    const double fps       = get_parameter("capture_fps").as_double();

    // Build camera matrix from individual focal length / principal point parameters
    const double fx = get_parameter("camera_fx").as_double();
    const double fy = get_parameter("camera_fy").as_double();
    const double cx = get_parameter("camera_cx").as_double();
    const double cy = get_parameter("camera_cy").as_double();
    camera_matrix_ = (cv::Mat_<double>(3, 3) <<
      fx, 0,  cx,
      0,  fy, cy,
      0,  0,  1);

    const auto dc = get_parameter("dist_coeffs").as_double_array();
    dist_coeffs_ = cv::Mat(dc.size(), 1, CV_64F);
    for (size_t i = 0; i < dc.size(); ++i) dist_coeffs_.at<double>(i) = dc[i];

    auto dict   = cv::aruco::getPredefinedDictionary(dict_id);
    auto params = cv::aruco::DetectorParameters();
    detector_   = std::make_unique<cv::aruco::ArucoDetector>(dict, params);

    cap_.open(camera_index);
    if (!cap_.isOpened()) {
      RCLCPP_FATAL(get_logger(), "Cannot open camera index %d", camera_index);
      throw std::runtime_error("Camera open failed");
    }

    pose_pub_     = create_publisher<geometry_msgs::msg::PoseArray>("/aruco/poses", 10);
    ids_pub_      = create_publisher<std_msgs::msg::Int32MultiArray>("/aruco/ids", 10);
    distance_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/aruco/distances", 10);
    debug_pub_    = create_publisher<sensor_msgs::msg::Image>("/aruco/image_debug", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    const auto period = std::chrono::duration<double>(1.0 / fps);
    timer_ = create_wall_timer(period, std::bind(&ArucoDistanceNode::timerCallback, this));

    RCLCPP_INFO(get_logger(),
      "ArUco distance node started (camera=%d, dict=%d, marker_size=%.3f m, fps=%.1f)",
      camera_index, dict_id, marker_size_, fps);
  }

  ~ArucoDistanceNode() { cap_.release(); }

private:
  void timerCallback()
  {
    cv::Mat image;
    if (!cap_.read(image) || image.empty()) {
      RCLCPP_WARN(get_logger(), "Empty frame from camera, skipping.");
      return;
    }

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    detector_->detectMarkers(image, corners, ids, rejected);

    if (ids.empty()) return;

    const size_t N = ids.size();
    const float half = static_cast<float>(marker_size_) / 2.f;
    const std::vector<cv::Point3f> obj_points = {
      {-half,  half, 0.f},
      { half,  half, 0.f},
      { half, -half, 0.f},
      {-half, -half, 0.f},
    };

    std::vector<cv::Vec3d> rvecs(N), tvecs(N);
    for (size_t i = 0; i < N; ++i) {
      cv::solvePnP(obj_points, corners[i], camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i]);
    }

    const auto stamp = now();

    // ---- /aruco/ids ----
    std_msgs::msg::Int32MultiArray ids_msg;
    for (int id : ids) ids_msg.data.push_back(id);
    ids_pub_->publish(ids_msg);

    // ---- /aruco/poses + /tf ----
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

    // ---- /aruco/distances — N×N symmetric distance matrix ----
    // data[i*N + j] = Euclidean 3-D distance in metres; row/col order matches /aruco/ids
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
        const double dx = tvecs[i][0] - tvecs[j][0];
        const double dy = tvecs[i][1] - tvecs[j][1];
        const double dz = tvecs[i][2] - tvecs[j][2];
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        dist_msg.data[i * N + j] = dist;
        dist_msg.data[j * N + i] = dist;
      }
    }
    distance_pub_->publish(dist_msg);

    // ---- /aruco/image_debug ----
    cv::aruco::drawDetectedMarkers(image, corners, ids);
    for (size_t i = 0; i < N; ++i) {
      cv::drawFrameAxes(image, camera_matrix_, dist_coeffs_,
                        rvecs[i], tvecs[i], half * 0.5f);
      const cv::Point2f centre =
        (corners[i][0] + corners[i][1] + corners[i][2] + corners[i][3]) * 0.25f;
      const std::string label =
        "id:" + std::to_string(ids[i]) + " z:" + cv::format("%.2f", tvecs[i][2]) + "m";
      cv::putText(image, label,
                  cv::Point(static_cast<int>(centre.x) + 5, static_cast<int>(centre.y) - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
    }
    std_msgs::msg::Header hdr;
    hdr.stamp    = stamp;
    hdr.frame_id = camera_frame_;
    debug_pub_->publish(*cv_bridge::CvImage(hdr, "bgr8", image).toImageMsg());
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
