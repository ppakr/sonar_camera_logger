#include <cmath>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

class ArucoDistanceNode : public rclcpp::Node
{
public:
  ArucoDistanceNode() : Node("aruco_distance_node")
  {
    declare_parameter("marker_size", 0.05);   // physical marker side length in metres
    declare_parameter("dictionary_id", 10);   // cv::aruco dict ID; 10 = DICT_6X6_250
    declare_parameter("image_topic", "/camera/image_raw");
    declare_parameter("camera_info_topic", "/camera/camera_info");
    declare_parameter("camera_frame", "camera_optical_frame");

    marker_size_   = get_parameter("marker_size").as_double();
    camera_frame_  = get_parameter("camera_frame").as_string();
    const int dict_id = get_parameter("dictionary_id").as_int();

    auto dict   = cv::aruco::getPredefinedDictionary(dict_id);
    auto params = cv::aruco::DetectorParameters();
    detector_   = std::make_unique<cv::aruco::ArucoDetector>(dict, params);

    const auto image_topic = get_parameter("image_topic").as_string();
    const auto info_topic  = get_parameter("camera_info_topic").as_string();

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, 10,
      std::bind(&ArucoDistanceNode::imageCallback, this, std::placeholders::_1));

    // Transient-local so we receive the latched camera info even if published before startup
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic, rclcpp::QoS(1).transient_local(),
      std::bind(&ArucoDistanceNode::cameraInfoCallback, this, std::placeholders::_1));

    pose_pub_     = create_publisher<geometry_msgs::msg::PoseArray>("/aruco/poses", 10);
    ids_pub_      = create_publisher<std_msgs::msg::Int32MultiArray>("/aruco/ids", 10);
    distance_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/aruco/distances", 10);
    debug_pub_    = create_publisher<sensor_msgs::msg::Image>("/aruco/image_debug", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(get_logger(),
      "ArUco distance node started (dict=%d, marker_size=%.3f m)", dict_id, marker_size_);
  }

private:
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (!camera_matrix_.empty()) return;
    camera_matrix_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();
    dist_coeffs_   = cv::Mat(msg->d.size(), 1, CV_64F, const_cast<double *>(msg->d.data())).clone();
    RCLCPP_INFO(get_logger(), "Camera intrinsics received.");
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (camera_matrix_.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for camera_info...");
      return;
    }

    cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    detector_->detectMarkers(image, corners, ids, rejected);

    if (ids.empty()) return;

    const size_t N = ids.size();

    // Marker 3-D corner points in marker-local frame (z=0 plane, origin at centre)
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

    // ---- /aruco/ids ----
    std_msgs::msg::Int32MultiArray ids_msg;
    for (int id : ids) ids_msg.data.push_back(id);
    ids_pub_->publish(ids_msg);

    // ---- /aruco/poses + /tf ----
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp    = msg->header.stamp;
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
      tf.header.stamp    = msg->header.stamp;
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
    // Access: data[i*N + j] = distance in metres between marker i and j
    // Row/column ordering matches the published /aruco/ids array
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
        dist_msg.data[j * N + i] = dist;  // symmetric
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
    debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg());
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

  std::unique_ptr<cv::aruco::ArucoDetector> detector_;
  cv::Mat camera_matrix_, dist_coeffs_;
  double marker_size_;
  std::string camera_frame_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
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
