// camera_calibration.cpp
// ROS 2 node for intrinsic camera calibration using an ArUco GridBoard.
// Based on the IFROS Perception Lab 1 calibration procedure.
//
// Controls
//   'c'  — capture the current frame (all board markers must be visible)
//   ESC  — run calibration and save; node exits
//
// Output: OpenCV FileStorage YAML compatible with aruco_distance_node.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

class CameraCalibrationNode : public rclcpp::Node {
public:
  CameraCalibrationNode() : Node("camera_calibration_node") {
    declare_parameter("camera_index", 1);
    declare_parameter("dictionary", std::string("DICT_4X4_50"));
    declare_parameter("detector_params_file", std::string(""));
    declare_parameter("rows", 4);
    declare_parameter("cols", 2);
    declare_parameter("marker_length", 0.05);
    declare_parameter("marker_separation", 0.02);
    declare_parameter("min_frames", 20);
    declare_parameter("calibration_output_file", std::string(""));

    const int camera_index = get_parameter("camera_index").as_int();
    const auto dict_type = get_parameter("dictionary").as_string();
    const auto params_file = get_parameter("detector_params_file").as_string();
    const int rows = get_parameter("rows").as_int();
    const int cols = get_parameter("cols").as_int();
    const float marker_len =
        static_cast<float>(get_parameter("marker_length").as_double());
    const float marker_sep =
        static_cast<float>(get_parameter("marker_separation").as_double());
    min_frames_ = get_parameter("min_frames").as_int();
    output_file_ = get_parameter("calibration_output_file").as_string();

    // ── ArUco dictionary
    // ──────────────────────────────────────────────────────
    static const std::map<std::string, int> kDictMap = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
    };

    auto it = kDictMap.find(dict_type);
    if (it == kDictMap.end()) {
      RCLCPP_FATAL(get_logger(), "Unknown dictionary: %s", dict_type.c_str());
      throw std::runtime_error("Unknown ArUco dictionary type");
    }
    auto dict = cv::aruco::getPredefinedDictionary(it->second);

    // ── Detector parameters
    // ───────────────────────────────────────────────────
    cv::aruco::DetectorParameters detector_params;
    cv::FileStorage fs_p(params_file, cv::FileStorage::READ);
    if (fs_p.isOpened()) {
      detector_params.readDetectorParameters(fs_p.root());
      fs_p.release();
      RCLCPP_INFO(get_logger(), "Detector params loaded from: %s",
                  params_file.c_str());
    } else {
      RCLCPP_WARN(get_logger(),
                  "Detector params file not found, using defaults: %s",
                  params_file.c_str());
    }

    detector_ =
        std::make_unique<cv::aruco::ArucoDetector>(dict, detector_params);
    board_ = std::make_unique<cv::aruco::GridBoard>(
        cv::Size(cols, rows), marker_len, marker_sep, dict);
    board_size_ = cv::Size(cols, rows);

    // ── Camera
    // ────────────────────────────────────────────────────────────────
    cap_.open(camera_index);
    if (!cap_.isOpened()) {
      RCLCPP_FATAL(get_logger(), "Cannot open camera index %d", camera_index);
      throw std::runtime_error("Camera open failed");
    }

    cv::namedWindow("Camera Calibration", cv::WINDOW_AUTOSIZE);

    RCLCPP_INFO(get_logger(),
                "Camera calibration ready — camera=%d  board=%dx%d  "
                "marker=%.3fm  sep=%.3fm",
                camera_index, cols, rows, marker_len, marker_sep);
    RCLCPP_INFO(
        get_logger(),
        "Press 'c' to capture a frame (need %d), ESC to calibrate and exit.",
        min_frames_);

    timer_ = create_wall_timer(
        std::chrono::milliseconds(33),
        std::bind(&CameraCalibrationNode::timerCallback, this));
  }

  ~CameraCalibrationNode() {
    cap_.release();
    cv::destroyAllWindows();
  }

private:
  void timerCallback() {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN(get_logger(), "Empty frame, skipping.");
      return;
    }

    // ── Detect markers and draw them ─────────────────────────────────────────
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    detector_->detectMarkers(frame, corners, ids, rejected);
    if (!ids.empty()) {
      cv::aruco::drawDetectedMarkers(frame, corners, ids);
    }

    // ── HUD overlay
    // ───────────────────────────────────────────────────────────
    const int captured = static_cast<int>(captured_corners_.size());
    const int needed = board_size_.width * board_size_.height;
    const cv::Scalar green(0, 255, 0);
    const cv::Scalar yellow(0, 220, 255);

    cv::putText(frame,
                "Captured: " + std::to_string(captured) + " / " +
                    std::to_string(min_frames_) +
                    "   'c' = capture   ESC = calibrate & exit",
                cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.6, green, 2);

    if (!ids.empty()) {
      const cv::Scalar col =
          (static_cast<int>(ids.size()) == needed) ? green : yellow;
      cv::putText(frame,
                  "Visible: " + std::to_string(ids.size()) + " / " +
                      std::to_string(needed) + " markers",
                  cv::Point(10, 56), cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2);
    }

    cv::imshow("Camera Calibration", frame);
    const char key = static_cast<char>(cv::waitKey(1));

    if (key == 'c') {
      captureFrame(frame, corners, ids, rejected);
    } else if (key == 27) { // ESC
      timer_->cancel();
      runCalibration(frame.size());
      rclcpp::shutdown();
    }
  }

  void captureFrame(const cv::Mat &frame,
                    std::vector<std::vector<cv::Point2f>> corners,
                    std::vector<int> ids,
                    std::vector<std::vector<cv::Point2f>> rejected) {
    // Refine detection before deciding whether to accept the frame
    detector_->refineDetectedMarkers(frame, *board_, corners, ids, rejected);

    const int needed = board_size_.width * board_size_.height;
    if (static_cast<int>(ids.size()) == needed) {
      captured_corners_.push_back(corners);
      captured_ids_.push_back(ids);
      RCLCPP_INFO(get_logger(), "Frame %zu captured.",
                  captured_corners_.size());
    } else {
      RCLCPP_WARN(
          get_logger(),
          "Only %zu / %d markers visible — reposition the board and try again.",
          ids.size(), needed);
    }
  }

  void runCalibration(const cv::Size &img_size) {
    const int n = static_cast<int>(captured_corners_.size());
    if (n < min_frames_) {
      RCLCPP_ERROR(
          get_logger(),
          "Only %d frames captured, need at least %d. Calibration aborted.", n,
          min_frames_);
      return;
    }

    RCLCPP_INFO(get_logger(), "Running calibration on %d frames…", n);

    std::vector<cv::Mat> obj_pts, img_pts;
    for (int i = 0; i < n; ++i) {
      cv::Mat cur_obj, cur_img;
      board_->matchImagePoints(captured_corners_[i], captured_ids_[i], cur_obj,
                               cur_img);
      if (cur_img.total() > 0 && cur_obj.total() > 0) {
        obj_pts.push_back(cur_obj);
        img_pts.push_back(cur_img);
      }
    }

    cv::Mat camera_matrix, dist_coeffs;
    const double rpe = cv::calibrateCamera(
        obj_pts, img_pts, img_size, camera_matrix, dist_coeffs, cv::noArray(),
        cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray(), 0,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100,
                         1e-6));

    RCLCPP_INFO(get_logger(), "Reprojection error: %.4f px", rpe);
    RCLCPP_INFO(get_logger(), "fx=%.4f  fy=%.4f  cx=%.4f  cy=%.4f",
                camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1),
                camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2));

    if (output_file_.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "calibration_output_file not set — results not saved to disk.");
      return;
    }

    cv::FileStorage fs(output_file_, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Cannot write to: %s", output_file_.c_str());
      return;
    }
    fs << "cameraMatrix" << camera_matrix;
    fs << "distCoeffs" << dist_coeffs;
    fs.release();
    RCLCPP_INFO(get_logger(), "Calibration saved to: %s", output_file_.c_str());
  }

  cv::VideoCapture cap_;
  std::unique_ptr<cv::aruco::ArucoDetector> detector_;
  std::unique_ptr<cv::aruco::GridBoard> board_;
  cv::Size board_size_;
  int min_frames_;
  std::string output_file_;

  std::vector<std::vector<std::vector<cv::Point2f>>> captured_corners_;
  std::vector<std::vector<int>> captured_ids_;

  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraCalibrationNode>());
  rclcpp::shutdown();
  return 0;
}
