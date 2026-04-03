/**
 * WaterLinked 3D Sonar Driver - ROS 2 C++ implementation
 * Uses rclcpp::Waitable to integrate the UDP socket into the ROS2 executor.
 *
 * Wire protocol: RIP2 = [magic 4B][total_len 4B LE][snappy(protobuf)][crc32 4B
 * LE] HTTP API used to enable acoustics and configure UDP multicast on
 * startup/shutdown.
 */

#include <rcl/wait.h>
#include <rclcpp/guard_condition.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/waitable.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <arpa/inet.h>
#include <endian.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <curl/curl.h>
#include <snappy.h>
#include <zlib.h>

#include "WaterLinkedSonarIntegrationProtocol.pb.h"
#include <google/protobuf/any.pb.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace proto = waterlinked::sonar::protocol;

// ── constants
// ─────────────────────────────────────────────────────────────────
static constexpr int UDP_MAX_DATAGRAM_SIZE = 65507;
static constexpr char DEFAULT_MCAST_GRP[] = "224.0.0.96";
static constexpr int DEFAULT_MCAST_PORT = 4747;
static constexpr size_t IDENTIFIER_LEN = 4;
static constexpr size_t PKT_LEN_FIELD_LEN = 4;
static constexpr size_t CHECKSUM_LEN = 4;
static constexpr size_t HEADER_LEN = IDENTIFIER_LEN + PKT_LEN_FIELD_LEN;

// ── HTTP helper
// ───────────────────────────────────────────────────────────────
static size_t curl_write_discard(char * /*ptr*/, size_t sz, size_t n,
                                 void * /*ud*/) {
  return sz * n;
}

static bool http_post_json(const std::string &url, const std::string &body) {
  CURL *c = curl_easy_init();
  if (!c)
    return false;
  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, curl_write_discard);
  curl_easy_setopt(c, CURLOPT_TIMEOUT, 5L);
  CURLcode rc = curl_easy_perform(c);
  curl_slist_free_all(hdrs);
  curl_easy_cleanup(c);
  return rc == CURLE_OK;
}

// ── PointCloud2 helpers
// ───────────────────────────────────────────────────────
static sensor_msgs::msg::PointField
make_field(const std::string &name, uint32_t offset, uint8_t datatype) {
  sensor_msgs::msg::PointField f;
  f.name = name;
  f.offset = offset;
  f.datatype = datatype;
  f.count = 1;
  return f;
}

static sensor_msgs::msg::PointCloud2
build_xyz_cloud(const builtin_interfaces::msg::Time &stamp,
                const std::string &frame_id,
                const std::vector<std::array<float, 3>> &pts) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(pts.size());
  msg.fields = {
      make_field("x", 0, sensor_msgs::msg::PointField::FLOAT32),
      make_field("y", 4, sensor_msgs::msg::PointField::FLOAT32),
      make_field("z", 8, sensor_msgs::msg::PointField::FLOAT32),
  };
  msg.is_bigendian = false;
  msg.point_step = 12;
  msg.row_step = 12 * msg.width;
  msg.is_dense = true;
  msg.data.resize(msg.row_step);
  std::memcpy(msg.data.data(), pts.data(), msg.row_step);
  return msg;
}

#pragma pack(push, 1)
struct PointXYZI {
  float x, y, z;
  uint8_t intensity;
};
#pragma pack(pop)

static sensor_msgs::msg::PointCloud2
build_xyzi_cloud(const builtin_interfaces::msg::Time &stamp,
                 const std::string &frame_id,
                 const std::vector<PointXYZI> &pts) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(pts.size());
  msg.fields = {
      make_field("x", 0, sensor_msgs::msg::PointField::FLOAT32),
      make_field("y", 4, sensor_msgs::msg::PointField::FLOAT32),
      make_field("z", 8, sensor_msgs::msg::PointField::FLOAT32),
      make_field("intensity", 12, sensor_msgs::msg::PointField::UINT8),
  };
  msg.is_bigendian = false;
  msg.point_step = static_cast<uint32_t>(sizeof(PointXYZI));
  msg.row_step = msg.point_step * msg.width;
  msg.is_dense = true;
  msg.data.resize(msg.row_step);
  std::memcpy(msg.data.data(), pts.data(), msg.row_step);
  return msg;
}

// ── UdpWaitable
// ───────────────────────────────────────────────────────────────
//
// This class bridges a raw UDP socket into the ROS2 executor's wait loop.
//
// It owns:
//   • The UDP socket file descriptor
//   • A GuardCondition (the ROS2 object that can be put in a wait set)
//   • A tiny watcher thread (only does select() + trigger())
//
// The executor owns the lifecycle of execute() — it is always called on the
// same thread as spin(), so the SonarDriver's maps, buffers, and publishers
// need no mutex protection.
class UdpWaitable : public rclcpp::Waitable {
public:
  // Callback type: called with (raw_bytes, length) for every received packet.
  // Runs on the executor thread (i.e. the spin() thread).
  using PacketCallback = std::function<void(const uint8_t *, ssize_t)>;

  UdpWaitable(rclcpp::Context::SharedPtr context, int sock, PacketCallback cb)
      : guard_condition_(context), sock_(sock), callback_(std::move(cb)) {
    running_ = true;
    watcher_ = std::thread(&UdpWaitable::watch_loop, this);
  }

  ~UdpWaitable() override {
    // Signal the watcher to stop and wait for it.
    // The watcher only blocks in select() with a 50ms timeout,
    // so this join completes quickly.
    running_ = false;
    if (watcher_.joinable())
      watcher_.join();
  }

  // ── rclcpp::Waitable interface ────────────────────────────────────────────

  // Step 1: Tell the executor how many guard conditions we need in the wait
  // set.
  size_t get_number_of_ready_guard_conditions() override { return 1; }

  // Step 2: The executor calls this to register our guard condition into its
  // internal wait set before calling rcl_wait(). gc_index_ is filled in by
  // rcl_wait_set_add_guard_condition so we know which slot is ours later.
  void add_to_wait_set(rcl_wait_set_t &wait_set) override {
    // rcl API takes a pointer, so pass &wait_set even though we receive a ref
    rcl_ret_t ret = rcl_wait_set_add_guard_condition(
        &wait_set, &guard_condition_.get_rcl_guard_condition(), &gc_index_);
    if (ret != RCL_RET_OK) {
      throw std::runtime_error(
          "UdpWaitable: failed to add guard condition to wait set");
    }
  }

  // Step 3: After rcl_wait() returns, the executor calls this to check if WE
  // were the reason it woke up. rcl_wait() sets triggered slots to non-null
  // and clears untriggered slots to null.
  bool is_ready(const rcl_wait_set_t &wait_set) override {
    return wait_set.guard_conditions[gc_index_] != nullptr;
  }

  // Step 4a: take_data() is called (while holding the entity lock) to extract
  // the data before execute(). We drain the socket here so that execute()
  // receives a ready-made list of raw packets — no socket reads in execute().
  std::shared_ptr<void> take_data() override {
    // Read all UDP packets currently available (MSG_DONTWAIT = non-blocking).
    // By the time take_data() is called, select() already confirmed data is
    // available, so at least one recvfrom() will succeed immediately.
    auto packets = std::make_shared<std::vector<std::vector<uint8_t>>>();
    std::vector<uint8_t> buf(UDP_MAX_DATAGRAM_SIZE);
    while (true) {
      ssize_t n = recvfrom(sock_, buf.data(), buf.size(), MSG_DONTWAIT, nullptr,
                           nullptr);
      if (n < 0)
        break; // EAGAIN/EWOULDBLOCK: no more packets right now
      packets->emplace_back(buf.data(), buf.data() + n);
    }
    return packets;
  }

  // Step 4b: execute() is called on the spin() thread with the data returned
  // by take_data(). This is where we invoke the node's packet processing.
  void execute(const std::shared_ptr<void> &data) override {
    auto packets =
        std::static_pointer_cast<std::vector<std::vector<uint8_t>>>(data);
    if (!packets)
      return;
    for (const auto &pkt : *packets) {
      callback_(pkt.data(), static_cast<ssize_t>(pkt.size()));
    }
  }

private:
  // The watcher thread: extremely simple — just watches the socket and pokes
  // the executor when data is available. Does NO parsing, NO publishing,
  // NO shared state access beyond the atomic running_ flag.
  void watch_loop() {
    while (running_) {
      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(sock_, &fds);
      // 50ms timeout so we re-check running_ and can exit cleanly.
      timeval tv{0, 50'000};
      if (select(sock_ + 1, &fds, nullptr, nullptr, &tv) > 0) {
        // A packet arrived — wake the executor.
        guard_condition_.trigger();
      }
    }
  }

  rclcpp::GuardCondition guard_condition_;
  int sock_;
  PacketCallback callback_;
  std::atomic<bool> running_{false};
  std::thread watcher_;
  size_t gc_index_{0};
};

// ── SonarDriver node
// ──────────────────────────────────────────────────────────
class SonarDriver : public rclcpp::Node {
public:
  SonarDriver() : Node("WL_sonar_driver") {
    sonar_ip_ = "192.168.1.96";
    frame_id_ = "sonar_link";

    pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/sonar_3d/pointcloud", 10);
    pub_img_ = create_publisher<sensor_msgs::msg::Image>(
        "/sonar_3d/strength_image", 10);
    pub_cloud_xyzi_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/sonar_3d/pointcloud_intensity", 10);

    std::string base = "http://" + sonar_ip_;
    if (!http_post_json(base + "/api/v1/integration/acoustics/enabled", "true"))
      RCLCPP_WARN(get_logger(), "Failed to enable acoustics via HTTP");
    if (!http_post_json(
            base + "/api/v1/integration/udp",
            R"({"mode":"multicast","unicast_destination_ip":"","unicast_destination_port":0})"))
      RCLCPP_WARN(get_logger(), "Failed to configure UDP multicast via HTTP");

    sock_ = open_multicast_socket();
    waitable_ = std::make_shared<UdpWaitable>(
        get_node_base_interface()->get_context(), sock_,
        [this](const uint8_t *data, ssize_t n) { on_packet(data, n); });

    // Register the waitable with the node's default callback group.
    // The executor will now include it in the wait set on every spin cycle.
    get_node_waitables_interface()->add_waitable(
        waitable_, get_node_base_interface()->get_default_callback_group());
  }

  ~SonarDriver() override {
    // 1. Destroy the waitable first — this joins the watcher thread,
    //    which may still be inside select(). The watcher thread only
    //    holds sock_ as a read-only fd, so it's safe to join before close().
    waitable_.reset();

    // 2. Now safe to close the socket.
    if (sock_ >= 0) {
      close(sock_);
      sock_ = -1;
    }

    http_post_json("http://" + sonar_ip_ +
                       "/api/v1/integration/acoustics/enabled",
                   "false");
  }

private:
  std::string sonar_ip_;
  std::string frame_id_;
  int sock_ = -1;

  // Declare waitable_ AFTER sock_ so its destructor runs FIRST
  // (C++ destroys members in reverse declaration order).
  std::shared_ptr<UdpWaitable> waitable_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_xyzi_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_;

  std::map<uint32_t, proto::RangeImage> range_buf_;
  std::map<uint32_t, proto::BitmapImageGreyscale8> strength_buf_;
  std::vector<uint8_t> proto_bytes_; // reused per call
  static constexpr int MAX_BUF = 50;

  // ── socket ────────────────────────────────────────────────────────────────
  int open_multicast_socket() {
    int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s < 0)
      throw std::runtime_error("socket() failed");

    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(DEFAULT_MCAST_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
      close(s);
      throw std::runtime_error("bind() failed");
    }

    ip_mreq mreq{};
    inet_pton(AF_INET, DEFAULT_MCAST_GRP, &mreq.imr_multiaddr);
    mreq.imr_interface.s_addr = INADDR_ANY;
    setsockopt(s, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));
    return s;
  }

  // ── called by UdpWaitable::execute() on the spin() thread ────────────────
  void on_packet(const uint8_t *data, ssize_t n) {
    if (!parse_rip_packet(data, n, proto_bytes_))
      return;

    proto::Packet pkt;
    if (!pkt.ParseFromArray(proto_bytes_.data(),
                            static_cast<int>(proto_bytes_.size()))) {
      RCLCPP_WARN(get_logger(), "Failed to parse protobuf Packet");
      return;
    }

    const auto &any = pkt.msg();
    proto::RangeImage range_msg;
    proto::BitmapImageGreyscale8 bitmap_msg;

    if (any.UnpackTo(&range_msg)) {
      handle_range_image(range_msg);
    } else if (any.UnpackTo(&bitmap_msg)) {
      handle_strength_image(bitmap_msg);
    } else {
      RCLCPP_DEBUG(get_logger(), "Unknown msg type: %s",
                   any.type_url().c_str());
      return;
    }

    buffer_handler();
  }

  // ── RIP2 packet parser ────────────────────────────────────────────────────
  bool parse_rip_packet(const uint8_t *data, ssize_t n,
                        std::vector<uint8_t> &proto_out) {
    if (n < static_cast<ssize_t>(HEADER_LEN + CHECKSUM_LEN))
      return false;

    bool rip1 = std::memcmp(data, "RIP1", 4) == 0;
    bool rip2 = std::memcmp(data, "RIP2", 4) == 0;
    if (!rip1 && !rip2)
      return false;

    uint32_t total_len;
    std::memcpy(&total_len, data + IDENTIFIER_LEN, PKT_LEN_FIELD_LEN);
    total_len = le32toh(total_len);

    if (static_cast<ssize_t>(total_len) > n)
      return false;
    if (total_len < HEADER_LEN + CHECKSUM_LEN)
      return false;

    uint32_t crc_calc = static_cast<uint32_t>(
        crc32(crc32(0L, Z_NULL, 0), data, total_len - CHECKSUM_LEN));
    uint32_t crc_recv;
    std::memcpy(&crc_recv, data + total_len - CHECKSUM_LEN, CHECKSUM_LEN);
    crc_recv = le32toh(crc_recv);
    if (crc_calc != crc_recv) {
      RCLCPP_WARN(get_logger(), "RIP CRC mismatch (calc=0x%08x recv=0x%08x)",
                  crc_calc, crc_recv);
      return false;
    }

    const uint8_t *payload = data + HEADER_LEN;
    size_t payload_len = total_len - HEADER_LEN - CHECKSUM_LEN;

    if (rip2) {
      size_t out_len = 0;
      if (!snappy::GetUncompressedLength(
              reinterpret_cast<const char *>(payload), payload_len, &out_len)) {
        RCLCPP_WARN(get_logger(), "Snappy: could not read uncompressed length");
        return false;
      }
      proto_out.resize(out_len);
      if (!snappy::RawUncompress(reinterpret_cast<const char *>(payload),
                                 payload_len,
                                 reinterpret_cast<char *>(proto_out.data()))) {
        RCLCPP_WARN(get_logger(), "Snappy decompression failed");
        return false;
      }
    } else {
      proto_out.assign(payload, payload + payload_len);
    }
    return true;
  }

  // ── timestamp conversion ──────────────────────────────────────────────────
  static builtin_interfaces::msg::Time
  to_ros_time(const google::protobuf::Timestamp &ts) {
    builtin_interfaces::msg::Time t;
    t.sec = static_cast<int32_t>(ts.seconds());
    t.nanosec = static_cast<uint32_t>(ts.nanos());
    return t;
  }

  // ── message handlers ──────────────────────────────────────────────────────
  void handle_range_image(const proto::RangeImage &msg) {
    range_buf_[msg.header().sequence_id()] = msg;
    auto stamp = to_ros_time(msg.header().timestamp());
    pub_cloud_->publish(
        build_xyz_cloud(stamp, frame_id_, range_image_to_xyz(msg)));
  }

  void handle_strength_image(const proto::BitmapImageGreyscale8 &msg) {
    if (msg.type() != proto::SIGNAL_STRENGTH_IMAGE)
      return;
    strength_buf_[msg.header().sequence_id()] = msg;

    sensor_msgs::msg::Image img;
    img.header.stamp = to_ros_time(msg.header().timestamp());
    img.header.frame_id = frame_id_;
    img.width = msg.width();
    img.height = msg.height();
    img.encoding = "mono8";
    img.step = msg.width();

    const auto &raw = msg.image_pixel_data();
    img.data.resize(raw.size());
    for (uint32_t row = 0; row < msg.height(); ++row) {
      uint32_t src_row = msg.height() - 1 - row;
      std::memcpy(img.data.data() + row * msg.width(),
                  reinterpret_cast<const uint8_t *>(raw.data()) +
                      src_row * msg.width(),
                  msg.width());
    }
    pub_img_->publish(img);
  }

  // ── buffer handler ────────────────────────────────────────────────────────
  void buffer_handler() {
    std::vector<uint32_t> common;
    for (auto &[id, _] : range_buf_) {
      if (strength_buf_.count(id))
        common.push_back(id);
    }
    std::sort(common.begin(), common.end());

    for (uint32_t id : common) {
      auto r = range_buf_.extract(id);
      auto s = strength_buf_.extract(id);
      auto pts = range_image_to_xyzi(r.mapped(), s.mapped());
      auto stamp = to_ros_time(r.mapped().header().timestamp());
      pub_cloud_xyzi_->publish(build_xyzi_cloud(stamp, frame_id_, pts));
    }

    auto evict = [&](auto &buf, const char *name) {
      while (static_cast<int>(buf.size()) > MAX_BUF) {
        uint32_t oldest = buf.begin()->first;
        buf.erase(buf.begin());
        RCLCPP_WARN(get_logger(),
                    "Buffer overflow: dropped orphaned %s message (id=%u)",
                    name, oldest);
      }
    };
    evict(range_buf_, "Range");
    evict(strength_buf_, "Strength");
  }

  // ── geometry ──────────────────────────────────────────────────────────────
  static std::vector<std::array<float, 3>>
  range_image_to_xyz(const proto::RangeImage &img) {
    uint32_t w = img.width(), h = img.height();
    float fov_h = img.fov_horizontal() * static_cast<float>(M_PI) / 180.0f;
    float fov_v = img.fov_vertical() * static_cast<float>(M_PI) / 180.0f;
    float scale = img.image_pixel_scale();

    std::vector<std::array<float, 3>> pts;
    pts.reserve(w * h);
    for (uint32_t px = 0; px < w; ++px) {
      for (uint32_t py = 0; py < h; ++py) {
        uint32_t val = img.image_pixel_data(static_cast<int>(py * w + px));
        if (val == 0)
          continue;
        float dist = val * scale;
        float yaw = (static_cast<float>(px) / (w - 1)) * fov_h - fov_h / 2.0f;
        float pitch = (static_cast<float>(py) / (h - 1)) * fov_v - fov_v / 2.0f;
        pts.push_back({dist * std::cos(pitch) * std::cos(yaw),
                       dist * std::cos(pitch) * std::sin(yaw),
                       -dist * std::sin(pitch)});
      }
    }
    return pts;
  }

  static std::vector<PointXYZI>
  range_image_to_xyzi(const proto::RangeImage &r,
                      const proto::BitmapImageGreyscale8 &s) {
    uint32_t w = r.width(), h = r.height();
    float fov_h = r.fov_horizontal() * static_cast<float>(M_PI) / 180.0f;
    float fov_v = r.fov_vertical() * static_cast<float>(M_PI) / 180.0f;
    float scale = r.image_pixel_scale();
    const auto &raw_s = s.image_pixel_data();

    std::vector<PointXYZI> pts;
    pts.reserve(w * h);
    for (uint32_t px = 0; px < w; ++px) {
      for (uint32_t py = 0; py < h; ++py) {
        int idx = static_cast<int>(py * w + px);
        uint32_t val = r.image_pixel_data(idx);
        if (val == 0)
          continue;
        float dist = val * scale;
        float yaw = (static_cast<float>(px) / (w - 1)) * fov_h - fov_h / 2.0f;
        float pitch = (static_cast<float>(py) / (h - 1)) * fov_v - fov_v / 2.0f;
        uint8_t intensity = (idx < static_cast<int>(raw_s.size()))
                                ? static_cast<uint8_t>(raw_s[idx])
                                : 0u;
        pts.push_back({dist * std::cos(pitch) * std::cos(yaw),
                       dist * std::cos(pitch) * std::sin(yaw),
                       -dist * std::sin(pitch), intensity});
      }
    }
    return pts;
  }
};

// ── main
// ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SonarDriver>());
  rclcpp::shutdown();
  return 0;
}
