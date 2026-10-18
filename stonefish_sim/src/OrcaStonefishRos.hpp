/*
    OrcaStonefishRos.hpp

    The ROS 2 face of the Stonefish Orca executable (orca_rl_sim).

    Threads:
      - ROS 2 executor thread : runs the subscription / service callbacks
                                below. They only copy data into
                                mutex-protected slots.
      - Stonefish sim thread  : OrcaRLSimulationManager::SimulationStepCompleted
                                reads those slots (TakeCommand, TakeReset...)
                                and calls the Publish* methods
                                (rclcpp publishers are thread-safe).
    Nothing here touches Stonefish objects, so no Stonefish call ever runs
    on the executor thread.

    Conventions on every topic (see README "Coordinate frames"):
      world frame  "world_ned"          x north, y east, z down
      body frame   "<robot>/base_link_frd" x forward, y starboard, z down
      yaw / yaw rate positive clockwise seen from above
*/

#ifndef ORCA_STONEFISH_ROS_HPP
#define ORCA_STONEFISH_ROS_HPP

#include "OrcaRLCommandAdapter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <array>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

enum class OrcaCommandKind
{
    NONE,       // nothing received yet
    SETPOINT,   // /orca/cmd/setpoint : speed, yaw rate, depth rate
    THRUST      // /orca/cmd/thrust   : raw actuator forces (manual test path)
};

struct OrcaCommand
{
    OrcaCommandKind kind = OrcaCommandKind::NONE;
    OrcaRLSetpoint setpoint;
    OrcaRLForces thrust;
    double receivedSimTime = -1.0;   // sim time of arrival
};

// Plain data the sim thread hands over for publishing.
struct OrcaImuData { double roll, pitch, yaw, wx, wy, wz, ax, ay, az; };
struct OrcaDvlData { double u, v, w, altitude; bool bottomLock; };
struct OrcaGpsData { double lat, lon, north, east; bool fix; };
struct OrcaOdomData { double x, y, z, qx, qy, qz, qw, u, v, w, p, q, r; };
struct OrcaScanData { double angleMin, angleIncrement, rangeMin, rangeMax; std::vector<float> ranges; };

class OrcaStonefishRos : public rclcpp::Node
{
public:
    explicit OrcaStonefishRos(const std::string& robotName);

    // ---- read by the sim thread ----
    void SetSimTime(double t);
    OrcaCommand LatestCommand() const;
    bool TakeResetRequest();
    std::optional<bool> TakeAutopilotRequest();
    double commandTimeout() const { return commandTimeout_; }

    // ---- published from the sim thread ----
    void PublishClock(double t);
    void PublishImu(double t, const OrcaImuData& d);
    void PublishDvl(double t, const OrcaDvlData& d);
    void PublishPressure(double t, double gaugePa);
    void PublishGps(double t, const OrcaGpsData& d);
    void PublishGroundTruth(double t, const OrcaOdomData& d);
    void PublishProfiler(double t, const OrcaScanData& d);
    void PublishMultibeam(double t, const OrcaScanData& d);
    void PublishActuators(double t, const OrcaRLForces& applied, double mainThrustActual,
                          const OrcaRLSetpoint& setpoint, double depthRef, int commandKind);
    void PublishContact(double t, bool inContact, const std::string& otherEntity);
    void PublishStatus(const std::string& text);

    const std::string& bodyFrame() const { return bodyFrame_; }

private:
    builtin_interfaces::msg::Time Stamp(double t) const;

    void OnSetpoint(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void OnThrust(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    std::string robotName_;
    std::string worldFrame_;
    std::string bodyFrame_;
    double commandTimeout_;

    mutable std::mutex mtx_;
    OrcaCommand command_;
    double simTime_ = 0.0;
    bool resetRequested_ = false;
    std::optional<bool> autopilotRequest_;

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr setpointSub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr thrustSub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr resetSrv_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr autopilotSrv_;

    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clockPub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imuPub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr dvlPub_;
    rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr altitudePub_;
    rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr pressurePub_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr gpsPub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr gtPub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr profilerPub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr multibeamPub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr actuatorPub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr contactPub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr contactNamePub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr statusPub_;
};

#endif // ORCA_STONEFISH_ROS_HPP
