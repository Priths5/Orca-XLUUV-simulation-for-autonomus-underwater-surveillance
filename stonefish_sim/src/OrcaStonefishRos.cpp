#include "OrcaStonefishRos.hpp"

#include <cmath>
#include <limits>

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ZYX (yaw-pitch-roll) Euler angles of body FRD in world NED -> quaternion.
void EulerToQuaternion(double roll, double pitch, double yaw,
                       double& qx, double& qy, double& qz, double& qw)
{
    double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
    double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
    double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
    qw = cr * cp * cy + sr * sp * sy;
    qx = sr * cp * cy - cr * sp * sy;
    qy = cr * sp * cy + sr * cp * sy;
    qz = cr * cp * sy - sr * sp * cy;
}
}

OrcaStonefishRos::OrcaStonefishRos(const std::string& robotName)
    : rclcpp::Node("orca_stonefish"), robotName_(robotName)
{
    worldFrame_ = declare_parameter<std::string>("world_frame", "world_ned");
    bodyFrame_ = declare_parameter<std::string>("body_frame", "orca/base_link_frd");
    commandTimeout_ = declare_parameter<double>("command_timeout", 3.0);

    auto sensorQos = rclcpp::SensorDataQoS();
    auto reliable = rclcpp::QoS(10);

    setpointSub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/orca/cmd/setpoint", reliable,
        std::bind(&OrcaStonefishRos::OnSetpoint, this, std::placeholders::_1));
    thrustSub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/orca/cmd/thrust", reliable,
        std::bind(&OrcaStonefishRos::OnThrust, this, std::placeholders::_1));

    resetSrv_ = create_service<std_srvs::srv::Trigger>(
        "/orca/reset",
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
               std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
            std::lock_guard<std::mutex> lk(mtx_);
            resetRequested_ = true;
            command_ = OrcaCommand();
            res->success = true;
            res->message = "Orca respawn queued for the next physics step (Robot::Respawn). "
                           "Static obstacles stay; animated fish keep their looping clock.";
        });
    autopilotSrv_ = create_service<std_srvs::srv::SetBool>(
        "/orca/set_autopilot",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
               std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
            std::lock_guard<std::mutex> lk(mtx_);
            autopilotRequest_ = req->data;
            res->success = true;
            res->message = req->data ? "autopilot (lawnmower survey) on" : "external RL control on";
        });

    clockPub_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());
    imuPub_ = create_publisher<sensor_msgs::msg::Imu>("/orca/imu", sensorQos);
    dvlPub_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/orca/dvl/twist", sensorQos);
    altitudePub_ = create_publisher<sensor_msgs::msg::Range>("/orca/dvl/altitude", sensorQos);
    pressurePub_ = create_publisher<sensor_msgs::msg::FluidPressure>("/orca/pressure", sensorQos);
    gpsPub_ = create_publisher<sensor_msgs::msg::NavSatFix>("/orca/gps", sensorQos);
    gtPub_ = create_publisher<nav_msgs::msg::Odometry>("/orca/ground_truth/odometry", sensorQos);
    profilerPub_ = create_publisher<sensor_msgs::msg::LaserScan>("/orca/profiler", sensorQos);
    multibeamPub_ = create_publisher<sensor_msgs::msg::LaserScan>("/orca/multibeam", sensorQos);
    actuatorPub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/orca/actuators", reliable);
    contactPub_ = create_publisher<std_msgs::msg::Bool>("/orca/contact", reliable);
    contactNamePub_ = create_publisher<std_msgs::msg::String>("/orca/contact/entity", reliable);
    statusPub_ = create_publisher<std_msgs::msg::String>("/orca/status", rclcpp::QoS(1).transient_local());

    RCLCPP_INFO(get_logger(), "orca_stonefish ROS 2 interface up for robot '%s' (frames %s / %s)",
                robotName_.c_str(), worldFrame_.c_str(), bodyFrame_.c_str());
}

builtin_interfaces::msg::Time OrcaStonefishRos::Stamp(double t) const
{
    builtin_interfaces::msg::Time s;
    if (!(t > 0.0))
        t = 0.0;
    s.sec = static_cast<int32_t>(std::floor(t));
    s.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
    return s;
}

void OrcaStonefishRos::SetSimTime(double t)
{
    std::lock_guard<std::mutex> lk(mtx_);
    simTime_ = t;
}

void OrcaStonefishRos::OnSetpoint(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg->data.size() < 3)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "/orca/cmd/setpoint needs [surge_speed_mps, yaw_rate_radps, depth_rate_mps]");
        return;
    }
    std::lock_guard<std::mutex> lk(mtx_);
    command_.kind = OrcaCommandKind::SETPOINT;
    command_.setpoint.surgeSpeed = msg->data[0];
    command_.setpoint.yawRate = msg->data[1];
    command_.setpoint.depthRate = msg->data[2];
    command_.receivedSimTime = simTime_;
}

void OrcaStonefishRos::OnThrust(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg->data.size() < 3)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "/orca/cmd/thrust needs [surge_N, yaw_N, heave_N]");
        return;
    }
    std::lock_guard<std::mutex> lk(mtx_);
    command_.kind = OrcaCommandKind::THRUST;
    command_.thrust.surge = msg->data[0];
    command_.thrust.yaw = msg->data[1];
    command_.thrust.heave = msg->data[2];
    command_.receivedSimTime = simTime_;
}

OrcaCommand OrcaStonefishRos::LatestCommand() const
{
    std::lock_guard<std::mutex> lk(mtx_);
    return command_;
}

bool OrcaStonefishRos::TakeResetRequest()
{
    std::lock_guard<std::mutex> lk(mtx_);
    bool r = resetRequested_;
    resetRequested_ = false;
    return r;
}

std::optional<bool> OrcaStonefishRos::TakeAutopilotRequest()
{
    std::lock_guard<std::mutex> lk(mtx_);
    auto r = autopilotRequest_;
    autopilotRequest_.reset();
    return r;
}

void OrcaStonefishRos::PublishClock(double t)
{
    rosgraph_msgs::msg::Clock c;
    c.clock = Stamp(t);
    clockPub_->publish(c);
}

void OrcaStonefishRos::PublishImu(double t, const OrcaImuData& d)
{
    sensor_msgs::msg::Imu m;
    m.header.stamp = Stamp(t);
    m.header.frame_id = bodyFrame_;
    EulerToQuaternion(d.roll, d.pitch, d.yaw, m.orientation.x, m.orientation.y,
                      m.orientation.z, m.orientation.w);
    m.angular_velocity.x = d.wx;
    m.angular_velocity.y = d.wy;
    m.angular_velocity.z = d.wz;
    m.linear_acceleration.x = d.ax;
    m.linear_acceleration.y = d.ay;
    m.linear_acceleration.z = d.az;
    // <noise angle="0.0001" angular_velocity="0.0001"/> in the .scn
    for (int i : {0, 4, 8})
    {
        m.orientation_covariance[i] = 1e-8;
        m.angular_velocity_covariance[i] = 1e-8;
    }
    m.linear_acceleration_covariance[0] = -1.0;   // not modelled
    imuPub_->publish(m);
}

void OrcaStonefishRos::PublishDvl(double t, const OrcaDvlData& d)
{
    sensor_msgs::msg::Range alt;
    alt.header.stamp = Stamp(t);
    alt.header.frame_id = bodyFrame_;
    alt.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
    alt.field_of_view = static_cast<float>(60.0 * M_PI / 180.0);
    alt.min_range = 0.5f;
    alt.max_range = 100.0f;
    // REP 117: +inf = nothing detected in range
    alt.range = d.bottomLock ? static_cast<float>(d.altitude) : std::numeric_limits<float>::infinity();
    altitudePub_->publish(alt);

    if (!d.bottomLock)
        return;   // no bottom lock -> no velocity measurement; do not invent one
    geometry_msgs::msg::TwistWithCovarianceStamped m;
    m.header.stamp = Stamp(t);
    m.header.frame_id = bodyFrame_;
    m.twist.twist.linear.x = d.u;
    m.twist.twist.linear.y = d.v;
    m.twist.twist.linear.z = d.w;
    double var = 0.0015 * 0.0015;   // <noise velocity="0.0015"/>
    m.twist.covariance[0] = m.twist.covariance[7] = m.twist.covariance[14] = var;
    for (int i : {21, 28, 35})
        m.twist.covariance[i] = -1.0;   // angular part not measured
    dvlPub_->publish(m);
}

void OrcaStonefishRos::PublishPressure(double t, double gaugePa)
{
    sensor_msgs::msg::FluidPressure m;
    m.header.stamp = Stamp(t);
    m.header.frame_id = bodyFrame_;
    m.fluid_pressure = gaugePa;     // Stonefish Pressure sensor = GAUGE pressure (no atmosphere)
    m.variance = 10.0 * 10.0;       // <noise pressure="10.0"/>
    pressurePub_->publish(m);
}

void OrcaStonefishRos::PublishGps(double t, const OrcaGpsData& d)
{
    sensor_msgs::msg::NavSatFix m;
    m.header.stamp = Stamp(t);
    m.header.frame_id = bodyFrame_;
    m.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
    if (d.fix)
    {
        m.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
        m.latitude = d.lat;
        m.longitude = d.lon;
        m.altitude = 0.0;
        m.position_covariance[0] = m.position_covariance[4] = 0.25;   // ned_position 0.5 m
        m.position_covariance[8] = 1e6;
        m.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
    }
    else
    {
        // antenna under water: Stonefish GPS returns no fix
        m.status.status = sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX;
        m.latitude = m.longitude = m.altitude = kNaN;
        m.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
    }
    gpsPub_->publish(m);
}

void OrcaStonefishRos::PublishGroundTruth(double t, const OrcaOdomData& d)
{
    nav_msgs::msg::Odometry m;
    m.header.stamp = Stamp(t);
    m.header.frame_id = worldFrame_;
    m.child_frame_id = bodyFrame_;
    m.pose.pose.position.x = d.x;
    m.pose.pose.position.y = d.y;
    m.pose.pose.position.z = d.z;
    m.pose.pose.orientation.x = d.qx;
    m.pose.pose.orientation.y = d.qy;
    m.pose.pose.orientation.z = d.qz;
    m.pose.pose.orientation.w = d.qw;
    m.twist.twist.linear.x = d.u;
    m.twist.twist.linear.y = d.v;
    m.twist.twist.linear.z = d.w;
    m.twist.twist.angular.x = d.p;
    m.twist.twist.angular.y = d.q;
    m.twist.twist.angular.z = d.r;
    gtPub_->publish(m);
}

static sensor_msgs::msg::LaserScan MakeScan(const OrcaScanData& d)
{
    sensor_msgs::msg::LaserScan m;
    m.angle_min = static_cast<float>(d.angleMin);
    m.angle_increment = static_cast<float>(d.angleIncrement);
    m.angle_max = static_cast<float>(d.angleMin + d.angleIncrement * (d.ranges.empty() ? 0 : d.ranges.size() - 1));
    m.range_min = static_cast<float>(d.rangeMin);
    m.range_max = static_cast<float>(d.rangeMax);
    m.ranges = d.ranges;
    return m;
}

void OrcaStonefishRos::PublishProfiler(double t, const OrcaScanData& d)
{
    auto m = MakeScan(d);
    m.header.stamp = Stamp(t);
    m.header.frame_id = robotName_ + "/profiler_frd";
    profilerPub_->publish(m);
}

void OrcaStonefishRos::PublishMultibeam(double t, const OrcaScanData& d)
{
    auto m = MakeScan(d);
    m.header.stamp = Stamp(t);
    m.header.frame_id = robotName_ + "/multibeam";
    multibeamPub_->publish(m);
}

void OrcaStonefishRos::PublishActuators(double t, const OrcaRLForces& applied, double mainThrustActual,
                                        const OrcaRLSetpoint& sp, double depthRef, int commandKind)
{
    (void)t;
    std_msgs::msg::Float64MultiArray m;
    // [0..2] force requested on MainThruster/YawThruster/HeaveThruster [N]
    // [3]    MainThruster thrust actually produced (SimpleThruster::getThrust) [N]
    // [4..6] setpoint in use (surge m/s, yaw rate rad/s, depth rate m/s)
    // [7]    depth reference held by the adapter [m]
    // [8]    command source: 0 none/timeout, 1 setpoint, 2 raw thrust, 3 autopilot
    m.data = {applied.surge, applied.yaw, applied.heave, mainThrustActual,
              sp.surgeSpeed, sp.yawRate, sp.depthRate, depthRef, static_cast<double>(commandKind)};
    actuatorPub_->publish(m);
}

void OrcaStonefishRos::PublishContact(double t, bool inContact, const std::string& otherEntity)
{
    (void)t;
    std_msgs::msg::Bool b;
    b.data = inContact;
    contactPub_->publish(b);
    std_msgs::msg::String s;
    s.data = otherEntity;
    contactNamePub_->publish(s);
}

void OrcaStonefishRos::PublishStatus(const std::string& text)
{
    std_msgs::msg::String s;
    s.data = text;
    statusPub_->publish(s);
}
