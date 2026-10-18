#include "OrcaController.hpp"

#include <core/SimulationApp.h>
#include <core/SimulationManager.h>
#include <utils/SystemUtil.hpp>

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>

std::vector<SurveyWaypoint> BuildLawnmowerPattern(double x0, double y0,
                                                    double x1, double y1,
                                                    double laneSpacing)
{
    std::vector<SurveyWaypoint> wps;
    bool forward = true;
    for (double y = y0; y <= y1 + 1e-6; y += laneSpacing)
    {
        if (forward)
        {
            wps.push_back({x0, y});
            wps.push_back({x1, y});
        }
        else
        {
            wps.push_back({x1, y});
            wps.push_back({x0, y});
        }
        forward = !forward;
    }
    return wps;
}

static double WrapAngle(double a)
{
    while (a > M_PI)  a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

static double YawFromQuaternion(double qx, double qy, double qz, double qw)
{
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    return std::atan2(siny_cosp, cosy_cosp);
}

OrcaController::OrcaController(sf::SimulationManager* manager,
                                const std::string& robotName,
                                std::vector<SurveyWaypoint> waypoints,
                                double targetDepth,
                                double cruiseForce,
                                double yawGain,
                                double safetyDistance,
                                double avoidYawForce,
                                double waypointRadius,
                                double depthGain,
                                double depthDamping,
                                double heaveLimit)
    : manager_(manager), robotName_(robotName), waypoints_(std::move(waypoints)),
      currentWaypoint_(0), targetDepth_(targetDepth),
      cruiseForce_(cruiseForce), yawGain_(yawGain),
      safetyDistance_(safetyDistance), avoidYawForce_(avoidYawForce),
      waypointRadius_(waypointRadius),
      depthGain_(depthGain), depthDamping_(depthDamping), heaveLimit_(heaveLimit),
      bound(false), missionDone_(false), logWritten_(false), autopilot_(true),
      robot_(nullptr), mainThruster_(nullptr), yawThruster_(nullptr),
      heaveThruster_(nullptr), odometry_(nullptr), profiler_(nullptr),
      pressure_(nullptr), imu_(nullptr), dvl_(nullptr), multibeam_(nullptr),
      avoidMemory_(0.0), avoidYawHold_(0.0), simTime_(0.0), statusTimer_(0.0),
      trackLength_(0.0), minClearance_(std::numeric_limits<double>::infinity()),
      lastMultibeamStamp_(-1.0)
{
}

bool OrcaController::TryBind()
{
    if (bound)
        return true;

    robot_ = manager_->getRobot(robotName_);
    if (robot_ == nullptr)
        return false;

    // Stonefish prefixes actuator and sensor names with the robot name.
    auto act = [&](const char* n) { return robot_->getActuator(robotName_ + "/" + n); };
    auto sens = [&](const char* n) { return robot_->getSensor(robotName_ + "/" + n); };

    mainThruster_ = dynamic_cast<sf::SimpleThruster*>(act("MainThruster"));
    yawThruster_  = dynamic_cast<sf::Push*>(act("YawThruster"));
    heaveThruster_ = dynamic_cast<sf::Push*>(act("HeaveThruster"));
    odometry_     = dynamic_cast<sf::ScalarSensor*>(sens("Odometry"));
    profiler_     = dynamic_cast<sf::ScalarSensor*>(sens("Profiler"));
    pressure_     = dynamic_cast<sf::ScalarSensor*>(sens("Pressure"));
    imu_          = dynamic_cast<sf::ScalarSensor*>(sens("IMU"));
    dvl_          = dynamic_cast<sf::ScalarSensor*>(sens("DVL"));
    multibeam_    = dynamic_cast<sf::ScalarSensor*>(sens("Multibeam"));

    bound = (mainThruster_ != nullptr && yawThruster_ != nullptr &&
             odometry_ != nullptr && profiler_ != nullptr);
    if (bound)
    {
        cInfo("%s controller bound (heave=%d pressure=%d imu=%d dvl=%d multibeam=%d)",
              robotName_.c_str(),
              heaveThruster_ != nullptr, pressure_ != nullptr, imu_ != nullptr,
              dvl_ != nullptr, multibeam_ != nullptr);
    }
    return bound;
}

void OrcaController::FillProfiler(double& minDist, double& minAngle)
{
    minDist = std::numeric_limits<double>::infinity();
    minAngle = 0.0;
    observation_.profiler.fill(30.0);

    const std::vector<sf::Sample>* hist = profiler_->getHistory();
    if (hist == nullptr || hist->empty())
    {
        delete hist;
        return;
    }

    const double forwardCone = 70.0 * M_PI / 180.0;
    const double missRange = 25.0;
    const double bin0 = -80.0 * M_PI / 180.0;
    const double binSpan = 160.0 * M_PI / 180.0;

    for (const sf::Sample& s : *hist)
    {
        double angle = static_cast<double>(s.getValue(0));
        double dist  = static_cast<double>(s.getValue(1));
        if (dist >= missRange)
            continue;

        int bin = static_cast<int>((angle - bin0) / binSpan * kProfilerBins);
        if (bin >= 0 && bin < kProfilerBins)
            observation_.profiler[static_cast<size_t>(bin)] =
                std::min(observation_.profiler[static_cast<size_t>(bin)], dist);

        bool ahead = std::fabs(angle) <= forwardCone;
        bool alongside = dist < safetyDistance_ * 0.65;
        if ((ahead || alongside) && dist < minDist)
        {
            minDist = dist;
            minAngle = angle;
        }
    }

    delete hist;
}

void OrcaController::AccumulateMultibeam(double x, double y, double yaw)
{
    if (multibeam_ == nullptr)
        return;

    // getLastSample() hands back a copy whose getId() is always 0, which made
    // this accept only the very first fan; the sim timestamp is unique per sample.
    sf::Sample sample = multibeam_->getLastSample();
    double stamp = static_cast<double>(sample.getTimestamp());
    if (stamp < 0.0 || stamp == lastMultibeamStamp_ || sample.getNumOfDimensions() < 2)
        return;
    lastMultibeamStamp_ = stamp;

    const size_t n = sample.getNumOfDimensions();
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);

    for (size_t i = 0; i < n; ++i)
    {
        double dist = static_cast<double>(sample.getValue(i));
        if (dist < 0.4 || dist > 40.0)
            continue;

        double angle = -0.5 * kMultibeamFovRad +
                       (n == 1 ? 0.0 : (static_cast<double>(i) / static_cast<double>(n - 1)) * kMultibeamFovRad);
        // Sensor +X is pitched onto body +Z, so a beam is (0, sin, cos) in the body frame.
        double bodyY = dist * std::sin(angle);
        double worldX = x - sy * bodyY;
        double worldY = y + cy * bodyY;
        coverage_.insert({static_cast<int>(std::floor(worldX / kCellSize)),
                          static_cast<int>(std::floor(worldY / kCellSize))});
    }
}

void OrcaController::WriteLog(double x, double y, double z) const
{
    namespace fs = std::filesystem;
    fs::path dir = fs::path(sf::GetDataPath()) / "logs";
    std::error_code ec;
    fs::create_directories(dir, ec);

    fs::path file = dir / (robotName_ + "_survey.txt");
    std::ofstream out(file);
    if (!out)
        return;

    out << "robot " << robotName_ << "\n";
    out << "mission " << (missionDone_ ? "complete" : "running") << "\n";
    out << "time_s " << simTime_ << "\n";
    out << "track_m " << trackLength_ << "\n";
    out << "final_xyz " << x << " " << y << " " << z << "\n";
    out << "waypoints " << waypoints_.size() << " reached " << currentWaypoint_ << "\n";
    out << "min_clearance_m " << minClearance_ << "\n";
    out << "coverage_cells " << coverage_.size()
        << " cell_m " << kCellSize
        << " area_m2 " << (coverage_.size() * kCellSize * kCellSize) << "\n";
    cInfo("%s survey log written to %s (%zu cells, %.0f m track)",
          robotName_.c_str(), file.string().c_str(), coverage_.size(), trackLength_);
}

void OrcaController::Update(double dt)
{
    if (!bound && !TryBind())
        return;
    if (waypoints_.empty() || dt <= 0.0)
        return;

    simTime_ += dt;

    sf::Sample odom = odometry_->getLastSample();
    double x  = static_cast<double>(odom.getValue(0));
    double y  = static_cast<double>(odom.getValue(1));
    double zOdom = static_cast<double>(odom.getValue(2));
    double vx = static_cast<double>(odom.getValue(3));
    double vy = static_cast<double>(odom.getValue(4));
    double vz = static_cast<double>(odom.getValue(5));
    double qx = static_cast<double>(odom.getValue(6));
    double qy = static_cast<double>(odom.getValue(7));
    double qz = static_cast<double>(odom.getValue(8));
    double qw = static_cast<double>(odom.getValue(9));
    double yaw = YawFromQuaternion(qx, qy, qz, qw);

    if (imu_ != nullptr)
    {
        sf::Sample imu = imu_->getLastSample();
        if (imu.getNumOfDimensions() >= 3)
        {
            // Euler yaw from the IMU is undefined at pitch = ±90°, and a bad
            // sample must not be allowed to push NaN into the thrusters.
            double imuYaw = static_cast<double>(imu.getValue(2));
            if (std::isfinite(imuYaw))
                yaw = imuYaw;
        }
    }

    double z = zOdom;
    if (pressure_ != nullptr)
    {
        sf::Sample pressure = pressure_->getLastSample();
        if (pressure.getNumOfDimensions() >= 1)
        {
            double depth = static_cast<double>(pressure.getValue(0)) / (kWaterDensity * kGravity);
            if (depth > 0.05)
                z = depth;
        }
    }

    trackLength_ += std::sqrt(vx * vx + vy * vy) * dt;
    AccumulateMultibeam(x, y, yaw);

    double surgeForce = 0.0;
    double yawForce = 0.0;
    double obstDist = 0.0, obstAngle = 0.0;
    FillProfiler(obstDist, obstAngle);
    bool obstacle = std::isfinite(obstDist) && obstDist < safetyDistance_;

    if (!autopilot_ || missionDone_)
    {
        surgeForce = 0.0;
        yawForce = 0.0;
    }
    else
    {
        if (obstacle)
            minClearance_ = std::min(minClearance_, obstDist);

        if (obstacle)
        {
            double closeness = std::clamp(1.0 - obstDist / safetyDistance_, 0.0, 1.0);
            // Bearing in [-1, 1]. A return dead ahead gets a port bias so the
            // bow does not oscillate about zero and drive into the contact.
            double bearing = std::clamp(obstAngle / (50.0 * M_PI / 180.0), -1.0, 1.0);
            if (std::fabs(obstAngle) < 0.12 && closeness > 0.35)
                bearing = -1.0;

            avoidYawHold_ = -avoidYawForce_ * bearing * (0.35 + 0.65 * closeness);
            avoidMemory_ = 0.55;
            surgeForce = cruiseForce_ * (1.0 - 1.25 * closeness);
            yawForce = avoidYawHold_;
        }
        else if (avoidMemory_ > 0.0)
        {
            avoidMemory_ = std::max(0.0, avoidMemory_ - dt);
            double fade = avoidMemory_ / 0.55;
            surgeForce = cruiseForce_ * (0.35 + 0.4 * (1.0 - fade));
            yawForce = avoidYawHold_ * fade;
        }
        else
        {
            const SurveyWaypoint& wp = waypoints_[currentWaypoint_];
            double dx = wp.x - x;
            double dy = wp.y - y;
            double distToWp = std::sqrt(dx * dx + dy * dy);

            if (distToWp < waypointRadius_)
            {
                if (currentWaypoint_ + 1 >= waypoints_.size())
                    missionDone_ = true;
                else
                    ++currentWaypoint_;
            }

            double desiredHeading = std::atan2(dy, dx);
            double headingError = WrapAngle(desiredHeading - yaw);
            if (!std::isfinite(headingError))
                headingError = 0.0;
            yawForce = yawGain_ * std::clamp(headingError / (M_PI / 3.0), -1.0, 1.0);

            double headingFactor = std::clamp(1.0 - std::fabs(headingError) / (M_PI / 2.0), 0.15, 1.0);
            surgeForce = cruiseForce_ * headingFactor;
        }
    }

    // Depth: positive heave force points down. DVL bottom-lock closer than
    // 2.5 m overrides the setpoint and climbs.
    double needDown = targetDepth_ - z;
    if (dvl_ != nullptr)
    {
        sf::Sample dvl = dvl_->getLastSample();
        if (dvl.getNumOfDimensions() >= 8)
        {
            double altitude = static_cast<double>(dvl.getValue(3));
            double status = static_cast<double>(dvl.getValue(7));
            // Ignore sub-metre returns: those are the hull, not the seabed.
            bool bottomLock = status < 2.5 && altitude > 1.2 && altitude < 40.0;
            if (bottomLock && altitude < 2.5)
                needDown = -1.2;
        }
    }

    double heaveForce = 0.0;
    if (!std::isfinite(needDown) || !std::isfinite(vz))
        heaveForce = 0.0;
    else
        heaveForce = depthGain_ * needDown - depthDamping_ * vz;
    heaveForce = std::clamp(heaveForce, -heaveLimit_, heaveLimit_);

    // External policy: same three forces, no survey law and no depth hold.
    if (!autopilot_)
    {
        surgeForce = action_.surgeForce;
        yawForce = action_.yawForce;
        heaveForce = std::clamp(action_.heaveForce, -heaveLimit_, heaveLimit_);
    }

    if (!std::isfinite(surgeForce))
        surgeForce = 0.0;
    if (!std::isfinite(yawForce))
        yawForce = 0.0;
    mainThruster_->setSetpoint(static_cast<sf::Scalar>(surgeForce), 0.0);
    yawThruster_->setForce(static_cast<sf::Scalar>(yawForce));
    if (heaveThruster_ != nullptr)
        heaveThruster_->setForce(static_cast<sf::Scalar>(heaveForce));

    double wpDist = 0.0;
    double headingErr = 0.0;
    if (!waypoints_.empty() && currentWaypoint_ < waypoints_.size())
    {
        double dx = waypoints_[currentWaypoint_].x - x;
        double dy = waypoints_[currentWaypoint_].y - y;
        wpDist = std::sqrt(dx * dx + dy * dy);
        headingErr = WrapAngle(std::atan2(dy, dx) - yaw);
    }
    double altitude = -1.0;
    if (dvl_ != nullptr)
    {
        sf::Sample dvl = dvl_->getLastSample();
        if (dvl.getNumOfDimensions() >= 8)
        {
            double alt = static_cast<double>(dvl.getValue(3));
            double status = static_cast<double>(dvl.getValue(7));
            if (status < 2.5 && alt > 1.2 && alt < 40.0)
                altitude = alt;
        }
    }
    observation_.x = x;
    observation_.y = y;
    observation_.z = z;
    observation_.yaw = yaw;
    observation_.vx = vx;
    observation_.vy = vy;
    observation_.vz = vz;
    observation_.depthError = targetDepth_ - z;
    observation_.altitude = altitude;
    observation_.obstacleDist = std::isfinite(obstDist) ? obstDist : 1e6;
    observation_.obstacleAngle = obstAngle;
    observation_.waypointDist = wpDist;
    observation_.headingError = std::isfinite(headingErr) ? headingErr : 0.0;
    observation_.coverageCells = static_cast<int>(coverage_.size());
    observation_.missionDone = missionDone_;

    if (missionDone_ && !logWritten_)
    {
        logWritten_ = true;
        WriteLog(x, y, z);
    }

    statusTimer_ += dt;
    if (statusTimer_ >= 5.0)
    {
        statusTimer_ = 0.0;
        cInfo("%s t=%.0f xyz=%.1f %.1f %.2f (target %.1f) wp %zu/%zu coverage %zu %s",
              robotName_.c_str(), simTime_, x, y, z, targetDepth_,
              currentWaypoint_, waypoints_.size(), coverage_.size(),
              missionDone_ ? "DONE" : "surveying");
    }
}
