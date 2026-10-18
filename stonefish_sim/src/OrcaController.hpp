/*
    OrcaController.hpp

    Waypoint survey plus reactive avoidance, shared by orca_survey_sim
    and orca_multisub_sim.

    Navigation uses the sensors that are actually mounted:
      - IMU yaw for heading (odometry yaw if the IMU is absent)
      - pressure for depth, held by HeaveThruster
      - DVL altitude as a floor guard so the hull cannot settle onto the seabed
      - odometry for horizontal position (there is no absolute underwater fix)
      - Profiler for avoidance
      - Multibeam for a coverage grid written when the lawnmower finishes

    Main propulsion is a SimpleThruster (visible propeller). Yaw and heave
    are Push actuators.
*/

#ifndef ORCA_CONTROLLER_HPP
#define ORCA_CONTROLLER_HPP

#include <core/Robot.h>
#include <actuators/Push.h>
#include <actuators/SimpleThruster.h>
#include <sensors/ScalarSensor.h>
#include <sensors/Sample.h>

#include <array>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Fixed-size packet a later training loop can read after every Update().
// Profiler bins cover the forward +/-80 deg sweep, left to right.
// altitude is negative when the DVL has no bottom lock.
// obstacleDist is 1e6 when the sweep is clear inside the safety range.
static constexpr int kProfilerBins = 16;

struct OrcaObservation
{
    double x = 0, y = 0, z = 0;
    double yaw = 0;
    double vx = 0, vy = 0, vz = 0;
    double depthError = 0;
    double altitude = -1;
    double obstacleDist = 1e6;
    double obstacleAngle = 0;
    double waypointDist = 0;
    double headingError = 0;
    std::array<double, kProfilerBins> profiler{};
    int coverageCells = 0;
    bool missionDone = false;
};

// Forces in newtons applied to the same actuators the autopilot uses.
// Positive surge is forward. Positive yaw force is the bow lateral push
// (starboard). Positive heave is downward.
struct OrcaAction
{
    double surgeForce = 0;
    double yawForce = 0;
    double heaveForce = 0;
};

struct SurveyWaypoint
{
    double x;
    double y;
};

std::vector<SurveyWaypoint> BuildLawnmowerPattern(double x0, double y0,
                                                   double x1, double y1,
                                                   double laneSpacing);

class OrcaController
{
public:
    OrcaController(sf::SimulationManager* manager,
                    const std::string& robotName,
                    std::vector<SurveyWaypoint> waypoints,
                    double targetDepth,
                    double cruiseForce = 18000.0,
                    double yawGain = 6000.0,
                    double safetyDistance = 8.0,
                    double avoidYawForce = 6000.0,
                    double waypointRadius = 4.0,
                    double depthGain = 28000.0,
                    double depthDamping = 90000.0,
                    double heaveLimit = 55000.0);

    bool TryBind();
    void Update(double dt);

    // Autopilot is the survey law. Turn it off and SetAction() drives the
    // three thrusters directly; sensing, coverage, and observation() keep updating.
    void SetAutopilot(bool enabled) { autopilot_ = enabled; }
    bool autopilot() const { return autopilot_; }
    void SetAction(const OrcaAction& action) { action_ = action; }
    const OrcaObservation& observation() const { return observation_; }

    bool isBound() const { return bound; }
    bool missionComplete() const { return missionDone_; }
    const std::string& name() const { return robotName_; }

private:
    void AccumulateMultibeam(double x, double y, double yaw);
    void WriteLog(double x, double y, double z) const;
    void FillProfiler(double& minDist, double& minAngle);

    sf::SimulationManager* manager_;
    std::string robotName_;
    std::vector<SurveyWaypoint> waypoints_;
    size_t currentWaypoint_;
    double targetDepth_;

    double cruiseForce_;
    double yawGain_;
    double safetyDistance_;
    double avoidYawForce_;
    double waypointRadius_;
    double depthGain_;
    double depthDamping_;
    double heaveLimit_;

    bool bound;
    bool missionDone_;
    bool logWritten_;
    bool autopilot_;
    OrcaAction action_;
    OrcaObservation observation_;
    sf::Robot* robot_;
    sf::SimpleThruster* mainThruster_;
    sf::Push* yawThruster_;
    sf::Push* heaveThruster_;
    sf::ScalarSensor* odometry_;
    sf::ScalarSensor* profiler_;
    sf::ScalarSensor* pressure_;
    sf::ScalarSensor* imu_;
    sf::ScalarSensor* dvl_;
    sf::ScalarSensor* multibeam_;

    double avoidMemory_;
    double avoidYawHold_;
    double simTime_;
    double statusTimer_;
    double trackLength_;
    double minClearance_;
    double lastMultibeamStamp_;
    // 4 m cells of seabed the multibeam actually hit.
    std::set<std::pair<int, int>> coverage_;

    static constexpr double kWaterDensity = 1025.0;
    static constexpr double kGravity = 9.81;
    static constexpr double kMultibeamFovRad = 120.0 * 3.14159265358979323846 / 180.0;
    static constexpr double kCellSize = 4.0;
};

#endif // ORCA_CONTROLLER_HPP
