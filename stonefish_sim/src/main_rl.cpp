/*
    main_rl.cpp  ->  executable "orca_rl_sim"

    ROS 2-aware Stonefish application. The existing Orca scenarios are
    loaded unchanged through ScenarioParser; this file only adds:

        ROS 2 /orca/cmd/setpoint ----> OrcaRLCommandAdapter (300 Hz inner loop)
        ROS 2 /orca/cmd/thrust   ---\         |
                                     +--> OrcaController (autopilot OFF = RL mode)
                                                |
                                     MainThruster / YawThruster / HeaveThruster
                                                |
                                         Stonefish physics
                                                |
        ROS 2 /orca/imu, /orca/dvl/*, /orca/pressure, /orca/gps,
              /orca/profiler, /orca/multibeam, /orca/contact,
              /orca/ground_truth/odometry (debug only), /clock

    Threading: Stonefish owns the main thread (rendering, or an idle loop in
    --headless mode) and runs physics on its own simulation thread, where
    SimulationStepCompleted() is called. The ROS 2 executor spins on a
    third std::thread. They share data only through OrcaStonefishRos's
    mutex-protected slots.

    Usage:
        ./orca_rl_sim [--world rl|survey|multisub] [--mode rl|autopilot]
                      [--headless] [--rtf <realtime factor>] [--data <dir>]
                      [--ros-args ...]
*/

// ROS 2 headers first: Stonefish's StonefishCommon.h defines a Max(a,b)
// macro that breaks libstatistics_collector if it is seen earlier.
#include "OrcaStonefishRos.hpp"
#include <rclcpp/rclcpp.hpp>

#include <core/ConsoleSimulationApp.h>
#include <core/GraphicalSimulationApp.h>
#include <core/Robot.h>
#include <core/ScenarioParser.h>
#include <core/SimulationManager.h>
#include <entities/Entity.h>
#include <graphics/OpenGLTrackball.h>
#include <sensors/ScalarSensor.h>
#include <sensors/Sample.h>
#include <actuators/Push.h>
#include <actuators/SimpleThruster.h>
#include <utils/SystemUtil.hpp>
#include <btBulletDynamicsCommon.h>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstdlib>

#include "OrcaController.hpp"
#include "OrcaRLCommandAdapter.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

struct Options
{
    std::string world = "rl";
    bool autopilot = false;
    bool headless = false;
    bool follow = false;     // camera glued to the Orca (for watching / screenshots)
    double rtf = 1.0;
    std::string dataPath;
};

struct WorldSpec
{
    std::string file;
    std::vector<SurveyWaypoint> orcaWaypoints;
    double targetDepth;
    glm::vec3 cameraCenter;
    double maxDepth;   // hard floor for the depth reference (seabed minus margin)
};

WorldSpec GetWorldSpec(const std::string& world)
{
    if (world == "survey")
        return {"orca_survey_world.scn", BuildLawnmowerPattern(-45.0, -35.0, 45.0, 35.0, 12.0),
                12.0, glm::vec3(-45.f, -35.f, 13.f), 15.0};
    if (world == "multisub")
        return {"orca_multisub_world.scn", BuildLawnmowerPattern(-50.0, -15.0, 50.0, 25.0, 15.0),
                12.0, glm::vec3(-50.f, -10.f, 13.f), 15.0};
    // "rl": open-ocean base world with an obstacle field + traffic sub
    return {"orca_rl_world.scn", BuildLawnmowerPattern(-80.0, -60.0, 80.0, 60.0, 30.0),
            12.0, glm::vec3(-80.f, -60.f, 13.f), 40.0};
}

std::string ExecutableDir()
{
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0)
        return ".";
    buf[n] = '\0';
    return std::filesystem::path(buf).parent_path().string();
}

// Stonefish refuses <light> in console (headless) mode, and every Orca
// model carries bow/down lights. Headless runs drop them after the normal
// include/argument pre-processing; nothing physical changes.
class HeadlessScenarioParser : public sf::ScenarioParser
{
public:
    HeadlessScenarioParser(sf::SimulationManager* sm, bool stripLights)
        : sf::ScenarioParser(sm), stripLights_(stripLights) {}
protected:
    bool PreProcess(XMLNode* root, const std::map<std::string, std::string>& args) override
    {
        if (!sf::ScenarioParser::PreProcess(root, args))
            return false;
        if (stripLights_)
            StripLights(root);
        return true;
    }
private:
    static void StripLights(XMLNode* node)
    {
        XMLNode* child = node->FirstChild();
        while (child != nullptr)
        {
            XMLNode* next = child->NextSibling();
            XMLElement* el = child->ToElement();
            if (el != nullptr && std::strcmp(el->Name(), "light") == 0)
                node->DeleteChild(child);
            else
                StripLights(child);
            child = next;
        }
    }
    bool stripLights_;
};

} // namespace

// ---------------------------------------------------------------------------
class OrcaRLSimulationManager : public sf::SimulationManager
{
public:
    OrcaRLSimulationManager(sf::Scalar stepsPerSecond, const Options& opt,
                            std::shared_ptr<OrcaStonefishRos> ros)
        : sf::SimulationManager(stepsPerSecond), opt_(opt), ros_(std::move(ros)),
          spec_(GetWorldSpec(opt.world))
    {}

    void BuildScenario() override
    {
        HeadlessScenarioParser parser(this, opt_.headless);
        std::string path = sf::GetDataPath() + spec_.file;
        bool ok = parser.Parse(path);
        parser.SaveLog(sf::GetDataPath() + spec_.file + "_parse.log");
        if (!ok)
            cCritical("Failed to parse %s - see the _parse.log next to it.", path.c_str());

        // The Orca is driven by the existing OrcaController; in RL mode its
        // autopilot is switched off and it applies the forces we hand it.
        controllers_.clear();
        controllers_.push_back(std::make_unique<OrcaController>(
            this, kRobot, spec_.orcaWaypoints, spec_.targetDepth,
            18000.0, 6000.0, 8.0, 6500.0, 5.0));
        orca_ = controllers_.front().get();
        orca_->SetAutopilot(opt_.autopilot);

        // Other traffic keeps its own scripted survey + reactive avoidance.
        if (opt_.world == "multisub")
        {
            controllers_.push_back(std::make_unique<OrcaController>(
                this, "SurveySubAlpha", BuildLawnmowerPattern(18.0, 12.0, 50.0, 28.0, 10.0),
                13.0, 2800.0, 1400.0, 9.0, 1400.0, 5.0, 2200.0, 7000.0, 3000.0));
            controllers_.push_back(std::make_unique<OrcaController>(
                this, "SurveySubBravo", BuildLawnmowerPattern(-20.0, -45.0, 20.0, 45.0, 10.0),
                10.5, 2800.0, 1400.0, 9.0, 1400.0, 5.0, 2200.0, 7000.0, 3000.0));
        }
        else if (opt_.world == "rl")
        {
            // North-south legs, so it crosses the Orca's east-west survey lanes.
            std::vector<SurveyWaypoint> legs;
            for (int k = 0; k < 6; ++k)
                for (SurveyWaypoint wp : {SurveyWaypoint{-40.0, -50.0}, SurveyWaypoint{-40.0, 50.0},
                                          SurveyWaypoint{25.0, 50.0}, SurveyWaypoint{25.0, -50.0}})
                    legs.push_back(wp);
            controllers_.push_back(std::make_unique<OrcaController>(
                this, "TrafficSub", std::move(legs),
                11.0, 2800.0, 1400.0, 9.0, 1400.0, 6.0, 2200.0, 7000.0, 3000.0));
        }

        OrcaRLAdapterParams ap;
        ap.maxDepth = spec_.maxDepth;
        adapter_ = OrcaRLCommandAdapter(ap);
        bound_ = false;

        if (getTrackball() != nullptr)
        {
            getTrackball()->MoveCenter(spec_.cameraCenter);
            sf::Robot* orca = getRobot(kRobot);
            if (opt_.follow && orca != nullptr)
            {
                // Chase view: glue the trackball to the hull, tilt it to look down
                // on the Orca and zoom out. Tunable without rebuilding:
                // ORCA_CAM_DEG (tilt, default 35), ORCA_CAM_YAW (default 0),
                // ORCA_CAM_ZOOM (scroll steps, default 6).
                auto env = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };
                getTrackball()->GlueToMoving(orca->getBaseLink());
                getTrackball()->Rotate(glm::angleAxis(glm::radians(static_cast<float>(env("ORCA_CAM_YAW", 0.0))),
                                                      glm::vec3(0.f, 0.f, 1.f)));
                getTrackball()->Rotate(glm::angleAxis(glm::radians(static_cast<float>(env("ORCA_CAM_DEG", 35.0))),
                                                      glm::vec3(1.f, 0.f, 0.f)));
                getTrackball()->MouseScroll(static_cast<GLfloat>(env("ORCA_CAM_ZOOM", 6.0)));
            }
        }
    }

    void SimulationStepCompleted(sf::Scalar timeStep) override
    {
        const double dt = static_cast<double>(timeStep);
        const double t = static_cast<double>(getSimulationTime());
        ros_->SetSimTime(t);

        if (!bound_ && !Bind())
            return;

        if (auto ap = ros_->TakeAutopilotRequest())
        {
            autopilot_ = *ap;
            orca_->SetAutopilot(autopilot_);
            ros_->PublishStatus(StatusText());
        }
        if (ros_->TakeResetRequest())
            DoReset();

        // ------------- read the real Stonefish sensors -------------
        ReadSensors();
        if (CheckPhysicsDivergence(t))
            return;

        // ------------- command -> actuator forces -------------
        OrcaCommand cmd = ros_->LatestCommand();
        bool stale = cmd.kind == OrcaCommandKind::NONE ||
                     (t - cmd.receivedSimTime) > ros_->commandTimeout();
        OrcaRLSetpoint sp;            // zero speed, zero yaw rate, hold depth
        OrcaRLForces forces;
        int source = 0;
        if (autopilot_)
        {
            source = 3;
        }
        else if (!stale && cmd.kind == OrcaCommandKind::THRUST)
        {
            forces = cmd.thrust;
            source = 2;
        }
        else
        {
            if (!stale)
            {
                sp = cmd.setpoint;
                source = 1;
            }
            forces = adapter_.Compute(sp, feedback_, dt);
        }
        if (!autopilot_)
            orca_->SetAction({forces.surge, forces.yaw, forces.heave});

        for (auto& c : controllers_)
            c->Update(dt);

        // ------------- publish -------------
        if (t - lastClock_ >= 0.01) { lastClock_ = t; ros_->PublishClock(t); }
        PublishSensors(t);
        if (t - lastActuators_ >= 0.05)
        {
            lastActuators_ = t;
            double actual = mainThruster_ ? static_cast<double>(mainThruster_->getThrust()) : 0.0;
            OrcaRLForces applied = forces;
            if (autopilot_)
            {
                applied.surge = mainThruster_ ? static_cast<double>(mainThruster_->getThrustSetpoint()) : 0.0;
                applied.yaw = yawThruster_ ? static_cast<double>(yawThruster_->getForce()) : 0.0;
                applied.heave = heaveThruster_ ? static_cast<double>(heaveThruster_->getForce()) : 0.0;
            }
            ros_->PublishActuators(t, applied, actual, sp, adapter_.depthReference(), source);
        }
        if (t - lastContact_ >= 0.1)
        {
            lastContact_ = t;
            std::string other;
            bool hit = DetectContact(other);
            if (hit && !inContact_)
                cWarning("CONTACT: %s touched %s at t=%.1f", kRobot, other.c_str(), t);
            inContact_ = hit;
            ros_->PublishContact(t, hit, other);
        }
        if (t - lastStatus_ >= 2.0)
        {
            lastStatus_ = t;
            cInfo("orca_rl_sim t=%.0f src=%d xyz=%.1f %.1f %.1f rpy=%.1f %.1f %.1f deg u=%.2f r=%.3f alt=%.1f F=[%.0f %.0f %.0f]",
                  t, source, gt_.x, gt_.y, gt_.z, imuData_.roll * 57.2958, imuData_.pitch * 57.2958,
                  imuData_.yaw * 57.2958, feedback_.surgeSpeed, feedback_.yawRate, feedback_.altitude,
                  forces.surge, forces.yaw, forces.heave);
            ros_->PublishStatus(StatusText());
        }
    }

private:
    static constexpr const char* kRobot = "OrcaXLUUV";

    std::string StatusText() const
    {
        return std::string("world=") + opt_.world + " mode=" + (autopilot_ ? "autopilot" : "rl");
    }

    bool Bind()
    {
        robot_ = getRobot(kRobot);
        if (robot_ == nullptr)
            return false;
        auto act = [&](const char* n) { return robot_->getActuator(std::string(kRobot) + "/" + n); };
        auto sens = [&](const char* n) {
            return dynamic_cast<sf::ScalarSensor*>(robot_->getSensor(std::string(kRobot) + "/" + n));
        };
        mainThruster_ = dynamic_cast<sf::SimpleThruster*>(act("MainThruster"));
        yawThruster_ = dynamic_cast<sf::Push*>(act("YawThruster"));
        heaveThruster_ = dynamic_cast<sf::Push*>(act("HeaveThruster"));
        imu_ = sens("IMU");
        dvl_ = sens("DVL");
        pressure_ = sens("Pressure");
        gps_ = sens("GPS");
        odom_ = sens("Odometry");
        profiler_ = sens("Profiler");
        multibeam_ = sens("Multibeam");
        if (!mainThruster_ || !yawThruster_ || !heaveThruster_ || !odom_ || !imu_ || !pressure_ || !profiler_)
        {
            cCritical("orca_rl_sim: %s is missing MainThruster/YawThruster/HeaveThruster or "
                      "IMU/Pressure/Odometry/Profiler - use data/orca_xluuv_survey.scn", kRobot);
            return false;
        }
        startPose_ = robot_->getTransform();
        if (opt_.follow && sf::SimulationApp::getApp()->hasGraphics())
            static_cast<sf::GraphicalSimulationApp*>(sf::SimulationApp::getApp())->HideHUD();  // clean view
        autopilot_ = opt_.autopilot;
        bound_ = true;
        cInfo("orca_rl_sim bound to %s (%s), DVL=%d GPS=%d Multibeam=%d",
              kRobot, StatusText().c_str(), dvl_ != nullptr, gps_ != nullptr, multibeam_ != nullptr);
        ros_->PublishStatus(StatusText());
        return true;
    }

    // Bullet can blow up when a scripted (kinematic, infinite-mass) fish
    // squeezes the hull against the seabed: the Orca is then flung away at
    // tens of m/s. Detect that, put the Orca back at its last healthy pose
    // (~2 s earlier, at rest) and report it, instead of silently continuing.
    bool CheckPhysicsDivergence(double t)
    {
        // After a respawn the sensors (Odometry at 20 Hz) still hold pre-reset
        // samples for a moment; give them time to refresh before re-checking.
        if (t < divergenceCooldownUntil_)
            return false;
        double speed = std::sqrt(gt_.u * gt_.u + gt_.v * gt_.v + gt_.w * gt_.w);
        bool finite = std::isfinite(gt_.x) && std::isfinite(gt_.y) && std::isfinite(gt_.z) && std::isfinite(speed);
        bool sane = finite && speed < 10.0 && std::fabs(gt_.x) < 2000.0 && std::fabs(gt_.y) < 2000.0 &&
                    gt_.z > -5.0 && gt_.z < 2000.0;
        if (sane)
        {
            if (t - lastGoodStamp_ >= 0.5)
            {
                lastGoodStamp_ = t;
                goodPoses_.push_back(robot_->getTransform());
                if (goodPoses_.size() > 5)
                    goodPoses_.erase(goodPoses_.begin());
            }
            return false;
        }
        sf::Transform back = goodPoses_.empty() ? startPose_ : goodPoses_.front();
        robot_->Respawn(this, back);
        adapter_.Reset(static_cast<double>(back.getOrigin().z()));
        goodPoses_.assign(1, back);
        divergenceCooldownUntil_ = t + 0.3;
        lastGoodStamp_ = t + 0.3;
        ++physicsResets_;
        cWarning("PHYSICS DIVERGENCE #%d at t=%.1f (speed %.1f m/s, xyz %.1f %.1f %.1f): %s respawned at its "
                 "pose ~2 s earlier", physicsResets_, t, speed, gt_.x, gt_.y, gt_.z, kRobot);
        ros_->PublishStatus(StatusText() + " physics_reset=" + std::to_string(physicsResets_));
        return true;
    }

    void DoReset()
    {
        robot_->Respawn(this, startPose_);
        adapter_.Reset(static_cast<double>(startPose_.getOrigin().z()));
        cInfo("orca_rl_sim: %s respawned at start pose (%.1f, %.1f, %.1f)", kRobot,
              startPose_.getOrigin().x(), startPose_.getOrigin().y(), startPose_.getOrigin().z());
    }

    // Rotation taking vectors from a sensor frame into the body FRD frame.
    sf::Matrix3 SensorToBody(sf::ScalarSensor* s) const
    {
        return robot_->getTransform().getBasis().transpose() * s->getSensorFrame().getBasis();
    }

    void ReadSensors()
    {
        sf::Sample o = odom_->getLastSample();
        gt_ = {o.getValue(0), o.getValue(1), o.getValue(2), o.getValue(6), o.getValue(7), o.getValue(8),
               o.getValue(9), o.getValue(3), o.getValue(4), o.getValue(5), o.getValue(10),
               o.getValue(11), o.getValue(12)};

        sf::Sample im = imu_->getLastSample();
        imuData_ = {im.getValue(0), im.getValue(1), im.getValue(2), im.getValue(3), im.getValue(4),
                    im.getValue(5), im.getValue(6), im.getValue(7), im.getValue(8)};

        sf::Sample pr = pressure_->getLastSample();
        gaugePa_ = pr.getValue(0);

        dvlData_ = {0, 0, 0, -1, false};
        if (dvl_ != nullptr)
        {
            sf::Sample d = dvl_->getLastSample();
            if (d.getNumOfDimensions() >= 8)
            {
                bool lock = d.getValue(7) < 0.5 && d.getValue(3) > 1.2;   // status 0 = bottom lock
                sf::Vector3 vb = SensorToBody(dvl_) * sf::Vector3(d.getValue(0), d.getValue(1), d.getValue(2));
                dvlData_ = {vb.x(), vb.y(), vb.z(), d.getValue(3), lock};
            }
        }

        // Inner-loop feedback. Surge speed: DVL when locked, otherwise the
        // vehicle's own body-frame velocity (Odometry) - see README.
        feedback_.surgeSpeed = dvlData_.bottomLock ? dvlData_.u : gt_.u;
        feedback_.yawRate = imuData_.wz;
        feedback_.depth = gaugePa_ / (1025.0 * 9.81);
        feedback_.heaveSpeed = gt_.w;
        feedback_.altitude = dvlData_.bottomLock ? dvlData_.altitude : -1.0;
    }

    void PublishSensors(double t)
    {
        // ScalarSensor::getLastSample() returns a copy whose getId() is always 0,
        // so new data is detected by its simulation timestamp instead.
        auto fresh = [](sf::ScalarSensor* s, double& last) {
            if (s == nullptr) return false;
            double ts = static_cast<double>(s->getLastSample().getTimestamp());
            if (ts < 0.0 || ts == last) return false;
            last = ts;
            return true;
        };
        if (fresh(imu_, idImu_)) ros_->PublishImu(t, imuData_);
        if (fresh(pressure_, idPressure_)) ros_->PublishPressure(t, gaugePa_);
        if (fresh(dvl_, idDvl_)) ros_->PublishDvl(t, dvlData_);
        if (fresh(odom_, idOdom_)) ros_->PublishGroundTruth(t, gt_);
        if (fresh(gps_, idGps_))
        {
            sf::Sample g = gps_->getLastSample();
            bool fix = g.getValue(0) < 1e30;
            ros_->PublishGps(t, {g.getValue(0), g.getValue(1), g.getValue(2), g.getValue(3), fix});
        }
        if (t - lastProfiler_ >= 0.1) { lastProfiler_ = t; PublishProfiler(t); }
        if (multibeam_ != nullptr && fresh(multibeam_, idMultibeam_))
        {
            sf::Sample m = multibeam_->getLastSample();
            size_t n = m.getNumOfDimensions();
            if (n >= 2)
            {
                const double fov = 120.0 * M_PI / 180.0;   // <specs fov="120.0"/>
                OrcaScanData s{-0.5 * fov, fov / static_cast<double>(n - 1), 0.3, 80.0, {}};
                s.ranges.resize(n);
                for (size_t i = 0; i < n; ++i)
                    s.ranges[i] = static_cast<float>(m.getValue(i));
                ros_->PublishMultibeam(t, s);
            }
        }
    }

    // The Profiler is ONE mechanically swept beam (one sample per update).
    // Assemble the newest return per angle bin from its history buffer.
    void PublishProfiler(double t)
    {
        const std::vector<sf::Sample>* hist = profiler_->getHistory();   // heap copy, caller deletes
        if (hist == nullptr)
            return;
        const int bins = 33;                                  // 5 deg bins across the 160 deg fan
        const double fov = 160.0 * M_PI / 180.0;              // <specs fov="160.0"/>
        const double rangeMax = 30.0;                         // <range distance_max="30.0"/>
        OrcaScanData s{-0.5 * fov, fov / (bins - 1), 0.3, rangeMax,
                       std::vector<float>(bins, std::numeric_limits<float>::quiet_NaN())};
        for (const sf::Sample& smp : *hist)                   // oldest -> newest
        {
            double a = smp.getValue(0);
            double d = smp.getValue(1);
            int b = static_cast<int>(std::lround((a - s.angleMin) / s.angleIncrement));
            if (b < 0 || b >= bins) continue;
            s.ranges[b] = d >= rangeMax - 1e-3 ? std::numeric_limits<float>::infinity()
                                               : static_cast<float>(d);
        }
        delete hist;
        ros_->PublishProfiler(t, s);
    }

    // Any Bullet contact manifold between an Orca link and another entity.
    bool DetectContact(std::string& other)
    {
        btDispatcher* disp = getDynamicsWorld()->getDispatcher();
        const std::string prefix = std::string(kRobot) + "/";
        for (int i = 0; i < disp->getNumManifolds(); ++i)
        {
            btPersistentManifold* m = disp->getManifoldByIndexInternal(i);
            bool touching = false;
            for (int j = 0; j < m->getNumContacts(); ++j)
                if (m->getContactPoint(j).getDistance() < sf::Scalar(0.02)) { touching = true; break; }
            if (!touching)
                continue;
            auto* ea = static_cast<sf::Entity*>(m->getBody0()->getUserPointer());
            auto* eb = static_cast<sf::Entity*>(m->getBody1()->getUserPointer());
            std::string na = ea ? ea->getName() : "";
            std::string nb = eb ? eb->getName() : "";
            bool aOrca = na.rfind(prefix, 0) == 0 || na == kRobot;
            bool bOrca = nb.rfind(prefix, 0) == 0 || nb == kRobot;
            if (aOrca != bOrca)
            {
                other = aOrca ? nb : na;
                return true;
            }
        }
        other.clear();
        return false;
    }

    Options opt_;
    std::shared_ptr<OrcaStonefishRos> ros_;
    WorldSpec spec_;
    std::vector<std::unique_ptr<OrcaController>> controllers_;
    OrcaController* orca_ = nullptr;
    OrcaRLCommandAdapter adapter_;
    OrcaRLFeedback feedback_;

    bool bound_ = false;
    bool autopilot_ = false;
    bool inContact_ = false;
    sf::Robot* robot_ = nullptr;
    sf::SimpleThruster* mainThruster_ = nullptr;
    sf::Push* yawThruster_ = nullptr;
    sf::Push* heaveThruster_ = nullptr;
    sf::ScalarSensor *imu_ = nullptr, *dvl_ = nullptr, *pressure_ = nullptr, *gps_ = nullptr,
                     *odom_ = nullptr, *profiler_ = nullptr, *multibeam_ = nullptr;
    sf::Transform startPose_;
    std::vector<sf::Transform> goodPoses_;
    double lastGoodStamp_ = -1.0;
    int physicsResets_ = 0;
    double divergenceCooldownUntil_ = -1.0;

    OrcaOdomData gt_{};
    OrcaImuData imuData_{};
    OrcaDvlData dvlData_{};
    double gaugePa_ = 0.0;

    double idImu_ = -1, idPressure_ = -1, idDvl_ = -1, idOdom_ = -1, idGps_ = -1, idMultibeam_ = -1;
    double lastClock_ = -1, lastActuators_ = -1, lastContact_ = -1, lastStatus_ = -1, lastProfiler_ = -1;
};

// Stonefish apps whose main loop also exits when ROS 2 shuts down (Ctrl-C).
class OrcaRLGraphicalApp : public sf::GraphicalSimulationApp
{
public:
    using sf::GraphicalSimulationApp::GraphicalSimulationApp;
protected:
    void LoopInternal() override
    {
        if (!rclcpp::ok()) { Quit(); return; }
        sf::GraphicalSimulationApp::LoopInternal();
    }
};

class OrcaRLConsoleApp : public sf::ConsoleSimulationApp
{
public:
    using sf::ConsoleSimulationApp::ConsoleSimulationApp;
protected:
    void LoopInternal() override
    {
        if (!rclcpp::ok()) { StopSimulation(); Quit(); return; }
        sf::ConsoleSimulationApp::LoopInternal();
    }
};

static Options ParseOptions(const std::vector<std::string>& args)
{
    Options o;
    for (size_t i = 1; i < args.size(); ++i)
    {
        const std::string& a = args[i];
        auto next = [&]() { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--world") o.world = next();
        else if (a == "--mode") o.autopilot = (next() == "autopilot");
        else if (a == "--headless") o.headless = true;
        else if (a == "--follow") o.follow = true;
        else if (a == "--rtf") o.rtf = std::stod(next());
        else if (a == "--data") o.dataPath = next();
        else if (a == "-h" || a == "--help")
        {
            std::cout << "orca_rl_sim [--world rl|survey|multisub] [--mode rl|autopilot] "
                         "[--headless] [--follow] [--rtf F] [--data DIR] [--ros-args ...]\n";
            std::exit(0);
        }
    }
    if (o.world != "rl" && o.world != "survey" && o.world != "multisub")
    {
        std::cerr << "unknown --world " << o.world << " (rl|survey|multisub)\n";
        std::exit(2);
    }
    if (o.dataPath.empty())
        o.dataPath = ExecutableDir() + "/resources/";
    if (o.dataPath.back() != '/')
        o.dataPath += '/';
    return o;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    Options opt = ParseOptions(rclcpp::remove_ros_arguments(argc, argv));

    auto ros = std::make_shared<OrcaStonefishRos>("OrcaXLUUV");
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(ros);
    std::thread rosThread([&exec]() { exec.spin(); });

    OrcaRLSimulationManager manager(300.0, opt, ros);
    manager.setCallSimulationStepCompleted(true);
    manager.setRealtimeFactor(opt.rtf);

    if (opt.headless)
    {
        OrcaRLConsoleApp app("Orca XLUUV RL (headless)", opt.dataPath, &manager);
        app.Run();
    }
    else
    {
        sf::RenderSettings s;
        s.windowW = 1280;
        s.windowH = 720;
        s.shadows = sf::RenderQuality::MEDIUM;
        s.ao = sf::RenderQuality::MEDIUM;
        s.atmosphere = sf::RenderQuality::MEDIUM;
        s.ocean = sf::RenderQuality::HIGH;
        s.aa = sf::RenderQuality::MEDIUM;
        s.ssr = sf::RenderQuality::MEDIUM;
        s.verticalSync = true;
        sf::HelperSettings h;
        h.showFluidDynamics = false;
        h.showCoordSys = false;
        h.showBulletDebugInfo = false;
        h.showSensors = false;
        h.showActuators = false;
        h.showForces = false;
        OrcaRLGraphicalApp app("Orca XLUUV - RL over ROS 2", opt.dataPath, s, h, &manager);
        app.Run();
    }

    exec.cancel();
    if (rosThread.joinable())
        rosThread.join();
    if (rclcpp::ok())
        rclcpp::shutdown();
    return 0;
}
