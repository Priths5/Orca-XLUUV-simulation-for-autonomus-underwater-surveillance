/*
    main_multisub.cpp
    Scenario 2: the Orca XLUUV surveys the seabed alongside two other,
    independently-moving survey submarines (resources/
    orca_multisub_world.scn). All three vehicles run the same
    OrcaController: their own lawnmower survey lane, plus reactive
    avoidance driven purely by each vehicle's own Profiler sensor - so
    when the Orca has to dodge another sub, it's because it actually
    detected it, not because of scripted choreography.

    Builds an executable called orca_multisub_sim (see CMakeLists.txt).
*/

#include <core/GraphicalSimulationApp.h>
#include <core/SimulationManager.h>
#include <core/ScenarioParser.h>
#include <graphics/OpenGLTrackball.h>
#include <utils/SystemUtil.hpp>
#include <glm/vec3.hpp>

#include "OrcaController.hpp"

#include <memory>
#include <vector>

class OrcaMultiSubSimulationManager : public sf::SimulationManager
{
public:
    explicit OrcaMultiSubSimulationManager(sf::Scalar stepsPerSecond)
        : sf::SimulationManager(stepsPerSecond)
    {}

    void BuildScenario() override
    {
        sf::ScenarioParser parser(this);
        bool success = parser.Parse(sf::GetDataPath() + "orca_multisub_world.scn");
        parser.SaveLog(sf::GetDataPath() + "orca_multisub_world_parse.log");
        if (!success)
        {
            cCritical("Failed to parse orca_multisub_world.scn - see orca_multisub_world_parse.log next to it for the exact line/tag that failed.");
        }

        // Orca: lane 1, northern half of the shared box, starting near
        // the SW corner (matches start_x/start_y in the .scn include).
        controllers_.push_back(std::make_unique<OrcaController>(
            this, "OrcaXLUUV",
            BuildLawnmowerPattern(-50.0, -15.0, 50.0, 25.0, 15.0),
            /*targetDepth*/ 12.0, 16000.0, 5500.0, 10.0, 6000.0, 5.0));

        // SurveySubAlpha: lane 2, sweeping the opposite diagonal so its
        // track crosses the Orca's lane near the middle of the box.
        controllers_.push_back(std::make_unique<OrcaController>(
            this, "SurveySubAlpha",
            BuildLawnmowerPattern(18.0, 12.0, 50.0, 28.0, 10.0),
            /*targetDepth*/ 13.0, 2800.0, 1400.0, 9.0, 1400.0, 5.0,
            /*depthGain*/ 2200.0, /*depthDamping*/ 7000.0, /*heaveLimit*/ 3000.0));

        // SurveySubBravo: lane 3, a slower cross-cut through the same
        // box at a shallower depth, timed differently again.
        controllers_.push_back(std::make_unique<OrcaController>(
            this, "SurveySubBravo",
            BuildLawnmowerPattern(-20.0, -45.0, 20.0, 45.0, 10.0),
            /*targetDepth*/ 10.5, 2800.0, 1400.0, 9.0, 1400.0, 5.0,
            /*depthGain*/ 2200.0, /*depthDamping*/ 7000.0, /*heaveLimit*/ 3000.0));

        if (getTrackball() != nullptr)
            getTrackball()->MoveCenter(glm::vec3(-50.f, -10.f, 13.f));
    }

    void SimulationStepCompleted(sf::Scalar timeStep) override
    {
        for (auto& c : controllers_)
            c->Update(static_cast<double>(timeStep));
    }

private:
    std::vector<std::unique_ptr<OrcaController>> controllers_;
};

int main(int argc, char** argv)
{
    sf::RenderSettings s;
    s.windowW = 1280;
    s.windowH = 720;
    s.shadows = sf::RenderQuality::HIGH;
    s.ao = sf::RenderQuality::HIGH;
    s.atmosphere = sf::RenderQuality::MEDIUM;
    s.ocean = sf::RenderQuality::HIGH;
    s.aa = sf::RenderQuality::HIGH;
    s.ssr = sf::RenderQuality::HIGH;
    s.verticalSync = true;

    sf::HelperSettings h;
    h.showFluidDynamics = false;
    h.showCoordSys = false;
    h.showBulletDebugInfo = false;
    h.showSensors = false;
    h.showActuators = false;
    h.showForces = false;

    OrcaMultiSubSimulationManager manager(300.0);
    manager.setCallSimulationStepCompleted(true);

    sf::GraphicalSimulationApp app("Orca XLUUV vs Survey Traffic", "./resources/", s, h, &manager);
    app.Run();

    return 0;
}
