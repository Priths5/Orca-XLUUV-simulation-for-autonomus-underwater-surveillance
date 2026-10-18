/*
    main_survey.cpp
    Scenario 1: Orca XLUUV surveying the ocean floor (resources/
    orca_survey_world.scn - rocks + fish schools over a sand seabed) while
    reactively avoiding obstacles using its forward Profiler sensor. See
    src/OrcaController.hpp for the controller itself.

    Builds an executable called orca_survey_sim (see CMakeLists.txt).
*/

#include <core/GraphicalSimulationApp.h>
#include <core/SimulationManager.h>
#include <core/ScenarioParser.h>
#include <graphics/OpenGLTrackball.h>
#include <utils/SystemUtil.hpp>
#include <glm/vec3.hpp>

#include "OrcaController.hpp"

#include <memory>

class OrcaSurveySimulationManager : public sf::SimulationManager
{
public:
    explicit OrcaSurveySimulationManager(sf::Scalar stepsPerSecond)
        : sf::SimulationManager(stepsPerSecond)
    {}

    void BuildScenario() override
    {
        sf::ScenarioParser parser(this);
        bool success = parser.Parse(sf::GetDataPath() + "orca_survey_world.scn");
        parser.SaveLog(sf::GetDataPath() + "orca_survey_world_parse.log");
        if (!success)
        {
            cCritical("Failed to parse orca_survey_world.scn - see orca_survey_world_parse.log next to it for the exact line/tag that failed.");
        }

        // Survey box roughly bracketing the rock field + fish loops laid
        // out in orca_survey_world.scn (x in [-60,60], y in [-40,40]),
        // with the vehicle entering at its start pose near the corner.
        auto waypoints = BuildLawnmowerPattern(-45.0, -35.0, 45.0, 35.0, 12.0);
        controller_ = std::make_unique<OrcaController>(
            this, "OrcaXLUUV", std::move(waypoints),
            /*targetDepth*/ 12.0,
            /*cruiseForce*/ 18000.0,
            /*yawGain*/ 6000.0,
            /*safetyDistance*/ 8.0,
            /*avoidYawForce*/ 6500.0,
            /*waypointRadius*/ 5.0);

        // Open on the boat. Default camera sits at the surface looking at nothing.
        if (getTrackball() != nullptr)
            getTrackball()->MoveCenter(glm::vec3(-45.f, -35.f, 13.f));
    }

    void SimulationStepCompleted(sf::Scalar timeStep) override
    {
        if (controller_)
            controller_->Update(static_cast<double>(timeStep));
    }

private:
    std::unique_ptr<OrcaController> controller_;
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

    OrcaSurveySimulationManager manager(300.0);
    manager.setCallSimulationStepCompleted(true);

    sf::GraphicalSimulationApp app("Orca XLUUV Seabed Survey", "./resources/", s, h, &manager);
    app.Run();

    return 0;
}
