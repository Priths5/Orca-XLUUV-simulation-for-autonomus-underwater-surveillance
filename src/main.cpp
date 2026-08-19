/*
    main.cpp
    Standalone Stonefish application for the Orca XLUUV scenario.

    This is a self-contained project: it does not depend on any other
    Stonefish example project or external data folder. Everything the
    simulation needs lives under ./resources relative to this project.
*/

#include <core/GraphicalSimulationApp.h>
#include <core/SimulationManager.h>
#include <core/ScenarioParser.h>
#include <utils/SystemUtil.hpp>

// NOTE: verified against the actual Stonefish source (patrykcieslak/stonefish)
// rather than assumed - header paths, GetDataPath(), and the RenderSettings/
// HelperSettings field names below all match Library/include/core and
// Library/include/graphics/OpenGLDataStructs.h in that repo.

class OrcaSimulationManager : public sf::SimulationManager
{
public:
    explicit OrcaSimulationManager(sf::Scalar stepsPerSecond)
        : sf::SimulationManager(stepsPerSecond)
    {}

    void BuildScenario() override
    {
        sf::ScenarioParser parser(this);
        bool success = parser.Parse(sf::GetDataPath() + "orca_world.scn");
        parser.SaveLog(sf::GetDataPath() + "orca_world_parse.log");
        if (!success)
        {
            cCritical("Failed to parse orca_world.scn - see orca_world_parse.log next to it for the exact line/tag that failed.");
        }
    }
};

int main(int argc, char** argv)
{
    // --- Rendering / helper display settings ---
    // RenderSettings' aa/ssr/shadows/ao/atmosphere/ocean fields are all
    // sf::RenderQuality enums (DISABLED/LOW/MEDIUM/HIGH), not bools.
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
    h.showSensors = true;
    h.showActuators = true;
    h.showForces = false;

    // --- Simulation manager driving the scenario at 300 Hz physics ---
    OrcaSimulationManager manager(300.0);

    // Data path: everything under ./resources/ (relative to the working
    // directory the executable is run from - see README for the run command).
    sf::GraphicalSimulationApp app("Orca XLUUV Simulation", "./resources/", s, h, &manager);
    app.Run();

    return 0;
}
