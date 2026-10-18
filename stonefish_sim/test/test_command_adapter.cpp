// Unit checks for OrcaRLCommandAdapter: signs, limits, depth reference,
// altitude guard and NaN safety. Build target: orca_rl_adapter_test.
#include "OrcaRLCommandAdapter.hpp"

#include <cmath>
#include <cstdio>

static int failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

int main()
{
    const double dt = 1.0 / 300.0;
    {
        OrcaRLCommandAdapter a;
        OrcaRLFeedback fb; fb.depth = 12.0;
        OrcaRLForces f = a.Compute({1.0, 0.0, 0.0}, fb, dt);
        CHECK(f.surge > 0.0);                       // forward command -> forward thrust
        CHECK(std::fabs(f.yaw) < 1e-9);
        CHECK(std::fabs(f.heave) < 1e-6);           // depth held where it is
    }
    {
        OrcaRLCommandAdapter a;
        OrcaRLFeedback fb; fb.depth = 12.0;
        OrcaRLForces f = a.Compute({0.0, 0.03, 0.0}, fb, dt);
        CHECK(f.yaw > 0.0);                         // +yaw rate (clockwise/NED) -> bow pushed to starboard
        f = a.Compute({0.0, -0.03, 0.0}, fb, dt);
        CHECK(f.yaw < 0.0);
    }
    {
        OrcaRLCommandAdapter a;
        OrcaRLFeedback fb; fb.depth = 12.0;
        for (int i = 0; i < 300; ++i) a.Compute({0.0, 0.0, 0.2}, fb, dt);
        CHECK(a.depthReference() > 12.15);          // +depth rate -> deeper reference (NED z down)
        OrcaRLForces f = a.Compute({0.0, 0.0, 0.2}, fb, dt);
        CHECK(f.heave > 0.0);                       // + heave force = downward
    }
    {
        OrcaRLCommandAdapter a;
        OrcaRLFeedback fb; fb.depth = 12.0; fb.altitude = 3.5;   // only 0.5 m above the guard
        for (int i = 0; i < 3000; ++i) a.Compute({0.0, 0.0, 0.3}, fb, dt);
        CHECK(a.depthReference() <= 12.5 + 1e-6);  // altitude guard stops the dive
    }
    {
        OrcaRLCommandAdapter a;
        OrcaRLFeedback fb; fb.depth = 12.0;
        OrcaRLForces f = a.Compute({100.0, 100.0, 100.0}, fb, dt);
        CHECK(f.surge <= a.params().surgeLimit);
        CHECK(f.yaw <= a.params().yawLimit);
        CHECK(f.heave <= a.params().heaveLimit);
        f = a.Compute({NAN, NAN, NAN}, fb, dt);
        CHECK(std::isfinite(f.surge) && std::isfinite(f.yaw) && std::isfinite(f.heave));
    }
    std::printf(failures == 0 ? "orca_rl_adapter_test: all checks passed\n"
                              : "orca_rl_adapter_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
