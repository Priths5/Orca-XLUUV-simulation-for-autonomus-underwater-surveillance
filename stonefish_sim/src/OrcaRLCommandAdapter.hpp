/*
    OrcaRLCommandAdapter.hpp

    Converts the RL policy's physical command
        (surge speed [m/s], yaw rate [rad/s], depth rate [m/s])
    into the three actuator forces the Orca model actually has
        MainThruster  (SimpleThruster, thrust along body +X)
        YawThruster   (Push at the bow, force along body +Y)
        HeaveThruster (Push through the CG, force along body +Z = down)

    The RL loop runs at ~1 Hz; this adapter runs inside the Stonefish
    physics callback (300 Hz) and closes a fast inner loop on the measured
    surge speed, yaw rate and depth so a held setpoint stays a setpoint.

    Frames (Stonefish, verified against the library source):
      world  = NED   (x north, y east, z down; depth = z)
      body   = FRD   (x forward, y starboard, z down)
      yaw    > 0 turns the bow from north toward east (clockwise from above)
      depth rate > 0 is downward

    Gains/feed-forwards are calibrated against the actual Stonefish Orca
    (see tools/sysid_stonefish.py and README section "Thrust scaling").
    No Stonefish or ROS types here so it can be unit-tested on its own.
*/

#ifndef ORCA_RL_COMMAND_ADAPTER_HPP
#define ORCA_RL_COMMAND_ADAPTER_HPP

#include <algorithm>
#include <cmath>

struct OrcaRLSetpoint
{
    double surgeSpeed = 0.0;   // m/s, body +X
    double yawRate = 0.0;      // rad/s, NED (+ = clockwise from above)
    double depthRate = 0.0;    // m/s, + = down
};

struct OrcaRLFeedback
{
    double surgeSpeed = 0.0;   // m/s, body +X
    double yawRate = 0.0;      // rad/s, body z (down) = NED yaw rate when level
    double depth = 0.0;        // m, + = down
    double heaveSpeed = 0.0;   // m/s, + = down
    double altitude = -1.0;    // m above seabed, < 0 if no DVL bottom lock
};

struct OrcaRLForces
{
    double surge = 0.0;        // N -> MainThruster
    double yaw = 0.0;          // N -> YawThruster
    double heave = 0.0;        // N -> HeaveThruster
};

struct OrcaRLAdapterParams
{
    // Actuator limits copied from data/orca_xluuv_survey.scn <specs>.
    double surgeLimit = 40000.0;
    double yawLimit = 8000.0;
    double heaveLimit = 55000.0;   // OrcaController clamps heave to this

    // Surge: F = dragQuad*u|u| + dragLin*u  (steady-state feed-forward)
    //           + kpSurge*(u_ref-u) + kiSurge*integral
    // Identified with tools/sysid_stonefish.py (raw thrust steps, Stonefish
    // Orca, trimmed model): 10/20/30 kN -> 0.37/0.68/0.93 m/s, tau63 ~6.5 s.
    double dragQuad = 9000.0;
    double dragLin = 23500.0;
    double kpSurge = 30000.0;
    double kiSurge = 3000.0;

    // Yaw: F = ffYaw*r_ref + kpYaw*(r_ref - r) + kiYaw*integral
    // Identified: r_ss ~= F_yaw / 150000 at cruise (2/4/8 kN -> 0.014/0.028/0.048
    // rad/s), yaw-rate tau63 < 1 s.
    double ffYaw = 150000.0;
    double kpYaw = 150000.0;
    double kiYaw = 30000.0;

    // Depth: the depth-rate command is integrated into a depth reference,
    // then held with the same PD law and gains OrcaController's autopilot uses.
    double depthGain = 28000.0;
    double depthDamping = 90000.0;
    double minDepth = 2.5;          // keep the sail under the surface
    double maxDepth = 3000.0;       // scenario-specific, tightened by altitude guard
    double minAltitude = 3.0;       // DVL floor guard (hull radius 1.3 m + margin)

    // Achievable envelope (identified in Stonefish); setpoints are clipped to it.
    double maxSurgeSpeed = 1.1;    // 40 kN MainThruster limit -> ~1.15 m/s
    double maxYawRate = 0.048;     // 8 kN YawThruster limit -> ~0.048 rad/s (2.7 deg/s)
    double maxDepthRate = 0.3;
};

class OrcaRLCommandAdapter
{
public:
    explicit OrcaRLCommandAdapter(const OrcaRLAdapterParams& p = OrcaRLAdapterParams()) : p_(p) {}

    void Reset(double currentDepth)
    {
        surgeInt_ = 0.0;
        yawInt_ = 0.0;
        depthRef_ = currentDepth;
        initialised_ = true;
    }

    const OrcaRLAdapterParams& params() const { return p_; }
    OrcaRLAdapterParams& params() { return p_; }
    double depthReference() const { return depthRef_; }

    OrcaRLForces Compute(const OrcaRLSetpoint& spIn, const OrcaRLFeedback& fb, double dt)
    {
        OrcaRLForces f;
        if (!(dt > 0.0))
            return f;
        if (!initialised_)
            Reset(fb.depth);

        OrcaRLSetpoint sp = spIn;
        sp.surgeSpeed = Clamp(Finite(sp.surgeSpeed), -0.3 * p_.maxSurgeSpeed, p_.maxSurgeSpeed);
        sp.yawRate = Clamp(Finite(sp.yawRate), -p_.maxYawRate, p_.maxYawRate);
        sp.depthRate = Clamp(Finite(sp.depthRate), -p_.maxDepthRate, p_.maxDepthRate);

        // ---- surge ----
        double u = Finite(fb.surgeSpeed);
        double eU = sp.surgeSpeed - u;
        double ff = p_.dragQuad * sp.surgeSpeed * std::fabs(sp.surgeSpeed) + p_.dragLin * sp.surgeSpeed;
        double surge = ff + p_.kpSurge * eU + p_.kiSurge * surgeInt_;
        if (std::fabs(surge) < p_.surgeLimit || surge * eU < 0.0)   // anti-windup
            surgeInt_ = Clamp(surgeInt_ + eU * dt, -4.0, 4.0);
        f.surge = Clamp(surge, -p_.surgeLimit, p_.surgeLimit);

        // ---- yaw ----
        double r = Finite(fb.yawRate);
        double eR = sp.yawRate - r;
        double yaw = p_.ffYaw * sp.yawRate + p_.kpYaw * eR + p_.kiYaw * yawInt_;
        if (std::fabs(yaw) < p_.yawLimit || yaw * eR < 0.0)
            yawInt_ = Clamp(yawInt_ + eR * dt, -0.2, 0.2);
        f.yaw = Clamp(yaw, -p_.yawLimit, p_.yawLimit);

        // ---- depth ----
        double maxDepth = p_.maxDepth;
        bool bottomLock = fb.altitude > 0.0;
        if (bottomLock)
            maxDepth = std::min(maxDepth, fb.depth + (fb.altitude - p_.minAltitude));
        depthRef_ = Clamp(depthRef_ + sp.depthRate * dt, p_.minDepth, std::max(p_.minDepth, maxDepth));
        double z = Finite(fb.depth);
        double w = Finite(fb.heaveSpeed);
        double heave = p_.depthGain * (depthRef_ - z) - p_.depthDamping * (w - sp.depthRate);
        f.heave = Clamp(heave, -p_.heaveLimit, p_.heaveLimit);
        return f;
    }

private:
    static double Clamp(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }
    static double Finite(double v) { return std::isfinite(v) ? v : 0.0; }

    OrcaRLAdapterParams p_;
    double surgeInt_ = 0.0;
    double yawInt_ = 0.0;
    double depthRef_ = 0.0;
    bool initialised_ = false;
};

#endif // ORCA_RL_COMMAND_ADAPTER_HPP
