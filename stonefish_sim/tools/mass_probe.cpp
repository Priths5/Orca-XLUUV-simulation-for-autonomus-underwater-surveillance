// orca_mass_probe: prints Stonefish's own mass, buoyancy, CG and CB (hull link
// frame, NED/FRD) for a robot in a scenario - the numbers used to trim the
// Orca and survey-sub models. Usage (from build/):
//   ./orca_mass_probe orca_rl_world.scn OrcaXLUUV ./resources/
// Hydrostatically stable when CG z > CB z (CG below CB, z down) and CG x == CB x.
#include <core/ConsoleSimulationApp.h>
#include <core/SimulationManager.h>
#include <core/ScenarioParser.h>
#include <core/Robot.h>
#include <entities/SolidEntity.h>
#include <cstring>
#include <utils/SystemUtil.hpp>
#include <cstdio>
#include <string>
using namespace tinyxml2;
class P : public sf::ScenarioParser { public: using sf::ScenarioParser::ScenarioParser; protected:
 bool PreProcess(XMLNode* r, const std::map<std::string,std::string>& a) override { if(!sf::ScenarioParser::PreProcess(r,a)) return false; strip(r); return true; }
 static void strip(XMLNode* n){ XMLNode* c=n->FirstChild(); while(c){ XMLNode* nx=c->NextSibling(); XMLElement* e=c->ToElement(); if(e && !strcmp(e->Name(),"light")) n->DeleteChild(c); else strip(c); c=nx; } } };
class M : public sf::SimulationManager { public: std::string f, robot; M(std::string f_, std::string r):sf::SimulationManager(300.0),f(f_),robot(r){}
 void BuildScenario() override { P p(this); if(!p.Parse(sf::GetDataPath()+f)) {printf("parse fail\n"); return;}
  sf::Robot* rb = getRobot(robot); sf::SolidEntity* b = rb->getBaseLink();
  sf::Transform o = b->getOTransform(); sf::Transform cg = b->getCGTransform();
  sf::Vector3 cgb = o.inverse()*cg.getOrigin();
  printf("%s: mass=%.1f kg volume=%.3f m3 (buoyancy %.1f kg) aug=%.1f\n", robot.c_str(), b->getMass(), b->getVolume(), b->getVolume()*1025.0, b->getAugmentedMass());
  printf("  CG in link frame: %.3f %.3f %.3f\n", cgb.x(), cgb.y(), cgb.z());
  sf::Vector3 cb = b->getCB(); printf("  getCB(): %.3f %.3f %.3f\n", cb.x(), cb.y(), cb.z());
  sf::Transform cg2o = b->getCG2OTransform(); printf("  CG2O origin: %.3f %.3f %.3f\n", cg2o.getOrigin().x(), cg2o.getOrigin().y(), cg2o.getOrigin().z());
  sf::Vector3 cbLink = o.inverse() * (cg * cb);
  printf("RESULT mass=%.2f buoy=%.2f cg=%.4f %.4f %.4f cb=%.4f %.4f %.4f\n", b->getMass(), b->getVolume()*1025.0, cgb.x(), cgb.y(), cgb.z(), cbLink.x(), cbLink.y(), cbLink.z());
  fflush(stdout); std::_Exit(0);
 } };
int main(int c, char** v){ M m(v[1], v[2]); sf::ConsoleSimulationApp app("probe", v[3], &m); app.Run(false); return 0; }
