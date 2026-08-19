# Orca XLUUV — standalone Stonefish project

A complete, self-contained project — nothing to merge into an existing
Stonefish example, no external `data/` folder needed beyond what's included
here.

```
orca_project/
├── CMakeLists.txt
├── src/
│   └── main.cpp              # app entry point, loads the scenario
└── resources/
    ├── orca_world.scn        # top-level scenario (ocean, seabed, includes vehicle)
    └── data/
        └── orca_xluuv.scn    # the Orca XLUUV robot definition
```

## Prerequisites

You need the Stonefish library itself already built and installed
(this project links against it, it doesn't vendor it):
https://github.com/patrykcieslak/stonefish

Follow that repo's own build/install instructions first (it depends on
Bullet Physics, OpenGL/GLFW, freetype, etc.). Once `find_package(Stonefish)`
can succeed on your machine, this project will build against it.

## Build

```bash
cd orca_project
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Run

```bash
# from the build directory, so ./resources/ resolves correctly
./orca_sim
```

This opens the Stonefish graphical console with the Orca XLUUV spawned
10 m below the surface above a flat seabed at 500 m depth.

## What's modeled, and from what spec

| Spec (given)                          | Modeled as |
|---|---|
| Length ~26 m (85 ft w/ payload module)| `MainHull` cylinder in `orca_xluuv.scn`, `height="26.0"` |
| Mass ~80–85 t (used 82,500 kg)        | Hollow steel shell (`thickness="0.04"`) + payload mass block. Verify the *actual* computed mass in the console log on first run and tune `thickness`/density if it's off — Stonefish derives mass from shell volume × material density, it isn't set directly |
| Payload bay: 10 m (34 ft), 8 t capacity | `PayloadMass` internal cylinder, `height="10.0"`, `mass="8000.0"`, centered amidships |
| Range ~12,000 km, hybrid diesel-electric | *Not* simulated as fuel/energy chemistry — Stonefish has no diesel model. `MainThruster` is given an unconstrained power source. If you want endurance tracking, route it through Stonefish's battery/power-system actuator with your own Wh budget — the source spec gives no cruise speed or power draw, so I didn't invent one |
| Hull diameter                          | Not given in the spec — used the publicly reported ~2.6 m Orca beam (radius 1.3 m) |
| Diesel-electric power plant / mast / control | `CommsMast`, four stern control fins, single stern thruster |

## Not modeled (by design)

The reference image also shows `'Moor-pedo'` mines in the payload bay.
Those weapon payloads are intentionally **not** included — the payload
bay is a generic mass/volume slot; attach your own non-weapon payload,
sensor pod, or cargo model to it if you need one.

## Caveats

- `main.cpp` and both `.scn` files were checked against the actual
  `patrykcieslak/stonefish` source (cloned and grepped directly, not
  guessed) — `Library/src/core/ScenarioParser.cpp` specifically.
- Fixed in this pass, after a real run against the built `orca_sim`
  surfaced them:
  - **Included `.scn` files need a `<scenario>` root**, same as the
    top-level file — `ScenarioParser::IncludeFiles` looks for
    `includedDoc.FirstChildElement("scenario")` and fails with "Root
    node not found" otherwise. `orca_xluuv.scn` previously had `<robot>`
    as its root; it's now wrapped in `<scenario>`.
  - **Only the first `<materials>` block in the merged document is
    read** (`root->FirstChildElement("materials")`, no loop) — unlike
    `<looks>`, which *is* read in a loop over every sibling `<looks>`
    element. So all materials (`HullShell`, `Composite`, plus the
    world's `Rock`/`Sand`) now live in one `<materials>` block in
    `orca_world.scn`; `orca_xluuv.scn` keeps its own `<looks>` block,
    which still works.
  - `physics="submerged"` — the attribute is `physics`, not `physical`.
  - Every primitive shape (`external_part`/`internal_part`/base link
    leaf) needs its own `<material>`, `<look>`, and `<origin>` child
    elements, and shape dimensions go in a `<dimensions .../>` child
    (e.g. `<dimensions radius="" height="" thickness=""/>` for a
    cylinder, `<dimensions xyz="x y z"/>` for a box) — not as bare
    attributes on the `external_part`/`internal_part` tag itself.
  - `<world_transform>` belongs directly under `<robot>`, not nested
    inside `<base_link>`.
  - Overriding a part's mass is `<mass value="8000.0"/>` as a child
    element, not a `mass="8000.0"` attribute.
  - The `thruster`/`simple_thruster`/`propeller` actuator types all
    require a propeller **mesh file** (`<propeller><mesh
    filename="..."/></propeller>`), which this project doesn't ship.
    `MainThruster` now uses `type="push"` instead — a plain force
    actuator with no mesh dependency (`<specs inverted="" lower_limit=""
    upper_limit=""/>`). Swap back to `thruster`/`propeller` if you add a
    propeller mesh later.
  - `main.cpp`: `sf::GetDataPath()` is a free function in
    `utils/SystemUtil.hpp`, not a `SimulationManager` method, and
    `RenderSettings::aa`/`shadows`/`ao`/`atmosphere`/`ocean`/`ssr` are
    `sf::RenderQuality` enums (`DISABLED`/`LOW`/`MEDIUM`/`HIGH`) — there's
    no `msaa`/`ssao` bool field.
- I ran a structural check (Python + `xml.etree`) confirming both files
  are well-formed and every part/actuator/sensor has the required child
  elements per `ScenarioParser.cpp`, but I could not do a full compile +
  scenario-load here (Stonefish's runtime deps — Bullet Physics, GLEW,
  SDL2, etc. — aren't installed in this sandbox). If anything still
  doesn't parse, `orca_world_parse.log` (written next to the scenario
  file on run) will show exactly which tag/line it choked on — paste it
  back and I'll trace it against the parser source again.
