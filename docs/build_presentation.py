#!/usr/bin/env python3
"""
build_presentation.py - builds docs/Orca_RL_Stonefish_Review.pptx from the
college template docs/templates/PPT_format_22AIE450.pptx (green header/footer
bands, title slide, "[... Review]" footer, slide numbers), using only the
Python standard library + Pillow: the template's own slide XML is reused as the
frame of every slide and the content is written as DrawingML.

    python3 docs/make_figures.py          # first, if the figures are missing
    python3 docs/build_presentation.py    # -> docs/Orca_RL_Stonefish_Review.pptx
"""
import json
import os
import re
import shutil
import zipfile
from xml.sax.saxutils import escape

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(ROOT, "docs")
FIG = os.path.join(DOCS, "figures")
TEMPLATE = os.path.join(DOCS, "templates", "PPT_format_22AIE450.pptx")
OUT = os.path.join(DOCS, "Orca_RL_Stonefish_Review.pptx")
LOGS = os.path.join(ROOT, "orca_rl", "logs")

REVIEW = "Project Review"
EMU = 914400

# palette: the template's green + one dark and one warm accent
GREEN = "1A9B46"
DKGREEN = "14532D"
NAVY = "1E3A8A"
TEAL = "0F766E"
ORANGE = "C2410C"
SLATE = "475569"
INK = "1F2937"
TINT_G = "E8F5EC"
TINT_O = "FFF1E6"
TINT_N = "EEF2FF"
TINT_S = "F1F5F9"


def emu(inches):
    return int(round(inches * EMU))


# ------------------------------------------------------------------ DrawingML
class Slide:
    def __init__(self, title):
        self.title = title
        self.shapes = []
        self.images = []          # (rId, path)
        self.next_id = 100

    def _id(self):
        self.next_id += 1
        return self.next_id

    @staticmethod
    def _runs(text, size, color, bold=False, italic=False, font=None):
        """text with **bold** spans -> <a:r> runs."""
        out = []
        for i, part in enumerate(re.split(r"\*\*", text)):
            if not part:
                continue
            b = bold or (i % 2 == 1)
            f = f'<a:latin typeface="{font}"/>' if font else ""
            out.append(f'<a:r><a:rPr lang="en-US" sz="{int(size * 100)}" b="{int(b)}" i="{int(italic)}" dirty="0">'
                       f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>{f}</a:rPr>'
                       f'<a:t>{escape(part)}</a:t></a:r>')
        return "".join(out)

    def _paras(self, items, size, color, align="l", space=6, font=None):
        """items: str or ('-', text) bullet or ('--', text) sub-bullet or ('h', text) heading."""
        ps = []
        for it in items:
            kind, text = (it if isinstance(it, tuple) else ("", it))
            if kind == "-":
                ppr = (f'<a:pPr marL="285750" indent="-285750" algn="{align}"><a:spcAft><a:spcPts val="{space * 100}"/>'
                       f'</a:spcAft><a:buClr><a:srgbClr val="{GREEN}"/></a:buClr><a:buFont typeface="Arial"/>'
                       f'<a:buChar char="&#8226;"/></a:pPr>')
                ps.append(f'<a:p>{ppr}{self._runs(text, size, color, font=font)}</a:p>')
            elif kind == "--":
                ppr = (f'<a:pPr marL="628650" lvl="1" indent="-228600" algn="{align}"><a:spcAft><a:spcPts val="{space * 60}"/>'
                       f'</a:spcAft><a:buFont typeface="Arial"/><a:buChar char="&#8211;"/></a:pPr>')
                ps.append(f'<a:p>{ppr}{self._runs(text, size - 2, SLATE, font=font)}</a:p>')
            elif kind == "h":
                ppr = f'<a:pPr algn="{align}"><a:spcBef><a:spcPts val="300"/></a:spcBef><a:spcAft><a:spcPts val="{space * 100}"/></a:spcAft><a:buNone/></a:pPr>'
                ps.append(f'<a:p>{ppr}{self._runs(text, size + 2, NAVY, bold=True, font=font)}</a:p>')
            else:
                ppr = f'<a:pPr algn="{align}"><a:spcAft><a:spcPts val="{space * 100}"/></a:spcAft><a:buNone/></a:pPr>'
                ps.append(f'<a:p>{ppr}{self._runs(text, size, color, font=font)}</a:p>')
        return "".join(ps)

    def text(self, x, y, w, h, items, size=16, color=INK, align="l", anchor="t", space=6, fill=None,
             line=None, geom="rect", inset=0.08, font=None, name="Text"):
        fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
        ln = f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
        adj = '<a:avLst><a:gd name="adj" fmla="val 8000"/></a:avLst>' if geom == "roundRect" else "<a:avLst/>"
        i = emu(inset)
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{self._id()}" name="{name} {self.next_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}">{adj}</a:prstGeom>{fill_xml}{ln}</p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="{i}" tIns="{i}" rIns="{i}" bIns="{i}" anchor="{anchor}"><a:normAutofit/></a:bodyPr>'
            f'<a:lstStyle/>{self._paras(items if isinstance(items, list) else [items], size, color, align, space, font)}</p:txBody></p:sp>')

    def card(self, x, y, w, h, head, body, color=GREEN, tint=TINT_G, size=16, head_size=18):
        items = [("h", head)] + [("-", b) if not isinstance(b, tuple) else b for b in body]
        self.text(x, y, w, h, [(k, t) for k, t in items], size=size, fill=tint, geom="roundRect", inset=0.15,
                  name="Card", space=4)
        # recolour the heading to the card colour
        self.shapes[-1] = self.shapes[-1].replace(f'<a:srgbClr val="{NAVY}"/>', f'<a:srgbClr val="{color}"/>', 1)

    def stat(self, x, y, w, h, big, label, color=GREEN, tint=TINT_G):
        self.text(x, y, w, h, [big], size=36, color=color, align="ctr", anchor="ctr", fill=tint, geom="roundRect",
                  name="Stat", space=0)
        self.text(x, y + h - 0.02, w, 0.55, [label], size=12, color=SLATE, align="ctr", name="StatLabel", space=0)

    def image(self, path, x, y, w, h, align="ctr"):
        rid = f"rId{10 + len(self.images)}"
        self.images.append((rid, path))
        with Image.open(path) as im:
            iw, ih = im.size
        s = min(w / iw, h / ih)
        dw, dh = iw * s, ih * s
        ox = x + (w - dw) / 2 if align == "ctr" else x
        oy = y + (h - dh) / 2
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{self._id()}" name="Picture {self.next_id}" descr="{escape(os.path.basename(path))}"/>'
            f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{emu(ox)}" y="{emu(oy)}"/><a:ext cx="{emu(dw)}" cy="{emu(dh)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')

    def table(self, x, y, w, rows, col_w, size=12, row_h=0.36, header_fill=GREEN):
        cols = "".join(f'<a:gridCol w="{emu(c)}"/>' for c in col_w)
        trs = []
        for ri, row in enumerate(rows):
            tcs = []
            for ci, cell in enumerate(row):
                hdr = ri == 0
                fill = header_fill if hdr else ("FFFFFF" if ri % 2 else TINT_S)
                col = "FFFFFF" if hdr else INK
                bold = hdr or ci == 0
                runs = self._runs(str(cell), size, col, bold=bold)
                border = (f'<a:lnB w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnB>')
                tcs.append(f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:buNone/></a:pPr>{runs}</a:p></a:txBody>'
                           f'<a:tcPr marL="{emu(0.08)}" marR="{emu(0.08)}" marT="{emu(0.04)}" marB="{emu(0.04)}" anchor="ctr">'
                           f'{border}<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill></a:tcPr></a:tc>')
            trs.append(f'<a:tr h="{emu(row_h)}">{"".join(tcs)}</a:tr>')
        self.shapes.append(
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{self._id()}" name="Table {self.next_id}"/>'
            f'<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(row_h * len(rows))}"/></p:xfrm>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl>'
            f'<a:tblPr firstRow="1" bandRow="1"/><a:tblGrid>{cols}</a:tblGrid>{"".join(trs)}</a:tbl></a:graphicData></a:graphic>'
            f'</p:graphicFrame>')

    def caption(self, x, y, w, text):
        self.text(x, y, w, 0.4, [text], size=11, color=SLATE, name="Caption", space=0)


# ------------------------------------------------------------------ package
def number_slide(x, number):
    """Slide number at the right end of the footer band, explicitly placed and styled so it is visible."""
    x = re.sub(r'(<p:ph type="sldNum"[^>]*/></p:nvPr></p:nvSpPr>)<p:spPr(?:/>|>.*?</p:spPr>)',
               r'\1<p:spPr><a:xfrm><a:off x="11430000" y="6356350"/><a:ext cx="640080" cy="365125"/></a:xfrm></p:spPr>',
               x, count=1, flags=re.S)
    return re.sub(r'(type="slidenum")><a:rPr[^>]*/><a:t>[^<]*(</a:t>)',
                  rf'\1><a:rPr lang="en-IN" sz="1400" b="1" smtClean="0"><a:solidFill><a:srgbClr val="1F3864"/></a:solidFill>'
                  rf'</a:rPr><a:t>{number}\2', x, count=1)


def content_slide_xml(base, slide, number):
    x = base
    # title text (keep the template's run formatting), sized to fit the header band
    size = 3600 if len(slide.title) <= 40 else 3000
    x = re.sub(r'(<p:ph type="title"/></p:nvPr></p:nvSpPr><p:spPr>.*?</p:spPr><p:txBody><a:bodyPr)/>',
               r'\1><a:normAutofit/></a:bodyPr>', x, count=1, flags=re.S)
    x = x.replace('<a:rPr lang="en-IN" dirty="0"><a:solidFill><a:schemeClr val="accent1"><a:lumMod val="50000"/></a:schemeClr></a:solidFill></a:rPr><a:t>Outline</a:t>',
                  f'<a:rPr lang="en-IN" sz="{size}" b="1" dirty="0"><a:solidFill><a:schemeClr val="accent1"><a:lumMod val="50000"/></a:schemeClr></a:solidFill></a:rPr><a:t>{escape(slide.title)}</a:t>', 1)
    x = re.sub(r'<a:br><a:rPr[^>]*>.*?</a:rPr></a:br>', '', x, count=1, flags=re.S)
    x = x.replace("<a:t>Mid Project Review]</a:t>", f"<a:t>{REVIEW}]</a:t>")
    x = number_slide(x, number)
    # drop the empty content placeholder, add our shapes before </p:spTree>
    x = re.sub(r'<p:sp><p:nvSpPr><p:cNvPr id="3" name="Content Placeholder 2">.*?</p:sp>', "", x, count=1, flags=re.S)
    x = x.replace("</p:spTree>", "".join(slide.shapes) + "</p:spTree>", 1)
    return x


def title_slide_xml(base, number):
    x = base
    rep = [("<a:t> Mid semester project review– </a:t>", "<a:t>Project review – </a:t>"),
           ("<a:t>August 2026</a:t>", "<a:t>October 2026</a:t>"),
           ("<a:t>22AIE450 Speech Processing</a:t>", "<a:t>Reinforcement Learning</a:t>"),
           ("<a:t>Mid Project Review]</a:t>", f"<a:t>{REVIEW}]</a:t>"),
           ("<a:t>Batch and Group Number:</a:t>", "<a:t>Batch and Group Number: C-7</a:t>")]
    for a, b in rep:
        x = x.replace(a, b)
    # project title + team (two "Title 3" placeholders on the template's title slide)
    x = x.replace("<a:t>Project Title</a:t>",
                  "<a:t>Orca-RL in Stonefish: obstacle-aware RL navigation over ROS 2</a:t>", 1)
    team = ("Vijay Srinivas P (CB.SC.U4AIE23263) · Joel Sahay D (CB.SC.U4AIE23234) · "
            "D.K. Preetham Reddy (CB.SC.U4AIE23222) · A Prithvi (CB.SC.U4AIE23201)  |  Guide: Dr. D. Palmani")
    x = x.replace("<a:t>List of Team Members</a:t>", f"<a:t>{escape(team)}</a:t>", 1)
    # shrink the two title-slide text sizes so the long title / team line fit
    x = x.replace('sz="6000"', 'sz="3600"', 1)
    x = re.sub(r'sz="2500"', 'sz="1500"', x, count=1)
    x = x.replace('<a:off x="150810" y="5675807"/><a:ext cx="2854308"', '<a:off x="150810" y="5675807"/><a:ext cx="4200000"', 1)
    x = number_slide(x, number)
    return x


def build(slides):
    work = os.path.join(DOCS, "_build", "pptx_work")
    shutil.rmtree(work, ignore_errors=True)
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(work)
    p = lambda *a: os.path.join(work, *a)
    base_title = open(p("ppt", "slides", "slide1.xml"), encoding="utf-8").read()
    base_content = open(p("ppt", "slides", "slide2.xml"), encoding="utf-8").read()
    # remove the template's slides and their notes
    shutil.rmtree(p("ppt", "slides"))
    shutil.rmtree(p("ppt", "notesSlides"), ignore_errors=True)
    os.makedirs(p("ppt", "slides", "_rels"))
    os.makedirs(p("ppt", "media"), exist_ok=True)

    ct = open(p("[Content_Types].xml"), encoding="utf-8").read()
    ct = re.sub(r'<Override PartName="/ppt/(slides|notesSlides)/[^"]+"[^>]*/>', "", ct)
    if 'Extension="png"' not in ct:
        ct = ct.replace('<Default Extension="rels"', '<Default Extension="png" ContentType="image/png"/><Default Extension="rels"')
    prels = open(p("ppt", "_rels", "presentation.xml.rels"), encoding="utf-8").read()
    prels = re.sub(r'<Relationship Id="[^"]+" Type="[^"]+/slide" Target="slides/[^"]+"/>', "", prels)
    pres = open(p("ppt", "presentation.xml"), encoding="utf-8").read()

    sld_ids, overrides, rels = [], [], []
    media_n = 0
    for n, slide in enumerate(slides, start=1):
        if slide == "TITLE":
            xml = title_slide_xml(base_title, n)
            layout, imgs = "slideLayout1.xml", []
        else:
            xml = content_slide_xml(base_content, slide, n)
            layout, imgs = "slideLayout2.xml", slide.images
        r = [f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/{layout}"/>']
        for rid, path in imgs:
            media_n += 1
            name = f"image{media_n}{os.path.splitext(path)[1].lower()}"
            shutil.copy(path, p("ppt", "media", name))
            r.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{name}"/>')
        with open(p("ppt", "slides", f"slide{n}.xml"), "w", encoding="utf-8") as f:
            f.write(xml)
        with open(p("ppt", "slides", "_rels", f"slide{n}.xml.rels"), "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    + "".join(r) + "</Relationships>")
        overrides.append(f'<Override PartName="/ppt/slides/slide{n}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
        rels.append(f'<Relationship Id="rIdS{n}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{n}.xml"/>')
        sld_ids.append(f'<p:sldId id="{300 + n}" r:id="rIdS{n}"/>')

    ct = ct.replace("</Types>", "".join(overrides) + "</Types>")
    prels = prels.replace("</Relationships>", "".join(rels) + "</Relationships>")
    pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", "<p:sldIdLst>" + "".join(sld_ids) + "</p:sldIdLst>", pres, flags=re.S)
    open(p("[Content_Types].xml"), "w", encoding="utf-8").write(ct)
    open(p("ppt", "_rels", "presentation.xml.rels"), "w", encoding="utf-8").write(prels)
    open(p("ppt", "presentation.xml"), "w", encoding="utf-8").write(pres)
    open(p("docProps", "app.xml"), "w", encoding="utf-8").write(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office PowerPoint</Application>'
        f'<PresentationFormat>Widescreen</PresentationFormat><Slides>{len(slides)}</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides></Properties>')
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p("[Content_Types].xml"), "[Content_Types].xml")
        for dp, _, fs in os.walk(work):
            for fn in fs:
                full = os.path.join(dp, fn)
                arc = os.path.relpath(full, work)
                if arc != "[Content_Types].xml":
                    z.write(full, arc)
    print("wrote", OUT, f"({len(slides)} slides)")


# ------------------------------------------------------------------ content
def F(name):
    return os.path.join(FIG, name)


def load(p):
    with open(p) as f:
        return json.load(f)


def deck():
    ev = load(os.path.join(LOGS, "eval_obstacle_summary.json"))
    rl_ev, pl_ev = ev["trained_policy_obstacle.npz"], ev["classical_planner"]
    rb = load(os.path.join(DOCS, "data", "reward_breakdown.json"))
    S = []
    S.append("TITLE")

    s = Slide("Outline")
    s.card(0.6, 1.75, 3.9, 4.8, "Part A · Why and how: RL", [
        "Real-world situations and why this method", "The RL objective and reward",
        "How the objective is optimised", "Diagnosis of the shipped policy",
        "Training and offline results"], color=TEAL, tint=TINT_G, size=18)
    s.card(4.72, 1.75, 3.9, 4.8, "Part B · The simulation", [
        "Stonefish and the Orca model", "The base world with obstacles", "Fixing the vehicle model",
        "System identification", "From action to thruster force"], color=NAVY, tint=TINT_N, size=18)
    s.card(8.84, 1.75, 3.9, 4.8, "Part C · Connecting them", [
        "Architecture over ROS 2", "From Gazebo to Stonefish", "Data exchanged, frames, timing",
        "Closed-loop results in 3 worlds", "Summary, limits, next phase"], color=ORANGE, tint=TINT_O, size=18)
    S.append(s)

    s = Slide("Introduction and Motivation")
    s.text(0.6, 1.7, 7.0, 4.9, [
        ("-", "An extra-large uncrewed submarine such as the **Boeing Orca** has to survey long distances "
              "**without GPS**, with a position estimate that degrades underwater."),
        ("-", "The mid-review delivered a reinforcement-learning stack (physics, sensors, particle filter, safety shield, "
              "trained policy) and, separately, a **Stonefish** simulator with survey scenarios."),
        ("-", "The two tracks were **not connected**: the bridge between them assumed a different simulator (Gazebo) "
              "and the Stonefish scenes used a hand-written reactive controller."),
        ("-", "This phase connects them: the RL policy now **drives the Stonefish Orca over ROS 2** and avoids rocks, "
              "fish and other submarines **using only its own sonar**.")], size=17)
    s.image(F("stonefish_orca_view.png"), 7.9, 1.75, 4.9, 3.6)
    s.caption(7.9, 5.0, 4.9, "Stonefish: the Orca among rock pinnacles in the obstacle world")
    S.append(s)

    s = Slide("Real-World Situations")
    cases = [
        ("Charts are incomplete", ORANGE, TINT_O,
         ["2005: a US Navy submarine struck an undersea mountain near Guam that was missing from its charts",
          "2021: another struck an uncharted seamount in the South China Sea",
          "A pre-planned route is not enough: the vehicle must sense and react"]),
        ("No GPS under water", NAVY, TINT_N,
         ["Below the surface only inertial and Doppler (DVL) dead reckoning is available",
          "The error grows with distance; a long-range vehicle cannot keep surfacing",
          "Decisions must be made on an uncertain estimate"]),
        ("Long survey missions", TEAL, TINT_G,
         ["Seabed mapping, pipeline and cable routes, wind-farm sites, search operations",
          "2014: deep-sea vehicles searched the seabed for weeks for flight MH370",
          "Lawnmower lanes, hours of autonomy, limited battery"]),
        ("Shared, moving water", "7C3AED", TINT_S,
         ["Fish schools, marine mammals, other survey vehicles, ship traffic",
          "Obstacles appear without warning and move",
          "Losing a vehicle is costly: Autosub2 was lost under an Antarctic ice shelf in 2005"]),
    ]
    for i, (h, col, tint, body) in enumerate(cases):
        x = 0.6 + (i % 2) * 6.2
        y = 1.7 + (i // 2) * 2.5
        s.card(x, y, 5.95, 2.35, h, body, color=col, tint=tint, size=16, head_size=17)
    S.append(s)

    s = Slide("Why This Method Is Proposed")
    s.image(F("20_motivation_map.png"), 0.3, 1.55, 8.9, 4.3)
    s.text(9.35, 1.65, 3.6, 5.0, [
        ("h", "Why reinforcement learning"),
        ("-", f"A hand-tuned planner trades goals against clearance poorly: it collides in "
              f"{pl_ev['collision_rate'] * 100:.0f}% of test missions, the learned policy in {rl_ev['collision_rate'] * 100:.0f}%."),
        ("h", "Why a safety shield"),
        ("-", "Learned behaviour is not a guarantee; hard rules wrap it."),
        ("h", "Why simulation first"),
        ("-", "Collisions cannot be tested on a 50-tonne vehicle; Stonefish gives real physics and sensors.")], size=14)
    s.text(0.6, 5.95, 8.5, 0.6, ["Each situation on the left maps to one design element of this work."], size=14,
           color=SLATE)
    S.append(s)

    s = Slide("Literature Review")
    s.table(0.5, 1.7, 12.3, [
        ["Sl No", "Title and year", "Venue", "Methodology", "Key findings", "Limitations"],
        ["1", "Evolution Strategies as a Scalable Alternative to RL (2017)", "arXiv / OpenAI", "Antithetic Gaussian perturbations, rank-shaped fitness", "Matches policy-gradient RL on control; trivially parallel", "Sample-hungry; slow from scratch on sparse tasks"],
        ["2", "Proximal Policy Optimization (2017)", "arXiv / OpenAI", "Clipped surrogate policy gradient", "Stable on-policy RL baseline", "Needs automatic differentiation and a deep-learning toolchain"],
        ["3", "Handbook of Marine Craft Hydrodynamics (2011)", "Wiley (Fossen)", "Rigid-body + hydrodynamic modelling", "Standard UUV dynamics used in our fast simulator", "Parameters for Orca not public"],
        ["4", "Stonefish: open-source marine robotics simulator (2019)", "IEEE/OES OCEANS", "Bullet physics + hydrodynamics + sonar/DVL models", "Realistic sensors, ROS interface", "Scripted bodies never yield; no run-time reset"],
        ["5", "UUV Simulator (2016)", "IEEE/MTS OCEANS", "Gazebo plugins for UUVs", "Common ROS UUV stack", "Gazebo Classic; the old bridge assumed it"],
        ["6", "DAgger: imitation learning with no-regret online learning (2011)", "AISTATS", "Iterative relabelling of learner-visited states", "Fixes covariate shift of behaviour cloning", "Needs an expert to query"],
    ], col_w=[0.6, 3.0, 1.5, 2.5, 2.5, 2.2], size=11.5, row_h=0.7)
    S.append(s)

    s = Slide("Objectives")
    objs = ["Make the Stonefish application ROS 2-aware and drive the real Orca thrusters",
            "Rewrite the Python bridge for Stonefish: real sensors in, physical setpoints out",
            "Keep the RL methodology: localisation, health monitoring, safety shield, network + ES",
            "Give the policy goal and sonar inputs and train it to avoid obstacles",
            "Add obstacles, fish schools and a traffic submarine to the base world",
            "Verify step by step, and report honestly where it fails"]
    for i, o in enumerate(objs):
        y = 1.75 + i * 0.8
        s.text(0.7, y, 0.62, 0.62, [str(i + 1)], size=20, color="FFFFFF", align="ctr", anchor="ctr", fill=GREEN,
               geom="ellipse", inset=0.0, name="Num")
        s.text(1.55, y + 0.02, 11.2, 0.62, [o], size=20, anchor="ctr", name="Obj")
    S.append(s)

    # ------------------------------------------------------------ Part A
    s = Slide("Part A · RL Formulation")
    s.card(0.6, 1.75, 4.0, 4.8, "State (observation)", [
        "Estimated position, depth, heading",
        "Uncertainty, battery, thruster wear, navigation residual, depth margin",
        "Terrain richness, risk",
        "New: goal bearing and distance, sonar sectors, flank clearances, depth error"], color=NAVY, tint=TINT_N, size=17)
    s.card(4.75, 1.75, 3.8, 4.8, "Action", [
        "a ∈ [-1, 1]³ from a 2 x 64 tanh network",
        "speed, yaw rate, depth rate",
        "scaled to what the vehicle can do",
        "then the safety shield, then ROS 2"], color=TEAL, tint=TINT_G, size=17)
    s.card(8.7, 1.75, 4.1, 4.8, "Reward", [
        "+ progress toward the waypoint",
        "- uncertainty, - risk, - energy",
        "- shield overrides, - idling",
        "+ waypoint bonus",
        "- collision (episode ends)"], color=ORANGE, tint=TINT_O, size=17)
    S.append(s)

    s = Slide("The RL Objective: What Is Optimised")
    s.image(F("eq_objective.png"), 0.7, 1.75, 11.9, 0.95)
    s.image(F("eq_policy.png"), 0.7, 2.85, 11.9, 0.8)
    s.card(0.6, 3.9, 6.0, 2.7, "What is optimised", [
        "θ: the 5,891 weights and biases of the policy network",
        "J(θ): the expected total reward of one mission",
        "Expectation over random worlds: start, obstacles, fish, currents, waypoints"], color=TEAL, tint=TINT_G, size=17)
    s.card(6.8, 3.9, 6.0, 2.7, "Episode", [
        "1 s per step, up to 420 steps (7 min), 3 waypoints",
        "Ends at the last waypoint, on collision or on time-out",
        "Undiscounted total reward: missions are finite"], color=NAVY, tint=TINT_N, size=17)
    S.append(s)

    s = Slide("The Reward Function, Term by Term")
    s.image(F("eq_reward.png"), 0.5, 1.65, 12.3, 0.62)
    s.table(0.5, 2.45, 12.3, [
        ["Term", "Weight", "What it measures", "Real-world meaning"],
        ["Progress  Δd", "1.0 per m", "drop in estimated distance to the waypoint", "get the survey done"],
        ["Waypoint bonus", "200", "waypoint reached (within 10 m)", "mission completion"],
        ["Uncertainty  σ²/100", "0.0025", "spread of the particle filter (m²)", "stay localisable without GPS"],
        ["Risk  ρ", "0.4", "sonar proximity and low altitude, 0 to 1", "keep a safe margin"],
        ["Energy  E/100", "0.02", "watt-hours used this step", "battery endurance"],
        ["Shield", "1.5", "the safety shield had to override", "do not lean on the safety net"],
        ["Idle", "0.5", "speed below 0.2 m/s", "do not stall to avoid penalties"],
        ["Collision", "300", "hull touched an obstacle; episode ends", "losing the vehicle"],
    ], col_w=[2.6, 1.5, 4.4, 3.8], size=13, row_h=0.46)
    S.append(s)

    s = Slide("From Action to a Safe Command")
    s.image(F("eq_action.png"), 0.5, 1.7, 12.3, 0.62)
    s.card(0.6, 2.6, 6.0, 3.95, "Limits from the real vehicle model", [
        "Top speed 1.1 m/s (40 kN main thruster)",
        "Yaw rate up to 0.045 rad/s, less at higher speed",
        "Depth rate up to 0.3 m/s",
        "Identified on the Stonefish Orca, not assumed"], color=NAVY, tint=TINT_N, size=17)
    s.card(6.8, 2.6, 6.0, 3.95, "Shield applied after the policy", [
        "Original rules: depth rating, battery reserve, localisation",
        "New rules: obstacle stop / slow, seabed floor, depth band",
        "Uses sensor-derived quantities only",
        "Each override costs reward (the shield term)"], color=ORANGE, tint=TINT_O, size=17)
    S.append(s)

    s = Slide("How the Objective Is Optimised")
    s.text(0.6, 1.65, 12.2, 0.4, [("h", "1  Warm start: imitate a classical planner (behaviour cloning + one DAgger round)")],
           size=16)
    s.image(F("eq_bc.png"), 0.8, 2.15, 6.0, 0.85)
    s.text(7.1, 2.1, 5.7, 1.0, ["111k planner samples + 25k relabelled states; Adam, mean-squared error"], size=14,
           color=SLATE)
    s.text(0.6, 3.2, 12.2, 0.4, [("h", "2  Optimise J(θ) directly with Evolution Strategies")], size=16)
    s.image(F("eq_es.png"), 0.8, 3.7, 7.3, 0.85)
    s.card(8.4, 3.65, 4.4, 2.95, "Settings", [
        "n = 48 (24 mirrored pairs)",
        "σ = 0.03, α = 0.01, 200 generations",
        "4 shared episodes per candidate",
        "rank shaping R(·) in [-0.5, 0.5]"], color=TEAL, tint=TINT_G, size=14)
    s.text(0.6, 4.8, 7.6, 1.8, [
        ("-", "No gradient through the simulator is needed; works with the abrupt collision penalty."),
        ("-", "Every candidate is an independent mission, so 14 cores evaluate in parallel.")], size=14)
    S.append(s)

    s = Slide("The Shipped Policy: Training")
    s.image(F("01_original_es_training.png"), 0.5, 1.7, 7.6, 4.3)
    s.text(8.4, 1.8, 4.4, 4.6, [
        ("-", "Trained with **Evolution Strategies** for 120 generations on the original 12-input environment."),
        ("-", "Mean return rises from -3000 to about -100: the population learns to **avoid penalties**, not to reach goals."),
        ("-", "We kept this policy unchanged; it can still be selected at run time.")], size=16)
    S.append(s)

    s = Slide("Diagnosis: Why It Cannot Navigate")
    s.image(F("02_original_policy_probe.png"), 0.4, 1.65, 8.6, 3.3)
    s.stat(9.4, 1.8, 3.3, 1.3, "0 / 5", "goals reached in its own evaluation", color=ORANGE, tint=TINT_O)
    s.stat(9.4, 3.65, 3.3, 1.3, "+1.0", "yaw output, for every heading", color=ORANGE, tint=TINT_O)
    s.text(0.6, 5.15, 12.2, 1.4, [
        ("-", "The observation contains **no goal and no obstacle information**, so no policy trained on it can steer to a "
              "waypoint or around a rock."),
        ("-", "In Stonefish it does exactly what the probe predicts: a tight circle near the start (1 of 10 waypoints in 200 s).")],
           size=16)
    S.append(s)

    s = Slide("A Sonar-Aware Observation")
    s.image(F("04_observation_layout.png"), 0.4, 1.7, 12.5, 2.8)
    s.text(0.6, 4.6, 6.0, 2.0, [
        ("-", "The **first 12 inputs are unchanged**, same order and formulas."),
        ("-", "+3 goal inputs from the next waypoint, using the **estimated** position."),
        ("-", "+5 profiling-sonar sector clearances (port to starboard) measured from the bow.")], size=16)
    s.text(6.8, 4.6, 6.0, 2.0, [
        ("-", "+2 hull-flank clearances from a short-term **sonar memory**."),
        ("-", "+1 depth error relative to the mission depth."),
        ("-", "Same network type: 23 -> 64 -> 64 -> 3, tanh, about 5.9k parameters.")], size=16)
    S.append(s)

    s = Slide("Stonefish-Scale Training Environment")
    s.image(F("06_perception.png"), 0.4, 1.65, 5.6, 4.95)
    s.text(6.3, 1.7, 6.6, 4.9, [
        ("h", "Fast training environment (offline only)"),
        ("-", "Vehicle limits **identified on the Stonefish Orca**: 1.1 m/s, speed-dependent yaw rate, time constants."),
        ("-", "27 m x 1.3 m **hull** for collisions; rocks, ridge walls, fish and subs that move."),
        ("-", "The **same swept sonar** as the model: 64 steps over 160° at 10 Hz, in 5° bins as the simulator publishes."),
        ("-", "The sonar memory keeps rocks that left the beam; a later beam that sees through a point clears it."),
        ("-", "Navigation = DVL dead reckoning into the existing particle filter, exactly as on the vehicle.")], size=15)
    S.append(s)

    s = Slide("Safety Shield: Original Rules + Obstacle Rules")
    s.table(0.6, 1.75, 12.1, [
        ["Rule", "Trigger (sensor-derived only)", "Override"],
        ["Depth rating (original)", "depth ≥ 98 % of rating", "climb"],
        ["Battery reserve (original)", "battery < 8 %", "limp-home speed, damped yaw"],
        ["Localisation (original)", "covariance trace > 5000 m²", "clamp yaw rate"],
        ["Obstacle stop", "something < 7 m dead ahead", "crawl 0.15 m/s, turn hard to the open side"],
        ["Obstacle slow", "something < 12 m in the front sectors", "cap speed at 0.45 m/s"],
        ["Seabed floor", "DVL altitude < 3 m", "no further descent"],
        ["Depth band", "|depth - mission depth| > 3 m", "push back into the band"],
    ], col_w=[3.0, 4.6, 4.5], size=15, row_h=0.52)
    s.text(0.6, 5.95, 12.1, 0.7, ["Every override costs the policy reward, so it learns to avoid needing the shield; "
                                  "the same shield runs in training and on the vehicle."], size=16, color=SLATE)
    S.append(s)

    s = Slide("Training Pipeline: Behaviour Cloning + ES")
    s.image(F("07_training_pipeline.png"), 0.4, 1.7, 12.5, 2.8)
    s.card(0.6, 4.55, 6.0, 2.0, "Why a warm start", [
        "ES from a random network reached only 8 % of goals after 50 generations",
        "Cloning a classical planner gives a sensible starting behaviour"], color=ORANGE, tint=TINT_O, size=15)
    s.card(6.8, 4.55, 6.0, 2.0, "What ES then optimises", [
        "The real reward, collisions included, with σ = 0.03",
        "Shared evaluation episodes per generation, 14 workers in parallel"], color=TEAL, tint=TINT_G, size=15)
    S.append(s)

    s = Slide("ES Fine-Tuning of the Cloned Policy")
    s.image(F("08_es_obstacle_training.png"), 0.5, 1.7, 7.9, 4.6)
    s.text(8.7, 1.8, 4.1, 4.6, [
        ("-", "200 generations x 48 candidates x 4 episodes (65 min on 14 cores)."),
        ("-", "Generation-to-generation noise comes from new episodes each generation."),
        ("-", "Model selection on 100 **validation** missions; a later 80-generation fine-tune was worse (16 % vs 8 % collisions) "
              "and was **rejected**.")], size=16)
    S.append(s)

    s = Slide("Where the Reward Goes")
    s.image(F("19_reward_breakdown.png"), 0.3, 1.6, 8.5, 3.75)
    r, p = rb["rl"], rb["planner"]
    s.text(8.8, 1.75, 4.0, 4.8, [
        ("-", f"The RL policy earns more **waypoint bonus** ({r['goal bonus']:.0f} vs {p['goal bonus']:.0f}) and more "
              f"**progress** ({r['progress']:.0f} vs {p['progress']:.0f})."),
        ("-", f"It **halves the collision penalty** ({r['collision']:.0f} vs {p['collision']:.0f})."),
        ("-", "It pays a little more for shield overrides: it accepts brief interventions to keep moving."),
        ("-", "Uncertainty and energy terms are small at this scale (flat seabeds, 7-minute missions).")], size=15)
    s.text(0.6, 5.4, 7.9, 1.0, ["Mean contribution of each reward term over 40 held-out missions "
                                "(energy approximated from speed)."], size=13, color=SLATE)
    S.append(s)

    s = Slide("Offline Results: 100 Unseen Missions")
    s.image(F("09_offline_eval.png"), 0.4, 1.65, 12.5, 3.5)
    s.stat(0.9, 5.35, 3.4, 1.0, f"{rl_ev['goal_fraction'] * 100:.0f} %", "goals reached (planner "
           f"{pl_ev['goal_fraction'] * 100:.0f} %)", color=TEAL)
    s.stat(4.95, 5.35, 3.4, 1.0, f"{rl_ev['collision_rate'] * 100:.0f} %", "collision rate (planner "
           f"{pl_ev['collision_rate'] * 100:.0f} %)", color=TEAL)
    s.stat(9.0, 5.35, 3.4, 1.0, f"{rl_ev['median_min_clearance_m']:.1f} m", "median closest pass (planner "
           f"{pl_ev['median_min_clearance_m']:.1f} m)", color=TEAL)
    S.append(s)

    s = Slide("Same Mission, Two Controllers")
    s.image(F("05_training_env_episode.png"), 0.5, 1.65, 8.4, 4.95)
    s.text(9.1, 1.8, 3.8, 4.7, [
        ("-", "Classical planner: attraction to the goal, repulsion from close sectors."),
        ("-", "It cuts the corner and hits the rocks beside the first goal."),
        ("-", "The RL policy slows down, swings wide around the cluster and completes all three goals."),
        ("-", "Both read the **same observation**; only the decision rule differs.")], size=16)
    S.append(s)

    # ------------------------------------------------------------ Part B
    s = Slide("Part B · Stonefish and the Orca Model")
    s.image(F("stonefish_orca_view.png"), 0.5, 1.7, 6.6, 4.1)
    s.caption(0.5, 5.85, 6.6, "The Orca hull in Stonefish (main hull selected: 36.6 t shell)")
    s.card(7.4, 1.75, 5.4, 2.3, "Thrusters", [
        "Main thruster (±40 kN)",
        "Yaw thruster at the bow (±8 kN)",
        "Heave thruster through the centre of gravity (±55 kN)"], color=NAVY, tint=TINT_N, size=15)
    s.card(7.4, 4.25, 5.4, 2.35, "Sensors", [
        "Inertial unit · Doppler velocity log · pressure · GPS",
        "Profiling sonar: one swept beam, 160°, 30 m (obstacles)",
        "Multibeam sonar: 120° downward fan (seabed)"], color=TEAL, tint=TINT_G, size=15)
    S.append(s)

    s = Slide("The Base World Now Has Obstacles")
    s.image(F("14_rl_world_layout.png"), 0.4, 1.6, 5.5, 5.1)
    s.text(6.2, 1.7, 6.7, 3.0, [
        ("h", "Obstacle world (generated procedurally)"),
        ("-", "12 rock pinnacles and 2 ridge walls rising from the 30 m seabed through the 12 m cruise depth"),
        ("-", "2 large boulders, 3 fish schools (9 fish, about 2 m/s loops)"),
        ("-", "A traffic submarine crossing the lanes with its own avoidance")], size=16)
    s.image(F("stonefish_survey_2.png"), 6.3, 4.5, 3.6, 2.1)
    s.text(10.1, 4.55, 2.7, 2.0, ["The Orca only learns about any of this through its own sonars and DVL."],
           size=15, color=SLATE)
    S.append(s)

    s = Slide("Fix 1: The Original Orca Capsized")
    s.image(F("11_trim.png"), 0.4, 1.65, 8.3, 3.6)
    s.text(8.9, 1.75, 3.9, 4.8, [
        ("-", "Stonefish's own mass properties: the solid sail put the **centre of gravity 5 cm above the centre of buoyancy**."),
        ("-", "Every scenario, the original survey demo included, rolled past 170° within about 35 s."),
        ("-", "Fix: foam sail and mast; the removed mass carried as low **keel ballast**, placed so the two centres line up."),
        ("-", "Same net buoyancy; gravity now 8 cm below buoyancy. Same fix for the survey subs.")], size=15)
    s.text(0.6, 5.4, 8.0, 1.2, ["Verified: the unchanged survey autopilot now completes its lanes with roll ≈ 0° and "
                                "pitch ≤ 10°."], size=16, color=SLATE)
    S.append(s)

    s = Slide("Fixes 2-4 Found While Integrating")
    s.table(0.6, 1.75, 12.1, [
        ["Problem", "Root cause (from the Stonefish source)", "Fix"],
        ["DVL never had bottom lock", "only a velocity range was given, leaving the altitude limit at about 10³⁰; the beam search then ran zero steps", "explicit altitude range in the vehicle description"],
        ["Seabed coverage stopped after one fan", "the sensor interface returns sample copies whose identifier is always zero", "detect new samples by their time stamp"],
        ["Windowless mode aborted", "lights are rejected when running without a window", "remove lights after loading the scene"],
        ["Orca flung out of the water", "scripted fish squeeze the hull into the seabed or another hull; the physics engine diverges", "watchdog: put the Orca back ~2 s earlier and report it"],
        ["Two simulators on one network", "interleaved sensor data, jumping estimate", "warning in the run script; separate ROS domain per run"],
    ], col_w=[3.0, 5.4, 3.7], size=14, row_h=0.8)
    S.append(s)

    s = Slide("System Identification on the Stonefish Orca")
    s.image(F("10_sysid.png"), 0.4, 1.65, 8.4, 3.0)
    s.text(0.6, 4.8, 8.2, 1.8, [
        ("-", "Raw thrust steps sent over ROS 2, with the vehicle reset between steps."),
        ("-", "The published Orca envelope (4.1 m/s, 3°/s) is **not achievable** by this model: 1.15 m/s at 40 kN.")], size=16)
    s.stat(9.2, 1.8, 3.5, 1.25, "1.15 m/s", "top speed (40 kN)", color=NAVY, tint=TINT_N)
    s.stat(9.2, 3.55, 3.5, 1.25, "2.7 °/s", "max yaw rate at 0.5 m/s", color=NAVY, tint=TINT_N)
    s.stat(9.2, 5.3, 3.5, 1.0, "τ ≈ 6.5 s", "surge time constant", color=NAVY, tint=TINT_N)
    S.append(s)

    s = Slide("From RL Action to Thruster Force")
    s.image(F("12_setpoint_tracking.png"), 0.5, 1.7, 6.4, 3.1)
    s.text(7.2, 1.75, 5.6, 4.8, [
        ("h", "Command adapter (C++, 300 Hz)"),
        ("-", "Surge: F = 9000·u|u| + 23500·u feed-forward + PI on DVL speed"),
        ("-", "Yaw: 150 000·r feed-forward + PI on the measured yaw rate"),
        ("-", "Depth: integrates the depth-rate command into a reference, held with the autopilot's gains + seabed guard"),
        ("-", "Drives the **existing** vehicle controller with its autopilot switched off"),
        ("-", "3 s watchdog: stop and hold depth if commands stop")], size=15)
    s.text(0.6, 4.95, 6.3, 1.6, ["Actions are not forces: the policy asks for physical setpoints and the C++ inner "
                                 "loop turns them into thrust using the identified model."], size=15, color=SLATE)
    S.append(s)

    # ------------------------------------------------------------ Part C
    s = Slide("Part C · How RL and Stonefish Connect")
    s.image(F("13_architecture.png"), 0.4, 1.65, 12.5, 5.0)
    S.append(s)

    s = Slide("From Gazebo Assumptions to Stonefish")
    s.table(0.5, 1.7, 12.3, [
        ["Old assumption (Gazebo / UUV Simulator)", "Stonefish replacement"],
        ["Ground-truth pose fed into the control loop", "simulator odometry, labelled ground truth; used only for evaluation or in debug mode"],
        ["Depth = (pressure - 1 atm) / ρg", "Stonefish reports gauge pressure: depth = pressure / ρg (else 10 m error)"],
        ["Velocity command to a simulator plugin", "speed / yaw-rate / depth-rate setpoint, converted to thruster forces in C++"],
        ["Per-thruster plugin messages", "direct thruster forces, kept for manual tests"],
        ["Fixed heading, zero velocity placeholders", "measured heading and DVL velocity into the particle filter"],
        ["Simulator reset / pause services", "the robot is re-spawned at its start pose on the physics thread"],
        ["Spawning the model into the world", "the Orca is loaded from the scenario description"],
        ["Procedural risk map", "risk from the profiling sonar and DVL altitude; terrain richness from the multibeam"],
    ], col_w=[4.8, 7.5], size=14, row_h=0.56)
    S.append(s)

    s = Slide("Data Exchanged over ROS 2")
    s.table(0.5, 1.7, 7.5, [
        ["Data", "Direction", "Rate"],
        ["Motion setpoint (speed, yaw rate, depth rate)", "RL -> simulator", "1 Hz"],
        ["Attitude and turn rate (inertial unit)", "simulator -> RL", "100 Hz"],
        ["Velocity over ground, altitude (DVL)", "simulator -> RL", "10 Hz"],
        ["Depth (pressure)", "simulator -> RL", "10 Hz"],
        ["Position fix (GPS, surface only)", "simulator -> RL", "1 Hz"],
        ["Profiling and multibeam sonar scans", "simulator -> RL", "10 / 2 Hz"],
        ["Contact events", "simulator -> RL", "10 Hz"],
        ["Simulation clock", "simulator -> RL", "100 Hz"],
        ["Reset, autopilot on/off", "service", "on request"],
    ], col_w=[4.1, 2.2, 1.2], size=13.5, row_h=0.49)
    s.card(8.3, 1.75, 4.5, 2.25, "Frames (checked in the source)", [
        "World north-east-down, body forward-right-down",
        "The Python simulator already uses the same axes",
        "+ yaw = bow to starboard; depth positive down"], color=NAVY, tint=TINT_N, size=14)
    s.card(8.3, 4.2, 4.5, 2.4, "Threads and timing", [
        "Physics thread: 300 Hz step callback",
        "ROS 2 on its own thread",
        "Only lock-protected data is shared",
        "RL at 1 Hz on simulation time"], color=TEAL, tint=TINT_G, size=14)
    S.append(s)

    s = Slide("Navigation: Realistic vs. Debug")
    s.image(F("16_nav_error.png"), 0.5, 1.7, 7.4, 3.2)
    s.card(8.2, 1.75, 4.6, 2.3, "Realistic sensing (default)", [
        "DVL x inertial dead reckoning at 10 Hz",
        "Existing particle filter, GPS at the surface",
        "Ground truth logged only for evaluation"], color=TEAL, tint=TINT_G, size=15)
    s.card(8.2, 4.25, 4.6, 2.3, "Debug: ground truth", [
        "Simulator pose used as the estimate",
        "Connectivity checks only",
        "Warns in the log every 10 s"], color=ORANGE, tint=TINT_O, size=15)
    s.text(0.6, 5.0, 7.3, 1.5, ["Under 1 m mean error over 25-minute missions in the obstacle and multi-sub worlds. "
                                "The survey-world error comes from physics resets, which navigation cannot observe."],
           size=15, color=SLATE)
    S.append(s)

    s = Slide("Closed-Loop Results in Stonefish")
    s.image(F("18_stonefish_results.png"), 0.4, 1.65, 12.5, 4.25)
    s.text(0.6, 5.95, 12.2, 0.7, ["RL: realistic navigation, 30-minute limit. Autopilot: the original C++ controller "
                                  "using ground-truth position, in the same worlds."], size=15, color=SLATE)
    S.append(s)

    s = Slide("Obstacle World: Full Survey Under RL Control")
    s.image(F("15_stonefish_run_rl.png"), 0.4, 1.55, 6.4, 5.15)
    s.stat(7.0, 1.8, 2.75, 1.2, "14 / 15", "waypoints (1 skipped: on a rock)", color=TEAL)
    s.stat(10.0, 1.8, 2.75, 1.2, "0.64 m", "mean navigation error", color=TEAL)
    s.text(7.0, 3.5, 5.8, 3.0, [
        ("-", "Detours around pinnacles, ridges and boulders on every lane."),
        ("-", "6 static contacts are grazes in tight passes; 5 are scripted fish crossing the hull."),
        ("-", "A waypoint the sonar shows to be occupied is skipped, as a real mission manager would.")], size=16)
    S.append(s)

    s = Slide("What the Policy Sensed and Did")
    s.image(F("17_clearance_timeline.png"), 0.4, 1.65, 12.5, 4.1)
    s.text(0.6, 5.85, 12.2, 0.8, ["Speed drops whenever forward clearance shrinks; the shield fires only in brief bursts; "
                                  "the flank memory keeps a rock alongside until the hull has passed it."], size=15, color=SLATE)
    S.append(s)

    s = Slide("Other Submarines and the Dense Survey Field")
    s.image(F("15_stonefish_run_multisub.png"), 0.3, 1.55, 4.3, 5.15)
    s.image(F("15_stonefish_run_survey.png"), 4.6, 1.55, 4.3, 5.15)
    s.text(9.0, 1.7, 3.9, 4.9, [
        ("h", "Multi-sub world: complete"),
        ("-", "8/9 waypoints, 4 grazes with another sub, 0.69 m navigation error"),
        ("h", "Survey world: the open problem"),
        ("-", "90 x 70 m box for a 26 m hull, lanes 23 m apart"),
        ("-", "Large scripted fish push the hull into the seabed: 15 physics resets, estimate drifts, legs time out"),
        ("-", "The C++ autopilot also only reaches 3/12 there")], size=14)
    S.append(s)

    s = Slide("Interim Summary / Conclusion")
    s.stat(0.7, 1.8, 2.8, 1.25, "ROS 2", "RL policy <-> Stonefish", color=GREEN)
    s.stat(3.75, 1.8, 2.8, 1.25, f"{rl_ev['goal_fraction'] * 100:.0f} %", f"offline goals (planner {pl_ev['goal_fraction'] * 100:.0f} %)", color=TEAL)
    s.stat(6.8, 1.8, 2.8, 1.25, f"{rl_ev['collision_rate'] * 100:.0f} %", f"offline collisions (planner {pl_ev['collision_rate'] * 100:.0f} %)", color=TEAL)
    s.stat(9.85, 1.8, 2.8, 1.25, "2 / 3", "Stonefish surveys completed", color=NAVY, tint=TINT_N)
    s.text(0.7, 3.65, 12.0, 3.0, [
        ("-", "The policy drives the real Stonefish thrusters through ROS 2 and reads only real Stonefish sensors."),
        ("-", "The objective rewards progress and waypoints and penalises uncertainty, risk, energy, shield use and collisions."),
        ("-", "Three vehicle-model bugs were found and fixed: capsizing, DVL lock, seabed coverage."),
        ("-", "Remaining gap: dense scenes with non-yielding scripted fish, and grazing contacts in tight passes.")], size=17)
    S.append(s)

    s = Slide("Limitations and Completion Plan")
    s.card(0.6, 1.75, 6.0, 4.8, "Limitations", [
        "Trained in a fast simulator calibrated to Stonefish, not in Stonefish itself",
        "Flat seabeds: terrain-aided navigation carries no information",
        "Scripted fish never yield and can force physics resets",
        "The swept sonar refreshes each bearing only every ~6.4 s",
        "Contacts still occur: 6 grazes in the obstacle world"], color=ORANGE, tint=TINT_O, size=16)
    s.card(6.8, 1.75, 6.0, 4.8, "Next phase", [
        "Fine-tune with ES directly in Stonefish, faster than real time, with automatic resets",
        "Randomise obstacle fields for every mission",
        "Real bathymetry charts so terrain-aided navigation helps",
        "Larger flank margin in tight passes",
        "PPO once a deep-learning toolchain is available"], color=TEAL, tint=TINT_G, size=16)
    S.append(s)

    s = Slide("References")
    s.text(0.6, 1.7, 12.2, 4.9, [
        ("-", "T. Salimans et al., \"Evolution Strategies as a Scalable Alternative to Reinforcement Learning,\" arXiv:1703.03864, 2017."),
        ("-", "J. Schulman et al., \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017."),
        ("-", "T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control, Wiley, 2011."),
        ("-", "P. Cieślak, \"Stonefish: An Advanced Open-Source Simulation Tool Designed for Marine Robotics, with a ROS Interface,\" IEEE/OES OCEANS, 2019."),
        ("-", "M. M. M. Manhães et al., \"UUV Simulator: A Gazebo-Based Package for Underwater Intervention and Multi-Robot Simulation,\" IEEE/MTS OCEANS, 2016."),
        ("-", "S. Ross, G. Gordon, D. Bagnell, \"A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning,\" AISTATS, 2011."),
        ("-", "S. Macenski et al., \"Robot Operating System 2: Design, Architecture, and Uses in the Wild,\" Science Robotics, 2022.")],
           size=16)
    S.append(s)

    s = Slide("Thank you")
    s.text(0.6, 2.4, 12.2, 1.2, ["Questions?"], size=40, color=DKGREEN, align="ctr", anchor="ctr")
    S.append(s)
    return S


if __name__ == "__main__":
    build(deck())
