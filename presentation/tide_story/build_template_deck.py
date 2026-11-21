"""Build the 5-slide tide story deck on the NC A&T 'Spirited brand' template.

Usage: python build_template_deck.py template.pptx img_dir out.pptx
"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

TEMPLATE, IMG, OUT = sys.argv[1:4]
BLUE = RGBColor(0x00, 0x46, 0x84)      # template accent1
GOLD = RGBColor(0xFD, 0xB9, 0x27)      # template accent2
GREY = RGBColor(0x4C, 0x4C, 0x4C)      # template dk1
MUTED = RGBColor(0x66, 0x66, 0x66)     # template accent3
TINT = RGBColor(0xEA, 0xF0, 0xF7)      # light tint of the template blue
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation(TEMPLATE)

# drop the template's 11 sample slides (unreferenced parts are not written on save)
sld_ids = prs.slides._sldIdLst
for sid in list(sld_ids):
    prs.part.drop_rel(sid.rId)
    sld_ids.remove(sid)

LAYOUT = prs.slide_layouts.get_by_name("Content Slide with Bullet Color 1")
TOP = 2.2                                # content starts below title + subheader


def new_slide(subject, title, sub, notes):
    s = prs.slides.add_slide(LAYOUT)
    for ph in list(s.placeholders):
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.left, ph.top, ph.width, ph.height = Inches(0.75), Inches(1.05), Inches(11.85), Inches(0.55)
            ph.text_frame.text = title
            ph.text_frame.paragraphs[0].runs[0].font.size = Pt(28)
        elif idx == 13:
            ph.left, ph.top, ph.width, ph.height = Inches(0.75), Inches(1.6), Inches(11.85), Inches(0.4)
            ph.text_frame.text = sub
            ph.text_frame.paragraphs[0].runs[0].font.size = Pt(17)
        elif idx == 14:
            ph.text_frame.text = subject
        elif idx == 1:                    # body placeholder: we compose our own content
            ph._element.getparent().remove(ph._element)
    s.notes_slide.notes_text_frame.text = notes
    return s


def picture(s, name, x, y, w=None, h=None):
    path = f"{IMG}/{name}"
    iw, ih = Image.open(path).size
    if w is not None and h is not None:   # fit inside the box, keep aspect, centre it
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        x, y, w, h = x + (w - pw) / 2, y + (h - ph) / 2, pw, ph
    elif w is not None:
        h = w * ih / iw
    else:
        w = h * iw / ih
    return s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))


def box(s, x, y, w, h, fill=TINT):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, runs, size=14, color=GREY, bold=False, italic=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, bullets=False, space=6):
    """runs: a string, or a list of paragraphs (each a string)."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    paras = [runs] if isinstance(runs, str) else runs
    for i, t in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        r = p.add_run()
        r.text = ("•  " + t) if bullets else t
        f = r.font
        f.size, f.bold, f.italic = Pt(size), bold, italic
        f.color.rgb = color
    return tb


def stat(s, x, y, w, big, label, big_size=30, color=BLUE):
    text(s, x, y, w, 0.55, big, size=big_size, color=color, bold=True, anchor=MSO_ANCHOR.BOTTOM)
    text(s, x, y + 0.6, w, 0.6, label, size=13, color=GREY)


def marker(s, x, y, label, fill=BLUE, fg=WHITE, d=0.45):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.color.rgb = Pt(15), True, fg


# ---------------- 1. The question ----------------
s = new_slide("1 · The question", "When does the tide flood the ditch?",
              "Same tide gauge, same ditch, three events, but the ditch behaves differently every time",
              "The question: can the shape of the tide tell us what the roadside ditch will do? Three events at the same "
              "site (tide at 6202 Old Point Rd, ditch at 1235 Hillside Dr, Hanahan SC): a regular tide in Nov 2025 "
              "(hourly data), King tides in Dec 2025 (15 min) and Sep 2026 (5 min), all reduced to hourly information "
              "so the sampling cannot drive the comparison. Each row of the figure: the tide and the ditch over time, "
              "their Fourier spectra (12.4 h and 24-25 h tides; the ditch adds a 6.2 h overtide), and the ditch depth "
              "against the tide. The ditch fills once the tide rises above about 0.60 m NAVD88 (pooled), but the "
              "standing water it starts from changes: about 0.06 m in Dec, 0.16 m in Nov and 0.36 m in Sep 2026, when "
              "it never drained. That difference is the twist we come back to at the end.")
items = [("3 events", "Regular tide Nov 2025; King tides Dec 2025 and Sep 2026"),
         ("≈ 0.60 m", "tide level (NAVD88) above which the ditch fills"),
         ("0.06–0.36 m", "standing water in the ditch, different in every event"),
         ("+15 min", "the ditch follows the tide almost at once")]
for i, (big, lab) in enumerate(items):
    stat(s, 0.75, TOP + 0.05 + i * 1.12, 3.3, big, lab, big_size=26)
box(s, 4.3, TOP, 8.3, 4.5, fill=WHITE).line.color.rgb = TINT
picture(s, "F1_overview.png", 4.4, TOP + 0.05, w=8.1, h=4.4)

# ---------------- 2. Four ways ----------------
s = new_slide("2 · The method", "Four ways to read the shape of the tide",
              "Slide a 15 h window along the tide, turn it into a cloud of points, and describe the cloud",
              "Left: one window of the Sep 2026 King tide (yellow), its delay-embedded point cloud with two ellipses "
              "(FastTDA's fitted ellipse curve in red, the Gaussian 2-sigma ellipse dashed) and every feature over time. "
              "Each window gives three clouds: the tide alone, the ditch alone, and the tide-vs-ditch loop. Four families: "
              "topology (VR and Alpha H1), algebraic curves of genus 0 (an ellipse, FastTDA) and genus 1 (a cubic, the real "
              "form of an elliptic curve, which can hold an oval plus an open branch), and statistics (the bivariate "
              "Gaussian, i.e. covariance, ellipse). Window 14.8 h and delay 0.5 h come from the tide's own spectrum, as in "
              "the DROPBEAR method. Which one reads the tide best?")
picture(s, "F2_window_King_Tide_Sep_2026.png", 0.75, TOP, h=4.45)
fam = [("Topology", "VR H1 · Alpha H1", "Is there a hole in the cloud, and how big?"),
       ("Ellipse curve (genus 0)", "FastTDA", "Fit an ellipse through the points; how wide is it?"),
       ("Elliptic-type curve (genus 1)", "Cubic oval", "Fit a cubic curve; does it close into an oval?"),
       ("Statistics", "Bivariate Gaussian ellipse", "How spread out is the cloud, and in which direction?")]
for i, (head, feat, q) in enumerate(fam):
    x, y = 4.15 + (i % 2) * 4.3, TOP + (i // 2) * 1.75
    hero = i == 3
    box(s, x, y, 4.05, 1.55, fill=GOLD if hero else TINT)
    marker(s, x + 0.2, y + 0.2, str(i + 1), fill=BLUE)
    text(s, x + 0.8, y + 0.17, 3.1, 0.35, head, size=16, color=BLUE, bold=True)
    text(s, x + 0.8, y + 0.52, 3.1, 0.3, feat, size=13, color=BLUE if hero else MUTED, bold=True)
    text(s, x + 0.2, y + 0.92, 3.7, 0.55, q, size=13, color=GREY)
text(s, 4.15, TOP + 3.65, 8.35, 0.7,
     "Window 14.8 h and delay 0.5 h come from the tide's own spectrum (12.4 h tide, 4.9 h harmonic), as in the DROPBEAR method.",
     size=13, color=MUTED, italic=True)

# ---------------- 3. Contest ----------------
s = new_slide("3 · The contest", "Only the Gaussian ellipse works every time",
              "Three tests on three events: does it fit, follow the ditch, and tell the events apart?",
              "Fit failures: the Gaussian ellipse comes from the covariance of the points and never fails; FastTDA, which "
              "fits an ellipse curve, fails on 72% of the tide-ditch loops (a hockey-stick shape is not an ellipse). Link "
              "with ditch depth (Spearman, tide cloud): Gaussian minor axis -0.45 (Nov), -0.47 (Sep), -0.64 (Dec), the only "
              "feature consistent in all three; FastTDA close (-0.42 to -0.54); VR and Alpha H1 near zero in Sep. Telling the "
              "events apart (share of variance between events, loop cloud): Gaussian 0.54, Alpha 0.54, VR 0.51, genus-1 0.47, "
              "FastTDA 0.38. The genus-1 cubic fit closed no oval in Sep 2026 and has no link with the ditch.")
picture(s, "F3_feature_comparison.png", 0.75, TOP, w=7.6)
text(s, 0.75, TOP + 7.6 * 1320 / 2430 + 0.08, 7.6, 0.3,
     "Bars: each feature on the tide, ditch and tide–ditch loop clouds.", size=11, color=MUTED, italic=True)
for i, (big, lab) in enumerate([("0% failures", "FastTDA fails on 72% of tide–ditch windows"),
                                ("ρ −0.45 to −0.64", "in all 3 events; VR / Alpha H1 drop to 0 in Sep"),
                                ("η² = 0.54", "top event separation, tied with Alpha H1")]):
    y = TOP + i * 1.25
    box(s, 8.65, y, 3.95, 1.1)
    stat(s, 8.9, y + 0.0, 3.5, big, lab, big_size=24)
box(s, 8.65, TOP + 3.75, 3.95, 0.75, fill=GOLD)
text(s, 8.85, TOP + 3.75, 3.6, 0.75, "Genus-1 found no oval in Sep 2026 and no signal in any event",
     size=13, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)

# ---------------- 4. Why ----------------
s = new_slide("4 · Why it wins", "Why it wins: it measures connectivity",
              "Tide against ditch: a connected ditch makes a thin line, a disconnected one a round blob",
              "On the tide-vs-ditch plot (each scaled within the window), a ditch that is flooded and connected moves with "
              "the tide, so the points form a thin line and the ellipse is thin (left: a King tide window, minor axis 1.10, "
              "ratio 0.38). A ditch that stays at its base level while the tide swings gives a round, scattered cloud "
              "(right: a Regular tide window, minor axis 2.23, ratio 0.80). So the ellipse is an index of hydraulic "
              "connectivity. Both King tides give thin loops (thickness about 1.4, ratio about 0.4); the Regular tide "
              "gives round ones (2.0, 0.73). Persistent homology looks for a hole and curve fits look for a curve; this "
              "cloud is a hockey stick, so those methods have little to grip. Caveat: the Nov ditch base level is "
              "noisier, which also thickens its loops.")
picture(s, "F5_bvg_loop_meaning.png", 0.75, TOP, h=4.45)
text(s, 7.35, TOP, 5.25, 2.2, [
    "Flooded ditch: depth moves with the tide → points on a line → thin ellipse",
    "Disconnected ditch: depth stays flat while the tide moves → round ellipse",
    "Topology needs a real hole and curve fits need a real curve; the tide–ditch cloud has neither"],
    size=14, bullets=True, space=8)
rows = [["Tide–ditch loop", "Nov 25", "Dec 25", "Sep 26"],
        ["Standing water", "0.16 m", "0.06 m", "0.36 m"],
        ["Ellipse thickness", "2.01", "1.40", "1.39"],
        ["Axis ratio", "0.73", "0.44", "0.40"]]
tbl = s.shapes.add_table(4, 4, Inches(7.35), Inches(TOP + 2.35), Inches(5.25), Inches(1.6)).table
for j, wd in enumerate([2.05, 1.06, 1.07, 1.07]):
    tbl.columns[j].width = Inches(wd)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = tbl.cell(i, j)
        c.text = v
        c.fill.solid()
        c.fill.fore_color.rgb = BLUE if i == 0 else (TINT if i % 2 else WHITE)
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        f = p.runs[0].font
        f.size, f.bold = Pt(13), i == 0
        f.color.rgb = WHITE if i == 0 else GREY
text(s, 7.35, TOP + 4.05, 5.25, 0.4, "Per-cycle median. Both King tides flood the ditch fully → thin loops.",
     size=11, color=MUTED, italic=True)

# ---------------- 5. Payoff ----------------
s = new_slide("5 · The payoff", "Payoff: better predictions on new events",
              "Predict the ditch depth from the tide alone, with and without the shape of the tide",
              "Ditch depth predicted from the tide only, window 18.5 h and delay 1.5 h. Held-out days: threshold model + "
              "tide memory 0.108 m; adding the Gaussian ellipse or FastTDA brings it to 0.100 m (+7%); gradient boosting "
              "with memory 0.082 m, with the ellipse 0.077 m (+6.5%). Held-out event (train on two events, predict the "
              "third): much harder (about 0.15-0.17 m) because each event starts from different standing water; the "
              "FastTDA/Gaussian hybrid gives +6.6% on the threshold model and the shape features +6-7% on gradient "
              "boosting. Across both window settings and both validations, the Gaussian ellipse improved the prediction "
              "in all 8 tests (+2% to +7%); genus-1 never helped (-1% to +1%); VR and Alpha H1 were mixed. Next step: give "
              "the model the ditch's standing water at the start of each event, the main remaining error.")
picture(s, "F4_prediction_tau1.5h.png", 0.75, TOP, w=7.4)
text(s, 0.75, TOP + 7.4 * 1404 / 2550 + 0.08, 7.4, 0.3,
     "Window 18.5 h, delay 1.5 h. Left: held-out days. Right: a whole held-out tide event.", size=11, color=MUTED, italic=True)
stat(s, 8.5, TOP, 1.9, "8 / 8", "tests improved by the Gaussian ellipse", big_size=30)
stat(s, 10.6, TOP, 2.0, "+6.6%", "ellipse hybrid on an unseen tide event", big_size=30)
text(s, 8.5, TOP + 1.45, 4.1, 1.7, [
    "Read the tide with the Gaussian ellipse; keep topology for real loops",
    "Genus-1 curves: a negative result",
    "Next: give the model each event's standing water, the biggest remaining error"],
    size=14, bullets=True, space=6)
box(s, 8.5, TOP + 3.35, 4.1, 1.05, fill=GOLD)
text(s, 8.65, TOP + 3.35, 3.8, 1.05, "Gaussian ellipse > FastTDA > VR / Alpha H1 >> genus-1",
     size=16, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)

# ---------------- Title page (added last, moved to the front) ----------------
t = prs.slides.add_slide(prs.slide_layouts.get_by_name("Title Slide"))
fields = {
    0: "Reading the shape of the tide: Gaussian ellipse vs TDA vs elliptic curves",
    10: "When does the tide flood the ditch? A five-part story",
    11: "Question · Method · Contest · Why it wins · Payoff",
    12: "Tide (6202 Old Point Rd) → ditch (1235 Hillside Dr), Hanahan SC",
    13: "Three tide events: Nov 2025, Dec 2025, Sep 2026",
}
for ph in t.placeholders:
    idx = ph.placeholder_format.idx
    tf = ph.text_frame
    if idx == 0:                       # two lines: topic, then the comparison
        tf.text = "Reading the shape of the tide"
        p = tf.add_paragraph()
        p.add_run().text = "Gaussian ellipse vs TDA vs elliptic curves"
        tf.paragraphs[0].runs[0].font.size = Pt(48)
        p.runs[0].font.size = Pt(32)
    else:                              # widen the info lines so each stays on one line
        ph.left, ph.top, ph.width, ph.height = ph.left, ph.top, Inches(11.0), ph.height
        tf.text = fields[idx]
t.notes_slide.notes_text_frame.text = (
    "This talk asks one question: when does the tide flood the roadside ditch, and can the shape of the tide tell us? "
    "We compare four ways of reading the shape of the tide in sliding windows: topology (VR and Alpha H1 persistence), "
    "a genus-0 ellipse curve (FastTDA), a genus-1 elliptic-type cubic curve, and the bivariate Gaussian ellipse. "
    "Story: 1 the question and the data, 2 the four methods, 3 the contest over three tide events, 4 why the Gaussian "
    "ellipse wins (it measures how connected the ditch is to the tide), 5 the payoff in better predictions.")
ids = prs.slides._sldIdLst
last = ids[-1]
ids.remove(last)
ids.insert(0, last)

prs.save(OUT)
print("wrote", OUT)
