const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/315ab91f-7e57-4858-9715-863ea6106349_b6f21924-0d6d-4aae-b475-447dd4274c7e/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "Tide_shape_story.pptx";

// Estuary navy and tidal teal; orange = the Gaussian ellipse, the hero of the story.
const THEME = {
  name: "Tide Estuary",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B2E3C", lt1: "FFFFFF", dk2: "1F4A5A", lt2: "EEF3F5",
    accent1: "1B7F8E", accent2: "E07A2F", accent3: "6B8A99",
    accent4: "F2B33D", accent5: "5E6B73", accent6: "BFD3DA",
    hlink: "1B7F8E", folHlink: "5E6B73",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Reading the shape of the tide: Gaussian ellipse vs TDA vs genus";
const C = pres.SchemeColor;

const FOOT = { x: 0.6, y: 7.0, w: 9.5, h: 0.3, fontSize: 10, margin: 0, valign: "middle" };
const NUM = { x: 12.1, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, align: "right", margin: 0, valign: "middle" };
const FOOT_TEXT = "Tide (6202 Old Point Rd) → ditch (1235 Hillside Dr), Hanahan SC  ·  Nov 2025, Dec 2025, Sep 2026";

for (const [title, bg, fg, muted] of [["DARK_TITLE_ONLY", C.text1, C.background1, C.accent6],
                                       ["TITLE_ONLY", C.background1, C.text1, C.accent5]]) {
  pres.defineSlideMaster({
    title,
    background: { color: bg },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontSize: 34,
          bold: true, color: fg, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: FOOT_TEXT, options: { ...FOOT, color: muted } } },
    ],
    slideNumber: { ...NUM, color: muted },
  });
}

const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.18 });
function card(slide, name, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill || C.background1 },
    line: { color: fill || C.background1, width: 0 }, shadow: shadow(), objectName: name });
}
function image(slide, name, path, x, y, w, ratio) {           // ratio = height / width
  slide.addImage({ path, x, y, w, h: w * ratio, objectName: name });
}
function stat(slide, name, x, y, w, big, label, bigColor, labelColor, bigSize) {
  slide.addText(big, { x, y, w, h: 0.62, fontSize: bigSize || 32, bold: true, fontFace: THEME.headFontFace,
    color: bigColor, margin: 0, valign: "bottom", isTextBox: true, objectName: name + " value" });
  slide.addText(label, { x, y: y + 0.66, w, h: 0.62, fontSize: 13, color: labelColor, margin: 0, valign: "top",
    isTextBox: true, objectName: name + " label" });
}
function subtitle(slide, text, color) {
  slide.addText(text, { x: 0.6, y: 1.13, w: 12.1, h: 0.38, fontSize: 15, italic: true, color, margin: 0,
    isTextBox: true, objectName: "Story line" });
}
function bullets(items, opts) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 7, ...opts } }));
}
function table(slide, name, rows, x, y, w, colW, rowH, hiRow) {
  slide.addTable(rows.map((r, i) => r.map((t, j) => ({ text: t, options: {
    bold: i === 0 || i === hiRow, color: i === 0 ? HEX.lt1 : HEX.dk1,
    fill: { color: i === 0 ? HEX.dk2 : (i === hiRow ? "FCE9DC" : (i % 2 ? HEX.lt2 : HEX.lt1)) },
    fontSize: 12, align: j === 0 ? "left" : "center", valign: "middle" } }))),
    { x, y, w, colW, rowH, border: { type: "solid", pt: 0.5, color: HEX.accent6 }, objectName: name });
}

// ---------------- 1. The puzzle ----------------
pres.addSection({ title: "1 Puzzle" });
let s = pres.addSlide({ masterName: "DARK_TITLE_ONLY", sectionTitle: "1 Puzzle" });
s.addText("When does the tide flood the ditch?", { placeholder: "title" });
subtitle(s, "Same tide gauge, same ditch, three tide events - but the ditch behaves differently every time.", C.accent6);
stat(s, "Events", 0.6, 1.7, 3.2, "3 tide events", "Regular tide Nov 2025, King tides Dec 2025 and Sep 2026", C.accent4, C.background1, 28);
stat(s, "Threshold", 0.6, 3.05, 3.2, "≈ 0.60 m", "tide level (NAVD88) above which the ditch fills", C.accent4, C.background1);
stat(s, "Standing water", 0.6, 4.4, 3.2, "0.06–0.36 m", "standing water in the ditch differs by event (Dec, Nov, Sep)", C.accent4, C.background1, 28);
stat(s, "Lag", 0.6, 5.75, 3.2, "+15 min", "the ditch follows the tide almost at once", C.accent4, C.background1);
card(s, "Overview card", 4.15, 1.62, 8.55, 5.2, C.background1);
image(s, "Overview figure", "img/F1_overview.png", 4.65, 1.68, 7.3, 1890 / 2700);
s.addNotes("The question: can the shape of the tide tell us what the roadside ditch will do? Three events at the same site: a regular tide in Nov 2025 (hourly data), King tides in Dec 2025 (15 min) and Sep 2026 (5 min); all reduced to hourly information so the sampling cannot drive the comparison. Each row of the figure: the tide and the ditch over time, their Fourier spectra (12.4 h and 24-25 h tides, the ditch adds a 6.2 h overtide), and the ditch depth against the tide. The ditch fills once the tide rises above roughly 0.60 m NAVD88 (pooled), but the standing water it starts from changes: about 0.06 m in Dec, 0.16 m in Nov and 0.36 m in Sep 2026, when it never drained. That event-to-event difference is the twist we come back to at the end.");

// ---------------- 2. Four ways to read a shape ----------------
pres.addSection({ title: "2 Method" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "2 Method" });
s.addText("Four ways to read the shape of the tide", { placeholder: "title" });
subtitle(s, "Slide a 15 h window along the tide, turn it into a cloud of points, and describe the cloud.", C.accent5);
card(s, "Window card", 0.6, 1.62, 3.55, 5.2, C.background1);
image(s, "Window figure", "img/F2_window_King_Tide_Sep_2026.png", 0.72, 1.7, 3.31, 1800 / 1200);
const fam = [
  ["Topology", "VR H1 · Alpha H1", "Is there a hole in the cloud, and how big?", C.accent3],
  ["Ellipse curve (genus 0)", "FastTDA", "Fit an ellipse through the points; how wide is it?", C.accent1],
  ["Elliptic-type curve (genus 1)", "Cubic oval", "Fit a cubic curve; does it close into an oval?", C.accent5],
  ["Statistics", "Bivariate Gaussian ellipse", "How spread out is the cloud, and in which direction?", C.accent2],
];
fam.forEach(([head, feat, q, col], i) => {
  const x = 4.5 + (i % 2) * 4.15, y = 1.62 + Math.floor(i / 2) * 2.0;
  card(s, `Family ${i + 1}`, x, y, 3.95, 1.8, C.background2);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.27, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col },
    objectName: `Family ${i + 1} marker` });
  s.addText(String(i + 1), { x: x + 0.25, y: y + 0.27, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1,
    align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `Family ${i + 1} number` });
  s.addText(head, { x: x + 0.9, y: y + 0.2, w: 2.9, h: 0.4, fontSize: 16, bold: true, fontFace: THEME.headFontFace,
    color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `Family ${i + 1} name` });
  s.addText(feat, { x: x + 0.9, y: y + 0.6, w: 2.9, h: 0.3, fontSize: 12, color: col, bold: true, margin: 0,
    isTextBox: true, objectName: `Family ${i + 1} feature` });
  s.addText(q, { x: x + 0.25, y: y + 1.0, w: 3.5, h: 0.7, fontSize: 13, color: C.text1, margin: 0, valign: "top",
    isTextBox: true, objectName: `Family ${i + 1} question` });
});
s.addText("Window 14.8 h and delay 0.5 h come from the tide's own spectrum (12.4 h tide, 4.9 h harmonic), as in the DROPBEAR method.",
  { x: 4.5, y: 5.75, w: 8.1, h: 0.9, fontSize: 13, color: C.accent5, italic: true, margin: 0, valign: "top", isTextBox: true,
    objectName: "Window note" });
s.addNotes("Left: one window of the Sep 2026 King tide (yellow), its delay-embedded point cloud with two ellipses - FastTDA's fitted ellipse curve (red) and the Gaussian 2-sigma ellipse (dashed) - and every feature over time. Each window gives three clouds: the tide alone, the ditch alone, and the tide-vs-ditch loop. Four families of shape description: topology (VR and Alpha H1, persistent homology of the cloud), algebraic curves of genus 0 (an ellipse, FastTDA) and genus 1 (a cubic, the real form of an elliptic curve, which can hold an oval plus an open branch), and statistics (the bivariate Gaussian, i.e. covariance, ellipse). The question: which one reads the tide best?");

// ---------------- 3. The contest ----------------
pres.addSection({ title: "3 Contest" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "3 Contest" });
s.addText("Only the Gaussian ellipse works in every event", { placeholder: "title" });
subtitle(s, "Tested on all three events: does it fit, does it follow the ditch, does it tell the events apart?", C.accent5);
card(s, "Comparison card", 0.6, 1.62, 7.7, 4.4, C.background1);
image(s, "Comparison figure", "img/F3_feature_comparison.png", 0.7, 1.7, 7.5, 1320 / 2430);
s.addText("Bars: each feature on the tide, ditch and tide–ditch loop clouds.", { x: 0.7, y: 5.62, w: 7.5, h: 0.3, fontSize: 11,
  italic: true, color: C.accent5, margin: 0, isTextBox: true, objectName: "Comparison caption" });
const sx = 8.65, sw = 4.05;
card(s, "Stat card 1", sx, 1.62, sw, 1.35, C.background2);
stat(s, "Never fails", sx + 0.25, 1.62, sw - 0.5, "0% failures", "FastTDA fails on 72% of tide–ditch windows", C.accent2, C.text1, 26);
card(s, "Stat card 2", sx, 3.14, sw, 1.35, C.background2);
stat(s, "Follows the ditch", sx + 0.25, 3.14, sw - 0.5, "ρ −0.45 to −0.64", "in all 3 events; VR/Alpha H1 drop to 0 in Sep", C.accent2, C.text1, 26);
card(s, "Stat card 3", sx, 4.66, sw, 1.35, C.background2);
stat(s, "Separates events", sx + 0.25, 4.66, sw - 0.5, "η² = 0.54", "top event separation, tied with Alpha H1", C.accent2, C.text1, 26);
s.addText("Genus-1 (cubic) found no oval at all in Sep 2026 and carried no signal in any event.",
  { x: 0.6, y: 6.2, w: 12.1, h: 0.45, fontSize: 14, bold: true, color: C.text1, margin: 0, isTextBox: true,
    objectName: "Genus verdict" });
s.addNotes("The contest over three events. Fit failures: the Gaussian ellipse is computed from the covariance of the points and never fails; FastTDA, which fits an ellipse curve, fails on 72% of the tide-ditch loops (a hockey-stick shape is not an ellipse). Link with the ditch depth (Spearman, tide cloud): Gaussian minor axis -0.45 (Nov), -0.47 (Sep), -0.64 (Dec) - the only feature consistent in all three; FastTDA close (-0.42 to -0.54); VR and Alpha H1 near zero in Sep. Telling the events apart (share of variance between events, loop cloud): Gaussian 0.54, Alpha 0.54, VR 0.51, genus-1 0.47, FastTDA 0.38. The genus-1 cubic fit closed no oval in Sep 2026 and has no link with the ditch.");

// ---------------- 4. Why it wins ----------------
pres.addSection({ title: "4 Why" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "4 Why" });
s.addText("Why it wins: the ellipse measures connectivity", { placeholder: "title" });
subtitle(s, "Plot the tide against the ditch: connected ditch = a thin line, disconnected ditch = a round blob.", C.accent5);
card(s, "Meaning card", 0.6, 1.62, 7.3, 5.2, C.background1);
image(s, "Connectivity figure", "img/F5_bvg_loop_meaning.png", 0.82, 1.68, 6.86, 1360 / 1870);
s.addText(bullets([
  "Flooded ditch: depth moves with the tide → points on a line → thin ellipse",
  "Disconnected ditch: depth stays flat while the tide moves → round ellipse",
  "Topology needs a real hole; curve fits need a real curve - the tide–ditch cloud has neither, so they struggle",
], { fontSize: 14, color: C.text1 }), { x: 8.25, y: 1.62, w: 4.45, h: 2.75, margin: 0, valign: "top", isTextBox: true,
  objectName: "Why bullets" });
table(s, "Event table", [
  ["Tide–ditch loop", "Nov 25", "Dec 25", "Sep 26"],
  ["Standing water", "0.16 m", "0.06 m", "0.36 m"],
  ["Ellipse thickness", "2.01", "1.40", "1.39"],
  ["Axis ratio", "0.73", "0.44", "0.40"],
], 8.25, 4.55, 4.45, [1.75, 0.9, 0.9, 0.9], 0.45, 0);
s.addText("Per-cycle median. Both King tides flood the ditch fully → thin loops.", { x: 8.25, y: 6.4, w: 4.45, h: 0.35,
  fontSize: 11, italic: true, color: C.accent5, margin: 0, isTextBox: true, objectName: "Table caption" });
s.addNotes("Why the Gaussian ellipse wins. On the tide-vs-ditch plot (each scaled within the window), a ditch that is flooded and connected moves with the tide, so the points form a thin line and the ellipse is thin (left: a King tide window, minor axis 1.10, ratio 0.38). A ditch that stays at its base level while the tide swings gives a round, scattered cloud (right: a Regular tide window, minor axis 2.23, ratio 0.80). So the ellipse is an index of hydraulic connectivity. Both King tides (Dec 2025 and Sep 2026) give thin loops (thickness about 1.4, ratio about 0.4); the Regular tide gives round ones (2.0, 0.73). Persistent homology looks for a hole and curve fits look for a curve; this cloud is a hockey stick, so those methods have little to grip. Caveat: the Nov ditch base level is noisier, which also thickens its loops.");

// ---------------- 5. Payoff ----------------
pres.addSection({ title: "5 Payoff" });
s = pres.addSlide({ masterName: "DARK_TITLE_ONLY", sectionTitle: "5 Payoff" });
s.addText("The payoff: better predictions, even for a new event", { placeholder: "title" });
subtitle(s, "Predict the ditch depth from the tide alone - with and without the shape of the tide.", C.accent6);
card(s, "Prediction card", 0.6, 1.62, 7.6, 4.4, C.background1);
image(s, "Prediction figure", "img/F4_prediction_tau1.5h.png", 0.68, 1.68, 7.44, 1404 / 2550);
s.addText("Window 18.5 h, delay 1.5 h. Left: held-out days. Right: a whole held-out tide event.", { x: 0.7, y: 5.8, w: 7.4,
  h: 0.2, fontSize: 10, italic: true, color: C.accent5, margin: 0, isTextBox: true, objectName: "Prediction caption" });
stat(s, "Every test", 8.6, 1.62, 1.95, "8 / 8", "tests improved by the Gaussian ellipse", C.accent4, C.background1, 30);
stat(s, "New event", 10.75, 1.62, 1.95, "+6.6%", "ellipse hybrid on an unseen tide event", C.accent4, C.background1, 30);
s.addText("What the story says", { x: 8.6, y: 3.15, w: 4.1, h: 0.4, fontSize: 18, bold: true, fontFace: THEME.headFontFace,
  color: C.accent4, margin: 0, isTextBox: true, objectName: "Story header" });
s.addText(bullets([
  "Read the tide with the Gaussian ellipse; keep topology for real loops",
  "Genus-1 curves: a negative result",
  "Next: give the model each event's standing water - the biggest remaining error",
], { fontSize: 14, color: C.background1 }), { x: 8.6, y: 3.6, w: 4.1, h: 2.4, margin: 0, valign: "top", isTextBox: true,
  objectName: "Story bullets" });
s.addText("Gaussian ellipse > FastTDA > VR / Alpha H1 >> genus-1", { x: 0.6, y: 6.25, w: 12.1, h: 0.45, fontSize: 18, bold: true,
  fontFace: THEME.headFontFace, color: C.accent4, align: "center", margin: 0, isTextBox: true, objectName: "Ranking" });
s.addNotes("Ditch depth predicted from the tide only, with an 18.5 h window and 1.5 h delay. Held-out days: threshold model + tide memory 0.108 m; adding the Gaussian ellipse or FastTDA brings it to 0.100 m (+7%); gradient boosting with memory 0.082 m, with the ellipse 0.077 m (+6.5%). Held-out event (train on two events, predict the third): much harder (about 0.15-0.17 m) because each event starts from different standing water; the FastTDA/Gaussian hybrid gives +6.6% on the threshold model and the shape features +6-7% on gradient boosting, which alone does no better than the simple threshold model. Across both window settings and both validations, the Gaussian ellipse improved the prediction in all 8 tests (+2% to +7%); genus-1 never helped (-1% to +1%); VR and Alpha H1 were mixed. Next step: give the model the ditch's standing water at the start of each event, the main remaining error.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
