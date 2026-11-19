const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/315ab91f-7e57-4858-9715-863ea6106349_b6f21924-0d6d-4aae-b475-447dd4274c7e/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "Tide_TDA_findings.pptx";

// Deep estuary navy with tidal teal; orange marks the winning feature.
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
pres.title = "Tide boundary condition vs ditch depth: TDA, Gaussian and genus features";
const C = pres.SchemeColor;

const FOOT = { x: 0.6, y: 7.0, w: 9, h: 0.3, fontSize: 10, margin: 0, valign: "middle" };
const NUM = { x: 12.1, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, align: "right", margin: 0, valign: "middle" };
const FOOT_TEXT = "Tide (6202 Old Point Rd) vs ditch depth (1235 Hillside Dr), Hanahan SC  ·  Nov & Dec 2025";

for (const [title, bg, fg, muted] of [["DARK_TITLE_ONLY", C.text1, C.background1, C.accent6],
                                       ["TITLE_ONLY", C.background1, C.text1, C.accent5]]) {
  pres.defineSlideMaster({
    title,
    background: { color: bg },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 34,
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
  slide.addText(big, { x, y, w, h: 0.6, fontSize: bigSize || 30, bold: true, fontFace: THEME.headFontFace,
    color: bigColor, margin: 0, valign: "bottom", isTextBox: true, objectName: name + " value" });
  slide.addText(label, { x, y: y + 0.63, w, h: 0.62, fontSize: 13, color: labelColor, margin: 0, valign: "top",
    isTextBox: true, objectName: name + " label" });
}
function bullets(items, opts) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 7, ...opts } }));
}

// ---------------- 1. Data and system ----------------
pres.addSection({ title: "System" });
let s = pres.addSlide({ masterName: "DARK_TITLE_ONLY", sectionTitle: "System" });
s.addText("Tides flood the ditch above a 0.50 m level", { placeholder: "title" });
s.addText("Signal: tidal water surface elevation (BC data). Target: water depth in the roadside ditch (validation data).",
  { x: 0.6, y: 1.22, w: 12.1, h: 0.4, fontSize: 15, color: C.accent6, margin: 0, isTextBox: true, objectName: "Subtitle" });
stat(s, "Events", 0.6, 1.75, 3.0, "2 events", "Regular tide 13–23 Nov, King tide 1–5 Dec 2025", C.accent4, C.background1);
stat(s, "Flooding level", 0.6, 3.1, 3.0, "0.50 m", "learned flooding level (NAVD88) ≈ ditch bed at 0.48 m", C.accent4, C.background1);
stat(s, "Periods", 0.6, 4.45, 3.0, "12.4 h · 24 h", "semidiurnal and diurnal tides; ditch adds 6.2 h", C.accent4, C.background1, 26);
stat(s, "Lag", 0.6, 5.8, 3.0, "+0.25 h", "ditch follows the tide; slow drain-down after floods", C.accent4, C.background1);
card(s, "Overview card", 3.95, 1.75, 8.75, 4.95, C.background1);
image(s, "Overview figure", "img/F1_overview.png", 4.05, 1.9, 8.55, 1260 / 2700);
s.addText("Left: both records. Middle: Fourier spectra. Right: ditch depth vs tide, with the learned flooding level (dashed).",
  { x: 4.15, y: 6.05, w: 8.35, h: 0.5, fontSize: 11, italic: true, color: C.accent5, margin: 0, isTextBox: true,
    objectName: "Overview caption" });
s.addNotes("Two tide events at Hanahan SC. The boundary condition is the tide at 6202 Old Point Road (WSE, NAVD88); the target is the water depth in the ditch beside 1235 Hillside Dr. Both records are put on a 15-minute grid carrying hourly information (Nov is hourly, Dec 15-minute), so the events differ only in the tide, not the sampling. The ditch responds above a threshold: below about 0.50 m NAVD88 it stays at its base level, above it the depth follows the tide almost 1:1 (the ditch bed is at 0.48 m). After a flood it drains slowly (about 0.17 m plateau in the King tide). Tide peaks on 21-22 Nov (1.77, 2.01, 1.34 m) look like a surge the ditch did not follow.");

// ---------------- 2. Method ----------------
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Method" });
s.addText("Sliding windows turn the tide into shapes", { placeholder: "title" });
card(s, "Window card", 0.6, 1.4, 3.75, 5.4, C.background1);
image(s, "Window figure", "img/F2_window_King_Tide.png", 0.75, 1.48, 3.45, 1800 / 1200);
s.addText("Pipeline", { x: 4.75, y: 1.4, w: 7.9, h: 0.42, fontSize: 20, bold: true, fontFace: THEME.headFontFace,
  color: C.accent1, margin: 0, isTextBox: true, objectName: "Pipeline header" });
s.addText(bullets([
  "Fourier spectrum sets the window: min_f = 12.4 h tide, max_f = 4.9 h harmonic → 15 h window, 0.5 h delay (DROPBEAR formulas)",
  "Each trailing window becomes a point cloud: the tide (Takens), the ditch (Takens) and the tide–ditch loop",
  "Every cloud is scored by five shape features, then compared and used to predict the ditch depth",
], { fontSize: 14, color: C.text1 }), { x: 4.75, y: 1.85, w: 7.9, h: 1.75, margin: 0, valign: "top", isTextBox: true,
  objectName: "Pipeline steps" });
const rows = [
  ["Feature", "Family", "What it measures in the cloud"],
  ["VR H1 / Alpha H1", "Topology (TDA)", "size of the largest hole (loop)"],
  ["FastTDA", "Algebraic, genus 0", "minor axis of a fitted ellipse curve"],
  ["BV Gaussian", "Statistical", "covariance ellipse: thickness and axis ratio"],
  ["Cubic oval", "Algebraic, genus 1", "closed oval of a fitted cubic (elliptic-type) curve"],
];
s.addTable(rows.map((r, i) => r.map(t => ({ text: t, options: {
  bold: i === 0, color: i === 0 ? HEX.lt1 : HEX.dk1, fill: { color: i === 0 ? HEX.dk2 : (i % 2 ? HEX.lt2 : HEX.lt1) },
  fontSize: 13, valign: "middle" } }))), { x: 4.75, y: 3.75, w: 7.9, colW: [2.0, 2.0, 3.9], rowH: 0.5,
  border: { type: "solid", pt: 0.5, color: HEX.accent6 }, objectName: "Feature table" });
s.addNotes("Left: one 15-hour window of the King tide (yellow), its delay-embedded point cloud with the FastTDA ellipse (red) and the Gaussian 2-sigma ellipse (dashed), and every feature over time. The window and delay come from the spectrum with the same formulas as the DROPBEAR code. Three clouds per window: the tide alone, the ditch alone, and the tide-vs-ditch loop (input-output phase portrait). Five features from three families: topological (VR, Alpha), algebraic curve fits (conic = genus 0, cubic = genus 1) and statistical (bivariate Gaussian).");

// ---------------- 3. Comparison ----------------
pres.addSection({ title: "Comparison" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Comparison" });
s.addText("The Gaussian ellipse wins the comparison", { placeholder: "title" });
card(s, "Comparison card", 0.6, 1.35, 12.1, 2.85, C.background1);
image(s, "Comparison figure", "img/F3_feature_comparison.png", 0.7, 1.42, 11.9, 630 / 2760);
const hi = { fill: { color: "FCE9DC" }, bold: true };
const T = [
  ["Feature", "Fit failures (loop)", "ms per window", "ρ with depth (Regular / King)", "Regular vs King separation (d)"],
  ["VR H1", "0%", "2.2", "−0.20 / −0.06", "2.0"],
  ["Alpha H1", "0%", "0.6", "−0.23 / −0.08", "2.2"],
  ["FastTDA (genus 0)", "49%", "0.2", "−0.41 / −0.51", "1.5"],
  ["BV Gaussian minor axis", "0%", "0.08", "−0.45 / −0.65", "3.4"],
  ["Cubic oval (genus 1)", "0%", "11.4", "0.01 / −0.11", "0.1"],
];
s.addTable(T.map((r, i) => r.map((t, j) => ({ text: t, options: {
  bold: i === 0 || i === 4, color: i === 0 ? HEX.lt1 : HEX.dk1,
  fill: { color: i === 0 ? HEX.dk2 : (i === 4 ? "FCE9DC" : (i % 2 ? HEX.lt2 : HEX.lt1)) },
  fontSize: 13, align: j === 0 ? "left" : "center", valign: "middle" } }))),
  { x: 0.6, y: 4.4, w: 12.1, colW: [3.1, 2.0, 1.7, 2.8, 2.5], rowH: 0.38,
    border: { type: "solid", pt: 0.5, color: HEX.accent6 }, objectName: "Scorecard table" });
s.addNotes("Top: every feature on every cloud (tide, ditch, loop): fit failures, Spearman correlation with the ditch depth in each event, and how well the per-cycle peak separates Regular from King tide (Cohen's d). Table: tide cloud for the correlation, loop cloud for failures and separation. The bivariate-Gaussian ellipse never fails, is the fastest, tracks the ditch depth best and separates the events best. FastTDA fails in half the loop windows because a hockey-stick cloud is not an ellipse. The genus-1 cubic fit produces spurious ovals and carries no signal. VR/Alpha separate the events but barely follow the depth.");

// ---------------- 4. What the Gaussian ellipse measures ----------------
pres.addSection({ title: "Meaning" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Meaning" });
s.addText("The ellipse measures tide–ditch connectivity", { placeholder: "title" });
card(s, "Meaning card", 0.6, 1.35, 7.4, 5.4, C.background1);
image(s, "Connectivity figure", "img/F5_bvg_loop_meaning.png", 0.72, 1.45, 7.16, 1360 / 1870);
s.addText(bullets([
  "Ditch flooded: depth follows the tide 1:1, so the loop is a thin line (small minor axis)",
  "Ditch disconnected: depth stays flat while the tide moves, so the loop is a round blob",
  "Loop thickness falls with the largest tide (ρ −0.61) and the ditch depth range (ρ −0.60)",
], { fontSize: 14, color: C.text1 }), { x: 8.35, y: 1.4, w: 4.35, h: 2.6, margin: 0, valign: "top", isTextBox: true,
  objectName: "Meaning bullets" });
const M = [["Median per window", "Regular", "King"], ["Highest tide", "0.80 m", "1.08 m"],
           ["Ditch depth range", "0.25 m", "0.53 m"], ["Loop minor axis", "1.53", "1.30"], ["Loop axis ratio", "0.51", "0.38"]];
s.addTable(M.map((r, i) => r.map((t, j) => ({ text: t, options: {
  bold: i === 0 || (i >= 3 && j > 0), color: i === 0 ? HEX.lt1 : HEX.dk1,
  fill: { color: i === 0 ? HEX.dk2 : (i % 2 ? HEX.lt2 : HEX.lt1) }, fontSize: 13,
  align: j === 0 ? "left" : "center", valign: "middle" } }))),
  { x: 8.35, y: 3.4, w: 4.35, colW: [2.15, 1.1, 1.1], rowH: 0.42,
    border: { type: "solid", pt: 0.5, color: HEX.accent6 }, objectName: "Median table" });
s.addNotes("Left: the most coupled window (King tide) and the most disconnected window (Regular tide), each with the tide-vs-ditch cloud scaled within the window and its Gaussian 2-sigma ellipse. When the tide floods the ditch, the depth moves with it and the cloud is a thin line; when the ditch stays at its base level, its small wiggles fill the scaled range and the cloud is round. So the loop's Gaussian minor axis and axis ratio are an index of hydraulic connectivity. King tides are larger and flood the ditch more fully, hence thinner loops. Caveat: the Nov ditch base level is noisier, which also thickens the loop.");

// ---------------- 5. Prediction and conclusions ----------------
pres.addSection({ title: "Prediction" });
s = pres.addSlide({ masterName: "DARK_TITLE_ONLY", sectionTitle: "Prediction" });
s.addText("Memory drives prediction; TDA adds 11% at best", { placeholder: "title" });
card(s, "Prediction card", 0.6, 1.35, 5.45, 5.45, C.background1);
image(s, "Prediction figure", "img/F4_prediction_tau1.5h.png", 0.68, 1.4, 5.3, 1359 / 1350);
stat(s, "Memory gain", 6.45, 1.35, 2.9, "−37%", "error with a nonlinear model and tide memory (0.090 → 0.057 m)", C.accent4, C.background1);
stat(s, "TDA gain", 9.75, 1.35, 2.95, "+11%", "all TDA features together at τ = 1.5 h (R² 0.77 → 0.82); none at τ = 0.5 h", C.accent4, C.background1);
s.addText("Conclusions", { x: 6.45, y: 2.95, w: 6.25, h: 0.42, fontSize: 20, bold: true, fontFace: THEME.headFontFace,
  color: C.accent4, margin: 0, isTextBox: true, objectName: "Conclusions header" });
s.addText(bullets([
  "BV Gaussian ellipse: best feature here – never fails, fastest, best link to depth",
  "VR / Alpha H1: keep for real loops; genus-1 cubic: negative result",
  "Single shape features add under ±3% to prediction; the combined gain depends on τ",
  "Next: more tide events (storms, seasons) to confirm the +11% and a fixed τ rule",
], { fontSize: 14, color: C.background1 }), { x: 6.45, y: 3.42, w: 6.25, h: 3.3, margin: 0, valign: "top",
  isTextBox: true, objectName: "Conclusions" });
s.addNotes("Ditch depth predicted from the tide only, holding out one day at a time over both events (figure: 18.5 h window, 1.5 h delay). A threshold (hinge) model at the learned 0.50 m flooding level reaches RMSE 0.091 m; adding tide memory 0.090 m; gradient boosting with memory 0.057 m (R² 0.77) - the largest gain comes from capturing the threshold and drain-down nonlinearly (with the 15 h / 0.5 h setting: 0.089 to 0.054 m). Single shape features change the error by less than ±3%. All tide TDA features together improved gradient boosting by 11% (0.0565 to 0.0503 m) with a 1.5 h delay, but not with the 0.5 h delay from the DROPBEAR rule, so the combined gain is suggestive, not established. With two events and 15 held-out days, more events are needed.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
