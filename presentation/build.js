const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/315ab91f-7e57-4858-9715-863ea6106349_b6f21924-0d6d-4aae-b475-447dd4274c7e/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "SEC_MTS_findings.pptx";

// Slate ink with a signal-orange accent: sensor trace on a test-rig panel.
const THEME = {
  name: "SEC Signal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2631", lt1: "FFFFFF", dk2: "33475B", lt2: "EEF2F5",
    accent1: "E4572E", accent2: "2E7D8C", accent3: "8A9BA8",
    accent4: "F2A541", accent5: "5B6C7A", accent6: "B8C4CC",
    hlink: "2E7D8C", folHlink: "5B6C7A",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "SEC sensor vs MTS: Fourier and FastTDA findings";
const C = pres.SchemeColor;

const FOOT = { x: 0.6, y: 7.0, w: 9, h: 0.3, fontSize: 10, margin: 0, valign: "middle" };
const NUM = { x: 12.1, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, align: "right", margin: 0, valign: "middle" };

pres.defineSlideMaster({
  title: "DARK_TITLE_ONLY",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.85,
        fontSize: 36, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "SEC capacitance vs MTS load frame  ·  trials 01–09", options: { ...FOOT, color: C.accent6 } } },
  ],
  slideNumber: { ...NUM, color: C.accent6 },
});
pres.defineSlideMaster({
  title: "TITLE_ONLY",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.85,
        fontSize: 36, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "SEC capacitance vs MTS load frame  ·  trials 01–09", options: { ...FOOT, color: C.accent5 } } },
  ],
  slideNumber: { ...NUM, color: C.accent5 },
});

const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.18 });

function card(slide, name, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background1 },
    line: { color: fill || C.background1, width: 0 }, shadow: shadow(), objectName: name,
  });
}

function stat(slide, name, x, y, w, big, label, bigColor, labelColor) {
  slide.addText(big, { x, y, w, h: 0.75, fontSize: 40, bold: true, fontFace: THEME.headFontFace,
    color: bigColor, margin: 0, valign: "bottom", isTextBox: true, objectName: name + " value" });
  slide.addText(label, { x, y: y + 0.8, w, h: 0.6, fontSize: 14, color: labelColor, margin: 0,
    valign: "top", isTextBox: true, objectName: name + " label" });
}

function bullets(items, opts) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8, ...opts } }));
}

// ---------------- Slide 1: problem, setup, first look ----------------
pres.addSection({ title: "Setup" });
let s = pres.addSlide({ masterName: "DARK_TITLE_ONLY", sectionTitle: "Setup" });
s.addText("SEC capacitance tracks how far the tire is pressed", { placeholder: "title" });
s.addText("Can the SEC sensor stand in for the MTS load frame? The MTS presses 0 → −50.8 mm in a 40 s triangle cycle (force to ≈ −350 N) while SEC records capacitance C1.",
  { x: 0.6, y: 1.3, w: 12.1, h: 0.75, fontSize: 16, color: C.accent6, margin: 0, isTextBox: true, objectName: "Question" });

stat(s, "Trials", 0.6, 2.35, 3.9, "6 of 9", "trials usable (04–09); 01–03 are 60 s single ramps that will not align", C.accent4, C.background1);
stat(s, "Linear fit", 0.6, 3.95, 3.9, "R² 0.77–0.81", "capacitance vs displacement, straight-line fit per trial", C.accent4, C.background1);
stat(s, "Hysteresis", 0.6, 5.55, 3.9, "≈150 N", "force hysteresis loop; capacitance shows very little", C.accent4, C.background1);

card(s, "Figure card", 4.95, 2.25, 7.75, 4.45, C.background1);
s.addImage({ path: "SEC_MTS_cap_force_vs_disp.png", x: 5.1, y: 2.35, w: 7.45, h: 7.45 * 1095 / 2172, objectName: "Capacitance and force vs displacement" });
s.addText("Top: S-shaped capacitance curve, flat beyond −30 mm. Bottom: force loops with a snap-through near −20 mm.",
  { x: 5.1, y: 6.12, w: 7.45, h: 0.45, fontSize: 11, italic: true, color: C.accent5, margin: 0, isTextBox: true, objectName: "Figure caption" });
s.addNotes("Problem: replace or complement the MTS load frame with the SEC capacitive sensor. Target = MTS displacement and force. Trials 04–09 are 160 s of 40 s triangle loading, aligned to SEC by cross-correlation (offset 8.9–10.1 s, |r| 0.86–0.90). Trials 01–03 are short single ramps with alignment |r| 0.30–0.58, so they are reported but not trusted. Capacitance is zeroed per trial (trial 04 baseline sits ≈0.15 pF higher).");

// ---------------- Slide 2: Fourier ----------------
pres.addSection({ title: "Fourier" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Fourier" });
s.addText("One frequency drives the system: 0.025 Hz", { placeholder: "title" });

card(s, "FFT card", 0.6, 1.45, 5.55, 5.35, C.background1);
s.addImage({ path: "trial_05_full_fft.png", x: 0.72, y: 1.55, w: 5.15 * 1650 / 1620, h: 5.15, objectName: "Trial 05 full-length FFT" });

s.addChart(pres.charts.BAR, [
  { name: "Displacement", labels: ["04", "05", "06", "07", "08", "09"], values: [99, 99, 99, 99, 99, 99] },
  { name: "Force", labels: ["04", "05", "06", "07", "08", "09"], values: [77, 74, 76, 77, 78, 80] },
  { name: "SEC capacitance", labels: ["04", "05", "06", "07", "08", "09"], values: [45, 46, 48, 49, 36, 32] },
], {
  x: 6.5, y: 1.4, w: 6.2, h: 3.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
  chartColors: [HEX.dk2, HEX.accent2, HEX.accent1],
  showTitle: true, title: "Share of signal power at 0.025 Hz (%), full-length FFT", titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
  catAxisTitle: "Trial", showCatAxisTitle: true, catAxisTitleFontSize: 11, catAxisTitleColor: HEX.accent5,
  catAxisLabelColor: HEX.accent5, valAxisLabelColor: HEX.accent5, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25,
  valGridLine: { color: "DDE3E8", size: 0.75 }, catGridLine: { style: "none" },
  objectName: "Power share chart",
});
s.addText(bullets([
  "Same 40 s peak in SEC, displacement and force in every trial 04–09",
  "Harmonics: displacement odd only (triangle wave); force and SEC add 0.05 Hz",
  "Over 99% of energy lies below 0.5 Hz, so the test is quasi-static with no resonances",
  "Trials 01–03 have no dominant peak",
], { fontSize: 14, color: C.text1 }), { x: 6.5, y: 4.85, w: 6.2, h: 1.95, margin: 0, valign: "top", isTextBox: true, objectName: "Fourier takeaways" });
s.addNotes("Full-record FFT (Python port of the MATLAB code, mean removed, |fft| over 0..Nyquist). A peak counts as dominant if it holds at least 20% of spectral power and fits at least 3 cycles in the record. SEC carries less of its power at 0.025 Hz (32–49%) because of its 0.05 Hz harmonic and low-frequency drift; trial 09 SEC reads 0.0254 Hz only because of bin resolution. This 40 s period sets the TDA window: 1 period = 40 s window, ¼ period = 10 s Takens delay.");

// ---------------- Slide 3: performance ----------------
pres.addSection({ title: "Performance" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Performance" });
s.addText("SEC predicts displacement with R² ≈ 0.8 across trials", { placeholder: "title" });

card(s, "Matrix card", 0.6, 1.45, 8.3, 5.35, C.background1);
s.addImage({ path: "SEC_MTS_performance_matrix.png", x: 0.72, y: 1.62, w: 8.06, h: 8.06 * 1227 / 2100, objectName: "Performance matrix" });

const sx = 9.3, sw = 3.4;
card(s, "Stat 1", sx, 1.45, sw, 1.65, C.background2);
stat(s, "Correlation", sx + 0.25, 1.45, sw - 0.5, "0.86–0.90", "|r| with displacement, trials 04–09", C.accent1, C.text1);
card(s, "Stat 2", sx, 3.3, sw, 1.65, C.background2);
stat(s, "Transfer", sx + 0.25, 3.3, sw - 0.5, "0.75–0.82", "R² when calibrated on the other trials", C.accent1, C.text1);
card(s, "Stat 3", sx, 5.15, sw, 1.65, C.background2);
stat(s, "Force", sx + 0.25, 5.15, sw - 0.5, "0.60–0.75", "R² for force (other trials): weaker", C.accent2, C.text1);
s.addNotes("Cubic SEC-to-target calibration. Own-trial R² for displacement 0.77–0.82 and other-trial R² 0.75–0.82: the calibration transfers between trials with almost no loss. Coherence with displacement 0.79–0.85 and with force 0.82–0.85. Force is harder because of its hysteresis and snap-through, which a one-to-one capacitance curve cannot represent. Trials 01–03 (red rows) fail: other-trial R² is negative.");

// ---------------- Slide 4: TDA ----------------
pres.addSection({ title: "TDA" });
s = pres.addSlide({ masterName: "DARK_TITLE_ONLY", sectionTitle: "TDA" });
s.addText("FastTDA adds loop shape that capacitance misses", { placeholder: "title" });

card(s, "Chart card", 0.6, 1.45, 7.2, 5.35, C.background1);
s.addChart(pres.charts.BAR, [
  { name: "Capacitance", labels: ["Displacement R²", "Force R²", "Loading vs unloading accuracy"], values: [0.880, 0.550, 0.644] },
  { name: "+ direction", labels: ["Displacement R²", "Force R²", "Loading vs unloading accuracy"], values: [0.885, 0.691, 0.900] },
  { name: "+ direction + FastTDA", labels: ["Displacement R²", "Force R²", "Loading vs unloading accuracy"], values: [0.908, 0.693, 0.956] },
], {
  x: 0.8, y: 1.6, w: 6.8, h: 5.05, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50,
  chartColors: [HEX.accent6, HEX.accent2, HEX.accent1],
  showTitle: true, title: "Leave-one-trial-out, 8 s windows, trials 04–09", titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
  catAxisLabelColor: HEX.accent5, valAxisLabelColor: HEX.accent5, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
  valGridLine: { color: "DDE3E8", size: 0.75 }, catGridLine: { style: "none" },
  objectName: "Validation chart",
});

s.addText("What TDA contributes", { x: 8.2, y: 1.45, w: 4.5, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: C.accent4, margin: 0, isTextBox: true, objectName: "Contribution header" });
s.addText(bullets([
  "Displacement R² 0.88 → 0.91 on held-out trials",
  "Direction accuracy 0.90 → 0.96 alongside the capacitance slope",
  "Window set by physics: one 40 s loading period, 10 s delay",
  "About 100× faster than VR or Alpha persistence",
], { fontSize: 14, color: C.background1 }), { x: 8.2, y: 1.95, w: 4.5, h: 2.25, margin: 0, valign: "top", isTextBox: true, objectName: "Contributions" });

s.addText("Limits", { x: 8.2, y: 4.05, w: 4.5, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: C.accent4, margin: 0, isTextBox: true, objectName: "Limits header" });
s.addText(bullets([
  "Blind to level and direction: alone it is at chance",
  "Force gain did not replicate (0.691 → 0.693)",
  "Ellipse fit fails on 25–48% of short windows",
  "6 usable trials, one amplitude: preliminary",
], { fontSize: 14, color: C.background1 }), { x: 8.2, y: 4.55, w: 4.5, h: 1.95, margin: 0, valign: "top", isTextBox: true, objectName: "Limits" });
s.addNotes("FastTDA: Takens embedding (dim 2), SVD least-squares ellipse fit, feature = minor axis 2b. Validation uses non-overlapping windows, leave-one-trial-out linear models, circular-shift permutation tests and partial Spearman given capacitance. FastTDA alone scores 0.59 on loading vs unloading (chance 0.60), because the loop shape is time-reversal invariant and min-max scaled. Its value is as a complement. FastTDA tracks the 40 s displacement profile well in trials 04–07 (rho 0.84–0.93) but only 0.50–0.55 in 08–09. Next steps: more trials, more load amplitudes, and testing the force gain on new data.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
