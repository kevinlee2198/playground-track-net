// The labelling page: one `state` object drawn by draw(). Loaded after label_math.js.
"use strict";
const ZOOM = 8;            // magnifier: view pixels per source pixel
const MAG_SPAN = 41;       // magnifier: source pixels across
const MAG_SIZE = ZOOM * MAG_SPAN;  // magnifier: CSS pixels across (328)
const RING_RADIUS = 7;     // source pixels (the ball is ~10 px across at 1080p)
const MIN_LOOK_MS = 250;   // a frame takes a label only after this long on screen
const PRELOAD_BEHIND = 3, PRELOAD_AHEAD = 6;  // frames loaded round the current one
const VISIBILITY_NAMES = { 0: "no ball", 1: "visible", 2: "hard to see", 3: "cannot tell" };
const RING_COLOURS = { 1: "#3f3", 2: "#fa0" };
const RUN_HEADER = "X-Label-Run";  // names this page's server run on every change

const view = document.getElementById("view"), viewCtx = view.getContext("2d");
const mag = document.getElementById("mag"), magCtx = mag.getContext("2d");
const statusEl = document.getElementById("status"), helpEl = document.getElementById("help");

// Everything the page shows is drawn from this by draw(). Handlers change it
// and call render(); nothing else touches the canvases or the status bar.
const state = {
  numFrames: 0,           // 0 until /api/state has loaded
  width: 0, height: 0,    // of every frame, in source pixels
  version: "",            // this server run's token, part of every frame URL and change
  labels: {},             // frame -> {visibility, x, y}, as saved on the server
  current: 0,             // the frame shown and labelled
  showMarks: true,
  showingPrevious: false, // B is held
  helpShown: false,
  saving: false,          // a change is on its way to the server
  error: "",              // shown first, in red, until the next success or move
  fatal: "",              // the page cannot go on: shown for good, and input is refused
  note: "",
  pointer: null,          // the cursor over the frame: {clientX, clientY}
};
const images = new Map();  // frame -> Image, for the frames round the current one
// The frame draw() last put on the view and when (performance.now()), or null
// while no frame image is on it. A label is taken only for a frame on screen
// for MIN_LOOK_MS.
let painted = null;
// The last pointer-down on the view: the frame current then, and why a label
// could not have been saved then ("" if it could). A click is taken only for
// the frame it started on.
let press = null;

let drawPending = false;
function render() {  // draw once, before the next repaint
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; draw(); });
}

// Images

function frameImage(frame) {
  let image = images.get(frame);
  if (!image) {
    image = new Image();
    image.onload = render;
    image.onerror = () => { image.failed = true; render(); };
    image.src = `/frames/${frame}?v=${state.version}`;
    images.set(frame, image);
  }
  return image;
}

function ready(image) { return image.complete && image.naturalWidth > 0; }

function preload() {
  const { current, numFrames } = state;
  for (let i = current - PRELOAD_BEHIND; i <= current + PRELOAD_AHEAD; i++) {
    if (i < 0 || i >= numFrames) continue;
    if (images.get(i)?.failed) images.delete(i);  // try a failed frame again
    frameImage(i);
  }
  for (const i of [...images.keys()])
    if (i < current - 2 * PRELOAD_BEHIND || i > current + 2 * PRELOAD_AHEAD) images.delete(i);
}

// Why a frame's image cannot be labelled, or "" when it can.
function frameProblem(frame) {
  const image = frameImage(frame), { width, height } = state;
  if (image.failed) return `frame ${frame} failed to load (go to another frame and back to retry)`;
  if (!ready(image)) return `frame ${frame} is still loading`;
  if (image.naturalWidth !== width || image.naturalHeight !== height)
    return `frame ${frame} is ${image.naturalWidth}x${image.naturalHeight}, not ${width}x${height}`;
  return "";
}

// Why the current frame cannot take a label now, or "" when it can.
function labelProblem() {
  const frame = state.current;
  if (state.showingPrevious) return "B was held: the previous frame was showing";
  const problem = frameProblem(frame);
  if (problem) return problem;
  if (!painted || painted.frame !== frame) return "too fast: the frame is not on screen yet";
  if (performance.now() - painted.since < MIN_LOOK_MS) return "too fast: look at the frame first";
  return "";
}

function shownFrame() {
  const { current, showingPrevious } = state;
  return showingPrevious && current > 0 ? current - 1 : current;
}

// Drawing

function draw() {
  drawStatus();
  helpEl.hidden = !state.helpShown;
  if (!state.numFrames) return;
  const scale = fitView(), shown = shownFrame(), image = frameImage(shown);
  viewCtx.fillStyle = "#000";
  viewCtx.fillRect(0, 0, state.width * scale.x, state.height * scale.y);
  if (ready(image)) {
    viewCtx.imageSmoothingEnabled = scale.x < 1;
    viewCtx.imageSmoothingQuality = "high";
    viewCtx.drawImage(image, 0, 0, state.width * scale.x, state.height * scale.y);
  }
  if (frameProblem(shown)) painted = null;
  else if (!painted || painted.frame !== shown) painted = { frame: shown, since: performance.now() };
  if (state.showMarks) drawMarks(viewCtx, scale, 0, 0);
  drawMagnifier(image);
}

// Sizes the view to the largest that fits above the status bar, its CSS size
// that of its bitmap so that the image is shown pixel for pixel, and returns
// its scale: CSS pixels per source pixel, across (x) and down (y).
function fitView() {
  const bar = statusEl.offsetHeight;
  document.documentElement.style.setProperty("--bar", bar + "px");
  const fit = Math.min(innerWidth / state.width, (innerHeight - bar) / state.height);
  const dpr = devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(state.width * fit * dpr));
  const pixelHeight = Math.max(1, Math.round(state.height * fit * dpr));
  if (view.width !== pixelWidth || view.height !== pixelHeight) {  // resizing clears it
    view.width = pixelWidth;
    view.height = pixelHeight;
  }
  view.style.width = pixelWidth / dpr + "px";
  view.style.height = pixelHeight / dpr + "px";
  viewCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { x: pixelWidth / dpr / state.width, y: pixelHeight / dpr / state.height };
}

// The marks in a view that shows the source from (startU, startV) at scale.x
// and scale.y view pixels per source pixel: dots for the two previous frames'
// labels, a ring for this frame's.
function drawMarks(ctx, scale, startU, startV) {
  const { labels, current } = state;
  const at = (label) => [
    LabelMath.labelToView(label.x, scale.x, startU), LabelMath.labelToView(label.y, scale.y, startV),
  ];
  for (const [back, alpha] of [[2, 0.35], [1, 0.6]]) {
    const label = labels[current - back];
    if (!label || label.x == null) continue;
    ctx.fillStyle = `rgba(0, 220, 255, ${alpha})`;
    ctx.beginPath();
    ctx.arc(...at(label), scale.x > 2 ? 4 : 2, 0, 2 * Math.PI);
    ctx.fill();
  }
  const label = labels[current];
  if (label && label.x != null) {
    ctx.strokeStyle = RING_COLOURS[label.visibility];
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(...at(label), RING_RADIUS * scale.x + 3, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

function drawMagnifier(image) {
  const { pointer, width, height } = state;
  mag.hidden = !pointer || !ready(image);
  if (mag.hidden) return;
  // The cursor's source position, from the view as it is now laid out
  const rect = view.getBoundingClientRect();
  const u = LabelMath.toSourceEdge(pointer.clientX, rect.left, rect.width, width);
  const v = LabelMath.toSourceEdge(pointer.clientY, rect.top, rect.height, height);
  const rightHalf = pointer.clientX > innerWidth / 2;
  mag.style.left = rightHalf ? "8px" : "auto";  // away from the cursor
  mag.style.right = rightHalf ? "auto" : "8px";
  const dpr = devicePixelRatio || 1, pixels = Math.round(MAG_SIZE * dpr);
  if (mag.width !== pixels) {
    mag.width = pixels;
    mag.height = pixels;
  }
  mag.style.width = mag.style.height = MAG_SIZE + "px";
  magCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // The source square shown, centred on the cursor, and the part of it inside the image
  const startU = u - MAG_SPAN / 2, startV = v - MAG_SPAN / 2;
  const u0 = Math.max(0, startU), v0 = Math.max(0, startV);
  const u1 = Math.min(width, startU + MAG_SPAN), v1 = Math.min(height, startV + MAG_SPAN);
  const size = MAG_SIZE;
  magCtx.fillStyle = "#222";
  magCtx.fillRect(0, 0, size, size);
  magCtx.imageSmoothingEnabled = false;
  if (u1 > u0 && v1 > v0)
    magCtx.drawImage(image, u0, v0, u1 - u0, v1 - v0,
      (u0 - startU) * ZOOM, (v0 - startV) * ZOOM, (u1 - u0) * ZOOM, (v1 - v0) * ZOOM);
  if (state.showMarks) drawMarks(magCtx, { x: ZOOM, y: ZOOM }, startU, startV);
  // The crosshair meets where a click lands, leaving the ball's pixels clear
  const centre = size / 2, gap = 1.5 * ZOOM;
  magCtx.strokeStyle = "rgba(255, 40, 40, 0.9)";
  magCtx.lineWidth = 1;
  magCtx.beginPath();
  magCtx.moveTo(0, centre); magCtx.lineTo(centre - gap, centre);
  magCtx.moveTo(centre + gap, centre); magCtx.lineTo(size, centre);
  magCtx.moveTo(centre, 0); magCtx.lineTo(centre, centre - gap);
  magCtx.moveTo(centre, centre + gap); magCtx.lineTo(centre, size);
  magCtx.stroke();
}

function describe(label) {
  if (!label) return "unlabelled";
  const at = label.x == null ? "" : ` at (${label.x.toFixed(2)}, ${label.y.toFixed(2)})`;
  return `${label.visibility} ${VISIBILITY_NAMES[label.visibility]}${at}`;
}

function drawStatus() {
  const { numFrames, labels, current, error, fatal, note } = state;
  statusEl.classList.toggle("error", !!(error || fatal));
  if (fatal || !numFrames) {
    statusEl.textContent = fatal || error || "Loading…";
    return;
  }
  const parts = [
    `Frame ${current} (0–${numFrames - 1})`,
    `${Object.keys(labels).length} of ${numFrames} labelled`,
    `this frame: ${describe(labels[current])}`,
  ];
  if (state.showingPrevious) parts.push("SHOWING PREVIOUS FRAME");
  if (!state.showMarks) parts.push("marks hidden");
  if (state.saving) parts.push("saving…");
  if (note) parts.push(note);
  if (error) parts.unshift(error);
  parts.push("H: help");
  statusEl.textContent = parts.join("   ·   ");
}

// Actions

function goTo(frame) {
  state.current = LabelMath.clamp(frame, 0, state.numFrames - 1);
  state.note = "";
  state.error = "";
  preload();
  render();
}

function goToFirstUnlabelled() {
  const frame = LabelMath.firstUnlabelled(state.labels, state.numFrames);
  if (frame < 0) { state.note = "every frame is labelled"; render(); } else goTo(frame);
}

// The response's JSON, or an Error with its status (0 if there was no
// response) and text.
async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const err = new Error(`${response.status} ${(await response.text()).trim()}`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

// Sends a change of a frame's label and returns the server's answer. A 409
// (the server is another run's) stops the page for good.
async function sendChange(frame, method, body) {
  try {
    return await fetchJson(`/api/labels/${frame}`, {
      method,
      headers: { "Content-Type": "application/json", [RUN_HEADER]: state.version },
      body,
    });
  } catch (err) {
    if (err.status === 409)
      state.fatal = "This page belongs to an earlier run of the label tool: reload it. " +
        `Frame ${frame} was NOT saved, and nothing more can be (${err.message}).`;
    throw err;
  }
}

// Saves a label for the current frame and moves to the next, unless `problem`
// or labelProblem() says why it cannot be saved.
async function saveLabel(visibility, x, y, problem = "") {
  const frame = state.current, before = state.labels[frame];
  problem = problem || labelProblem();
  state.note = "";
  if (problem) {
    state.error = `Frame ${frame} NOT saved: ${problem}`;
    render();
    return;
  }
  state.saving = true;
  render();
  try {
    state.labels[frame] = await sendChange(frame, "PUT", JSON.stringify({ visibility, x, y }));
    state.error = "";
    const notes = before ? [`frame ${frame} replaced: was ${describe(before)}`] : [];
    if (frame + 1 < state.numFrames) goTo(frame + 1); else notes.push("that was the last frame");
    state.note = notes.join("; ");
  } catch (err) {
    state.error = `Frame ${frame} NOT saved (${err.message}); its label is unchanged`;
  } finally {
    state.saving = false;
    render();
  }
}

async function clearLabel() {
  const frame = state.current;
  if (!(frame in state.labels)) {
    state.note = `frame ${frame} has no label to clear`;
    render();
    return;
  }
  state.saving = true;
  render();
  try {
    await sendChange(frame, "DELETE");
    delete state.labels[frame];
    state.error = "";
    state.note = `cleared frame ${frame}`;
  } catch (err) {
    state.error = `Frame ${frame} NOT cleared (${err.message})`;
  } finally {
    state.saving = false;
    render();
  }
}

// True, with a note saying so, while a change is being saved: one at a time.
function stillSaving() {
  if (state.saving) {
    state.note = "still saving the last change: try again";
    render();
  }
  return state.saving;
}

function whenIdle(action) {
  return () => { if (!stillSaving()) action(); };
}

// Input

view.addEventListener("pointermove", (e) => {
  state.pointer = { clientX: e.clientX, clientY: e.clientY };
  render();
});
view.addEventListener("pointerleave", () => { state.pointer = null; render(); });
view.addEventListener("pointerdown", (e) => {
  press = e.button === 0 ? { frame: state.current, problem: labelProblem() } : null;
});
view.addEventListener("click", (e) => {
  const down = press;
  press = null;
  if (e.button !== 0 || state.fatal || !state.numFrames || stillSaving()) return;
  let problem = "";
  if (!down) problem = "the click did not start on the frame";
  else if (down.frame !== state.current) problem = `the click started on frame ${down.frame}`;
  else problem = down.problem;
  const rect = view.getBoundingClientRect();
  saveLabel(e.shiftKey ? 2 : 1,
    LabelMath.clickToLabel(e.clientX, rect.left, rect.width, state.width),
    LabelMath.clickToLabel(e.clientY, rect.top, rect.height, state.height),
    problem);
});

const MOVES = { ArrowLeft: -1, a: -1, ArrowRight: 1, d: 1 };  // the keys that repeat when held
const KEY_ACTIONS = {
  h: () => { state.helpShown = !state.helpShown; },
  "?": () => { state.helpShown = !state.helpShown; },
  Escape: () => { state.helpShown = false; },
  t: () => { state.showMarks = !state.showMarks; },
  b: () => {
    if (state.current > 0) state.showingPrevious = true;
    else state.note = "frame 0 has no previous frame";
  },
  " ": whenIdle(() => saveLabel(0, null, null)),
  n: whenIdle(() => saveLabel(0, null, null)),
  u: whenIdle(() => saveLabel(3, null, null)),
  Backspace: whenIdle(clearLabel),
  Delete: whenIdle(clearLabel),
  g: whenIdle(goToFirstUnlabelled),
};
for (const [key, step] of Object.entries(MOVES))
  KEY_ACTIONS[key] = whenIdle(() => goTo(state.current + step));

addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || !state.numFrames || state.fatal) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (!KEY_ACTIONS[key] || (key === "Escape" && !state.helpShown)) return;
  e.preventDefault();
  if (e.repeat && !(key in MOVES)) return;  // a held key acts once
  KEY_ACTIONS[key]();
  render();
});
addEventListener("keyup", (e) => {
  if (e.key.toLowerCase() === "b" && state.showingPrevious) {
    state.showingPrevious = false;
    state.note = "";
    render();
  }
});
addEventListener("blur", () => { state.showingPrevious = false; render(); });
addEventListener("resize", render);

(async function start() {
  try {
    const clip = await fetchJson("/api/state");
    state.numFrames = clip.frames;
    state.width = clip.width;
    state.height = clip.height;
    state.version = clip.version;
    state.labels = clip.labels;
    const first = LabelMath.firstUnlabelled(state.labels, state.numFrames);
    state.current = first < 0 ? 0 : first;
    if (first < 0) state.note = "every frame is labelled";
    preload();
  } catch (err) {
    state.error = `Cannot load the clip: ${err.message}`;
  }
  render();
})();
