// Pure functions of the labelling page, loadable by Node for tests.
// Source pixels have their centres at integers: the centre of the top-left
// pixel is (0, 0), and its left edge is at -0.5.
const LabelMath = (function () {
  "use strict";

  function clamp(value, lo, hi) {
    return Math.min(hi, Math.max(lo, value));
  }

  // The continuous position, in source pixels from the image's left (top)
  // edge, of a pointer at clientPos over an image shown from rectStart over
  // rectSize CSS pixels.
  function toSourceEdge(clientPos, rectStart, rectSize, sourceSize) {
    return ((clientPos - rectStart) * sourceSize) / rectSize;
  }

  // The label coordinate of a click: through pixel centres (u - 0.5), clamped
  // to the centres of the edge pixels, rounded to the two decimals stored.
  function clickToLabel(clientPos, rectStart, rectSize, sourceSize) {
    const u = toSourceEdge(clientPos, rectStart, rectSize, sourceSize);
    return Math.round(clamp(u - 0.5, 0, sourceSize - 1) * 100) / 100;
  }

  // Where a label coordinate is drawn, in pixels of a view that shows the
  // source from viewStart (a continuous source position) at `scale` view
  // pixels per source pixel.
  function labelToView(coord, scale, viewStart) {
    return (coord + 0.5 - (viewStart || 0)) * scale;
  }

  // The first frame in 0..numFrames-1 with no label (labels: an object keyed
  // by frame), or -1 when every frame is labelled.
  function firstUnlabelled(labels, numFrames) {
    for (let i = 0; i < numFrames; i++) {
      if (!Object.prototype.hasOwnProperty.call(labels, String(i))) return i;
    }
    return -1;
  }

  return { clamp, toSourceEdge, clickToLabel, labelToView, firstUnlabelled };
})();

if (typeof module !== "undefined") module.exports = LabelMath;
