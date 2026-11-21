// Tests of the labelling page's script (labeling/static/page.js), run in a vm
// context with a stubbed DOM, canvas, image loading, fetch and clock. Run by
// tests/test_labeling.py:
//   node tests/js/label_page_test.js labeling/static
// Prints {"passed": [names], "failed": [{name, message}]} as JSON.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert/strict");

const STATIC = process.argv[2];
const SOURCE = [
  fs.readFileSync(path.join(STATIC, "label_math.js"), "utf8"),
  fs.readFileSync(path.join(STATIC, "page.js"), "utf8"),
  ";globalThis.__state = state;",
].join("\n");
const RECT = { left: 0, top: 0, width: 800, height: 450 };  // the view, 5 CSS px per source px

// A canvas context that records every call, with the stroke style of the time.
function context(canvas, calls) {
  return new Proxy({}, {
    get(target, prop) {
      if (prop in target) return target[prop];
      return (...args) => calls.push({ canvas, call: prop, args, strokeStyle: target.strokeStyle });
    },
    set(target, prop, value) { target[prop] = value; return true; },
  });
}

// The page loaded afresh. `frames` and `labels` are what /api/state answers;
// `image(frame, attempt)` says how the attempt-th load of a frame ends
// ("load", "fail" or "never"); `respond(request)` answers a change with
// {status, body}, or "hold" to keep it in flight until release().
function makePage({ frames = 5, labels = {}, image = () => "load", respond, dpr = 1 } = {}) {
  const clock = { now: 1000 };
  const calls = [], requests = [], held = [], loads = {}, imageUrls = [];
  let rafQueue = [];
  const layout = { rect: { ...RECT }, innerWidth: 800, innerHeight: 480 };
  const winListeners = {}, docListeners = {};
  function element(id) {
    const listeners = {};
    return {
      id, hidden: id === "mag" || id === "help", style: {}, width: 300, height: 150,
      offsetHeight: 30, textContent: "",
      classList: { set: new Set(), toggle(c, on) { if (on) this.set.add(c); else this.set.delete(c); } },
      getContext: () => context(id, calls),
      getBoundingClientRect: () => ({ ...layout.rect }),
      addEventListener: (type, fn) => { listeners[type] = fn; },
      fire: (type, e) => listeners[type](e),
    };
  }
  const els = { view: element("view"), mag: element("mag"), status: element("status"), help: element("help") };
  class FakeImage {
    constructor() { this.complete = false; this.naturalWidth = 0; this.naturalHeight = 0; }
    set src(url) {
      imageUrls.push(url);
      const frame = Number(/\/frames\/(\d+)/.exec(url)[1]);
      loads[frame] = (loads[frame] || 0) + 1;
      const how = image(frame, loads[frame]);
      Promise.resolve().then(() => {
        if (how === "load") {
          Object.assign(this, { complete: true, naturalWidth: 160, naturalHeight: 90 });
          this.onload();
        } else if (how === "fail") {
          this.complete = true;
          this.onerror();
        }
      });
    }
  }
  const answer = (status, body) => ({
    ok: status < 400,
    status,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
    json: async () => body,
  });
  const defaultRespond = (req) => req.method === "PUT"
    ? { status: 200, body: JSON.parse(req.body) }
    : { status: 200, body: { deleted: 0 } };
  const document = {
    getElementById: (id) => els[id],
    documentElement: { style: { setProperty() {} } },
    visibilityState: "visible",
    addEventListener: (type, fn) => { docListeners[type] = fn; },
  };
  const sandbox = {
    document,
    get innerWidth() { return layout.innerWidth; },
    get innerHeight() { return layout.innerHeight; },
    devicePixelRatio: dpr,
    performance: { now: () => clock.now },
    addEventListener: (type, fn) => { winListeners[type] = fn; },
    requestAnimationFrame: (fn) => rafQueue.push(fn),
    Image: FakeImage,
    console,
    fetch: async (url, options = {}) => {
      const req = { method: options.method || "GET", url, body: options.body, headers: options.headers || {} };
      requests.push(req);
      if (url === "/api/state")
        return answer(200, {
          frames, width: 160, height: 90, run: "run1", write_token: "token1", labels: structuredClone(labels),
        });
      const reply = (respond || defaultRespond)(req);
      if (reply === "network") throw new TypeError("Failed to fetch");
      if (reply === "hold") {
        return new Promise((resolve) => held.push(() => resolve(answer(200, defaultRespond(req).body))));
      }
      return answer(reply.status, reply.body);
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(SOURCE, sandbox);

  const page = {
    els, calls, requests, loads, layout, imageUrls,
    get state() { return sandbox.__state; },
    get status() { return els.status.textContent; },
    get errorShown() { return els.status.classList.set.has("error"); },
    changes: () => requests.filter((r) => r.method !== "GET"),
    puts: () => requests.filter((r) => r.method === "PUT"),
    putUrls: () => page.puts().map((p) => p.url),
    // Lets every pending promise, image load and fetch finish (the clock stands still)
    settle: async () => { for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r)); },
    paint: () => { const queue = rafQueue; rafQueue = []; queue.forEach((fn) => fn()); },
    // Everything pending finishes, then the next animation frame is drawn
    show: async () => { await page.settle(); page.paint(); },
    wait: (ms) => { clock.now += ms; },
    // The labeller looks at the frame on screen for longer than the minimum
    look: async () => { await page.show(); page.wait(300); },
    pointerDown: (x = 400, y = 225, extra = {}) =>
      els.view.fire("pointerdown", { button: 0, pointerType: "mouse", clientX: x, clientY: y, ...extra }),
    pointerUp: (x = 400, y = 225, extra = {}) =>
      els.view.fire("click", { button: 0, clientX: x, clientY: y, shiftKey: false, ...extra }),
    click: (x = 400, y = 225, extra = {}) => { page.pointerDown(x, y); page.pointerUp(x, y, extra); },
    key: (k, extra = {}) => {
      let prevented = false;
      winListeners.keydown({ key: k, repeat: false, preventDefault() { prevented = true; }, ...extra });
      return prevented;
    },
    keyUp: (k) => winListeners.keyup({ key: k }),
    fireWindow: (type) => winListeners[type]({}),
    // The tab is hidden ("hidden") or shown again ("visible")
    setVisibility: (visibility) => {
      document.visibilityState = visibility;
      docListeners.visibilitychange({});
    },
    release: () => held.splice(0).forEach((fn) => fn()),
    start: async () => { await page.show(); return page; },
  };
  return page;
}

const TESTS = [];
function test(name, fn) { TESTS.push([name, fn]); }

// Labelling

test("starts at the first unlabelled frame and shows it in the status line", async () => {
  const page = await makePage({ labels: { 0: { visibility: 1, x: 5, y: 5 } } }).start();
  assert.equal(page.state.current, 1);
  assert.ok(page.status.startsWith("Frame 1 (0–4)"), page.status);
  assert.ok(page.status.includes("1 of 5 labelled"), page.status);
});

test("a click after looking saves that pixel, with the write token, and moves on", async () => {
  const page = await makePage().start();
  await page.look();
  page.click(400, 225);
  await page.settle();
  const [put] = page.puts();
  assert.equal(put.url, "/api/labels/0");
  assert.deepEqual(JSON.parse(put.body), { visibility: 1, x: 79.5, y: 44.5 });
  assert.equal(put.headers["X-Label-Write-Token"], "token1");
  assert.ok(page.imageUrls.length > 0);
  assert.ok(page.imageUrls.every((url) => url.endsWith("?run=run1")), page.imageUrls);
  assert.equal(page.state.current, 1);
  assert.deepEqual(page.state.labels[0], { visibility: 1, x: 79.5, y: 44.5 });
});

test("shift-click saves hard to see (2)", async () => {
  const page = await makePage().start();
  await page.look();
  page.click(10, 10, { shiftKey: true });
  await page.settle();
  assert.equal(JSON.parse(page.puts()[0].body).visibility, 2);
});

test("N, Space and U save no ball (0) and cannot tell (3)", async () => {
  const page = await makePage().start();
  for (const k of ["n", " ", "u"]) {
    await page.look();
    assert.ok(page.key(k));
    await page.settle();
  }
  assert.deepEqual(page.puts().map((p) => [p.url, JSON.parse(p.body)]), [
    ["/api/labels/0", { visibility: 0, x: null, y: null }],
    ["/api/labels/1", { visibility: 0, x: null, y: null }],
    ["/api/labels/2", { visibility: 3, x: null, y: null }],
  ]);
  assert.equal(page.state.current, 3);
});

test("the last frame stays and says so", async () => {
  const page = await makePage({ frames: 2 }).start();
  await page.look(); page.key("n");
  await page.look(); page.key("u");
  await page.show();
  assert.equal(page.state.current, 1);
  assert.equal(page.state.labels[1].visibility, 3);
  assert.ok(page.status.includes("that was the last frame"), page.status);
});

test("replacing a label says what it was", async () => {
  const page = await makePage({ labels: { 0: { visibility: 0, x: null, y: null } } }).start();
  page.key("ArrowLeft");
  await page.look();
  page.click(400, 225);
  await page.show();
  assert.equal(page.state.current, 1);
  assert.ok(page.status.includes("frame 0 replaced: was 0 no ball"), page.status);
});

// A frame takes a label only once it has been on screen for 250 ms

test("a double click labels one frame: the second is too fast for the next", async () => {
  const page = await makePage().start();
  await page.look();
  page.click();
  await page.show();  // the save returns and the page moves to frame 1, drawn
  page.wait(100);
  page.click();
  await page.show();
  assert.deepEqual(page.putUrls(), ["/api/labels/0"]);
  assert.equal(page.state.current, 1);
  assert.ok(!(1 in page.state.labels));
  assert.ok(page.status.includes("Frame 1 NOT saved: too fast"), page.status);
  assert.ok(page.errorShown);
});

test("a click before the next frame is drawn is refused, however late", async () => {
  const page = await makePage().start();
  await page.look();
  page.click();
  await page.settle();  // moved to frame 1, but no animation frame has drawn it
  page.wait(1000);
  page.click();
  await page.settle();
  assert.deepEqual(page.putUrls(), ["/api/labels/0"]);
  page.paint();
  assert.ok(page.status.includes("not on screen yet"), page.status);
  page.wait(300);
  page.click();
  await page.settle();
  assert.deepEqual(page.putUrls(), ["/api/labels/0", "/api/labels/1"]);
});

test("a click or key within 250 ms of the frame appearing is refused, then taken", async () => {
  const page = await makePage().start();
  page.wait(249);
  page.click();
  page.key("n");
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("too fast: look at the frame first"), page.status);
  page.wait(1);
  page.key("n");
  await page.settle();
  assert.equal(page.puts().length, 1);
});

test("on the last frame, a double click saves once: the clock starts again after a save", async () => {
  const page = await makePage({ frames: 1 }).start();
  await page.look();
  page.click(400, 225);
  await page.show();
  page.wait(100);
  page.click(402, 226);
  await page.show();
  assert.deepEqual(page.putUrls(), ["/api/labels/0"]);
  assert.ok(page.status.includes("Frame 0 NOT saved: too fast"), page.status);
  page.wait(250);
  page.click(402, 226);
  await page.settle();
  assert.deepEqual(page.putUrls(), ["/api/labels/0", "/api/labels/0"]);
});

// Coming back to the window: the click that activates it, or a key at once,
// labels nothing, however long the frame was on screen before

const RETURNS = {
  "window focus": (page) => { page.fireWindow("blur"); page.wait(60000); page.fireWindow("focus"); },
  "tab shown again": (page) => { page.setVisibility("hidden"); page.wait(60000); page.setVisibility("visible"); },
};
for (const [how, comeBack] of Object.entries(RETURNS)) {
  test(`${how}: a click at once is refused and says why; 250 ms later it is taken`, async () => {
    const page = await makePage().start();
    await page.look();
    comeBack(page);
    page.click();
    await page.show();
    assert.equal(page.puts().length, 0);
    assert.ok(page.status.startsWith("Frame 0 NOT saved: the window has only just come back"), page.status);
    assert.ok(page.errorShown);
    page.wait(249);
    page.click();
    await page.settle();
    assert.equal(page.puts().length, 0);
    page.wait(1);
    page.click();
    await page.settle();
    assert.deepEqual(page.putUrls(), ["/api/labels/0"]);
  });

  test(`${how}: a press that came just before it is refused, however long it is held`, async () => {
    const page = await makePage().start();
    await page.look();
    page.fireWindow("blur");
    page.pointerDown();  // the activating click's press, delivered first
    comeBack(page);
    await page.look();  // the frame is painted again and on screen long enough
    page.pointerUp();
    await page.show();
    assert.equal(page.puts().length, 0);
    assert.ok(page.status.includes("the window has only just come back"), page.status);
  });

  test(`${how}: N at once is refused; 250 ms later it is taken`, async () => {
    const page = await makePage().start();
    await page.look();
    comeBack(page);
    page.key("n");
    await page.show();
    page.wait(249);
    page.key("n");
    await page.settle();
    assert.equal(page.puts().length, 0);
    assert.ok(page.status.includes("the window has only just come back"), page.status);
    page.wait(1);
    page.key("n");
    await page.settle();
    assert.deepEqual(page.putUrls(), ["/api/labels/0"]);
  });
}

// A click is where the button went down and came up

test("a click whose pointer moved up to 2 CSS px is taken, at the release point", async () => {
  const page = await makePage().start();
  await page.look();
  page.pointerDown(400, 225);
  page.pointerUp(400, 227);
  await page.settle();
  assert.deepEqual(JSON.parse(page.puts()[0].body), { visibility: 1, x: 79.5, y: 44.9 });
});

test("a click whose pointer moved more than 2 CSS px is refused", async () => {
  const page = await makePage().start();
  await page.look();
  page.pointerDown(400, 225);
  page.pointerUp(401.5, 226.5);  // 2.12 px
  await page.show();
  page.pointerDown(100, 100);
  page.pointerUp(400, 300);  // a drag
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("Frame 0 NOT saved: the pointer moved during the click"), page.status);
});

test("a touch is refused: there is no magnifier for it", async () => {
  const page = await makePage().start();
  await page.look();
  page.pointerDown(200, 200, { pointerType: "touch" });
  page.pointerUp(200, 200);
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("touch is not taken"), page.status);
});

test("the arrow keys, A and D move at once", async () => {
  const page = await makePage().start();
  page.key("ArrowRight"); page.key("d"); page.key("d"); page.key("a");
  assert.equal(page.state.current, 2);
  page.key("ArrowLeft");
  assert.equal(page.state.current, 1);
});

test("a frame whose image never loads cannot be labelled", async () => {
  const page = await makePage({ image: (frame) => (frame === 0 ? "never" : "load") }).start();
  await page.look();
  page.click();
  page.key("n");
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("frame 0 is still loading"), page.status);
});

test("a frame whose image failed is loaded again when shown again", async () => {
  const page = await makePage({ image: (frame, attempt) => (frame === 0 && attempt === 1 ? "fail" : "load") }).start();
  await page.look();
  page.click();
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("frame 0 failed to load"), page.status);
  page.key("ArrowRight");
  page.key("ArrowLeft");
  await page.look();
  assert.equal(page.loads[0], 2);
  page.click();
  await page.settle();
  assert.deepEqual(page.putUrls(), ["/api/labels/0"]);
});

// A held key labels one frame

test("holding N labels one frame, not one per auto-repeat", async () => {
  const page = await makePage({ frames: 50 }).start();
  await page.look();
  page.key("n");
  for (let i = 0; i < 20; i++) {
    await page.show();
    page.wait(33);
    assert.ok(page.key("n", { repeat: true }));  // handled, so Space does not scroll
  }
  await page.settle();
  assert.deepEqual(page.putUrls(), ["/api/labels/0"]);
  assert.equal(page.state.current, 1);
});

test("pressing N again before the next frame has been seen is refused", async () => {
  const page = await makePage().start();
  await page.look();
  page.key("n");
  await page.show();
  page.wait(100);
  page.key("n");
  await page.settle();
  assert.equal(page.puts().length, 1);
});

test("a held arrow key keeps moving", async () => {
  const page = await makePage({ frames: 50 }).start();
  page.key("ArrowRight");
  for (let i = 0; i < 5; i++) page.key("ArrowRight", { repeat: true });
  assert.equal(page.state.current, 6);
});

// Another tab, or a newer run, has the write token

test("a 409 (another tab or a newer run has taken over) stops the page for good", async () => {
  const page = await makePage({
    respond: () => ({ status: 409, body: "another tab or a newer run has taken over: reload this page\n" }),
  }).start();
  await page.look();
  page.click();
  await page.show();
  assert.equal(page.changes().length, 1);
  assert.ok(page.status.startsWith("Another tab or a newer run has taken over: reload this page. "
    + "Frame 0 was NOT saved"), page.status);
  assert.ok(page.errorShown);
  assert.equal(page.state.current, 0);
  assert.ok(!(0 in page.state.labels));
  for (const k of ["n", "u", " ", "ArrowRight", "d", "Backspace", "g"]) page.key(k);
  await page.look();
  page.click();
  await page.show();
  assert.equal(page.changes().length, 1);
  assert.equal(page.state.current, 0);
  assert.ok(page.status.includes("reload this page"), page.status);
});

test("a clear sends the write token too", async () => {
  const page = await makePage({ labels: { 1: { visibility: 0, x: null, y: null } } }).start();
  page.key("ArrowRight");
  page.key("Backspace");
  await page.show();
  const [del] = page.changes();
  assert.equal(del.method, "DELETE");
  assert.equal(del.url, "/api/labels/1");
  assert.equal(del.headers["X-Label-Write-Token"], "token1");
  assert.ok(!(1 in page.state.labels));
  assert.ok(page.status.includes("cleared frame 1"), page.status);
  page.key("Backspace");
  page.paint();
  assert.ok(page.status.includes("frame 1 has no label to clear"), page.status);
});

// B, or a frame change, between pointer-down and pointer-up

test("B held at pointer-down and released before pointer-up: refused", async () => {
  const page = await makePage().start();
  page.key("ArrowRight");
  await page.look();
  page.key("b");
  page.paint();
  page.pointerDown();
  page.keyUp("b");
  await page.look();  // long enough after the release
  page.pointerUp();
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("B was held"), page.status);
});

test("B pressed during a click: refused", async () => {
  const page = await makePage().start();
  page.key("ArrowRight");
  await page.look();
  page.pointerDown();
  page.key("b");
  page.paint();
  page.wait(300);
  page.pointerUp();
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("B was held"), page.status);
});

test("N, Space and U are refused while B is held, and taken after", async () => {
  const page = await makePage().start();
  page.key("ArrowRight");
  await page.look();
  page.key("b");
  await page.look();
  for (const k of ["n", " ", "u"]) page.key(k);
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("Frame 1 NOT saved: B was held"), page.status);
  assert.ok(page.status.includes("SHOWING PREVIOUS FRAME"), page.status);
  page.keyUp("b");
  await page.look();
  page.key("n");
  await page.settle();
  assert.deepEqual(page.putUrls(), ["/api/labels/1"]);
});

test("a frame change between pointer-down and pointer-up: refused", async () => {
  const page = await makePage().start();
  await page.look();
  page.pointerDown();
  page.key("ArrowRight");
  await page.look();
  page.pointerUp();
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("the click started on frame 0"), page.status);
});

test("a click that did not start on the frame: refused", async () => {
  const page = await makePage().start();
  await page.look();
  page.pointerUp();
  await page.show();
  assert.equal(page.puts().length, 0);
  assert.ok(page.status.includes("did not start on the frame"), page.status);
});

// Failures of a save

test("a failed save stays, shows the error, and the next click works", async () => {
  let fail = true;
  const page = await makePage({
    respond: (req) => (fail ? { status: 500, body: "[Errno 28] No space left on device\n" }
      : { status: 200, body: JSON.parse(req.body) }),
  }).start();
  await page.look();
  page.click();
  await page.show();
  assert.equal(page.state.current, 0);
  assert.ok(!(0 in page.state.labels));
  assert.ok(page.errorShown);
  assert.ok(page.status.startsWith("Frame 0 NOT saved (500 [Errno 28] No space left on device)"), page.status);
  fail = false;
  page.click();
  await page.show();
  assert.equal(page.state.current, 1);
  assert.equal(page.state.labels[0].visibility, 1);
  assert.ok(!page.errorShown);
});

test("one change at a time: a key or click while saving is refused, in red", async () => {
  const page = await makePage({ respond: () => "hold" }).start();
  await page.look();
  page.key("n");
  page.key("ArrowRight");
  page.click();
  page.paint();
  assert.equal(page.changes().length, 1);
  assert.ok(page.status.startsWith("Right on frame 0 NOT done: the last change was still being saved"
    + "   ·   the click on frame 0 NOT done"), page.status);
  assert.ok(page.errorShown);
  page.release();
  await page.settle();
  assert.equal(page.state.current, 1);
});

test("a refusal during a save outlasts the save's result, until the next input", async () => {
  const page = await makePage({ respond: () => "hold" }).start();
  await page.look();
  page.click();          // a mistaken click on frame 0, still being saved
  page.key("Backspace"); // undo at once: refused
  page.key("Backspace"); // the same refusal is listed once
  page.release();
  await page.show();
  assert.equal(page.state.current, 1);  // the save went through and moved on
  assert.deepEqual(page.changes().map((c) => c.method), ["PUT"]);
  const refusal = "Backspace on frame 0 NOT done: the last change was still being saved";
  assert.ok(page.status.startsWith(refusal + "   ·   Frame 1 (0–4)"), page.status);
  assert.equal(page.status.split(refusal).length, 2);
  assert.ok(page.errorShown);
  page.wait(5000);
  page.paint();
  assert.ok(page.status.includes(refusal), "a refusal stays until the next input");
  page.key("t");
  page.paint();
  assert.ok(!page.status.includes("NOT done"), page.status);
  assert.ok(!page.errorShown);
});

test("a save the server refused says NOT saved; one with no answer may have been made", async () => {
  let reply = { status: 400, body: "visibility 1 needs x in 0-159, got 160\n" };
  const page = await makePage({ respond: () => reply }).start();
  await page.look();
  page.click();
  await page.show();
  assert.equal(page.status.split("   ·   ")[0], "Frame 0 NOT saved (400 visibility 1 needs x in 0-159, got 160)");
  reply = "network";
  page.click();
  await page.show();
  assert.equal(page.status.split("   ·   ")[0],
    "Frame 0 may not have been saved (Failed to fetch): reload the page to check");
  assert.ok(page.errorShown);
  page.key("ArrowRight");
  page.key("ArrowLeft");
  await page.look();
  page.key("u");
  await page.show();
  assert.equal(page.status.split("   ·   ")[0],
    "Frame 0 may not have been saved (Failed to fetch): reload the page to check");
});

test("a clear with no answer may have been made", async () => {
  const page = await makePage({
    labels: { 0: { visibility: 0, x: null, y: null } }, respond: () => "network",
  }).start();
  page.key("ArrowLeft");
  page.key("Backspace");
  await page.show();
  assert.ok(page.status.startsWith(
    "Frame 0 may not have been cleared (Failed to fetch): reload the page to check"), page.status);
});

// Drawing

test("this frame's label is drawn as a ring round its position", async () => {
  const page = await makePage().start();
  await page.look();
  page.click(400, 225);
  await page.settle();
  page.key("ArrowLeft");
  page.calls.length = 0;
  page.paint();
  const ring = page.calls.find((c) => c.canvas === "view" && c.call === "arc"
    && c.args[2] === 7 * 5 + 3);
  assert.ok(ring, "no ring drawn");
  assert.deepEqual(ring.args.slice(0, 2), [400, 225]);  // (79.5 + 0.5) * 5, (44.5 + 0.5) * 5
  const stroke = page.calls.find((c) => c.canvas === "view" && c.call === "stroke");
  assert.equal(stroke.strokeStyle, "#3f3");
  page.key("t");
  page.calls.length = 0;
  page.paint();
  assert.ok(!page.calls.some((c) => c.canvas === "view" && c.call === "arc"));
  assert.ok(page.status.includes("marks hidden"), page.status);
});

test("the view's CSS size is its bitmap's, at any device pixel ratio", async () => {
  const page = await makePage({ dpr: 1.5 }).start();
  const view = page.els.view;
  assert.deepEqual([view.width, view.height], [1200, 675]);
  assert.deepEqual([view.style.width, view.style.height], ["800px", "450px"]);
  page.layout.innerWidth = 801;  // 801 / 160 x 1.5 x 160 = 1201.5 device pixels
  page.layout.innerHeight = 1000;
  page.fireWindow("resize");
  page.paint();
  assert.equal(view.width, 1202);
  assert.equal(view.style.width, `${1202 / 1.5}px`);
});

test("the magnifier is scaled for the device pixel ratio and avoids the cursor", async () => {
  const page = await makePage({ dpr: 2 }).start();
  page.els.view.fire("pointermove", { clientX: 600, clientY: 225 });
  page.paint();
  const mag = page.els.mag;
  assert.equal(mag.hidden, false);
  assert.deepEqual([mag.width, mag.height, mag.style.width], [656, 656, "328px"]);
  assert.deepEqual([mag.style.left, mag.style.right], ["8px", "auto"]);
  page.els.view.fire("pointermove", { clientX: 100, clientY: 225 });
  page.paint();
  assert.deepEqual([mag.style.left, mag.style.right], ["auto", "8px"]);
  page.els.view.fire("pointerleave", {});
  page.paint();
  assert.equal(mag.hidden, true);
});

test("the magnifier follows the view when the window is resized", async () => {
  const page = await makePage().start();
  page.els.view.fire("pointermove", { clientX: 400, clientY: 225 });
  page.paint();
  const lastMagImage = () => page.calls.filter((c) => c.canvas === "mag" && c.call === "drawImage").at(-1);
  assert.equal(lastMagImage().args[1], 80 - 20.5);  // u = 400 / 5
  page.layout.rect = { left: 0, top: 0, width: 400, height: 225 };
  page.fireWindow("resize");
  page.paint();
  assert.equal(lastMagImage().args[1], 160 - 20.5);  // u = 400 / 2.5, the same cursor
});

// Keys

test("help, Escape, T, G and modifier keys", async () => {
  const page = await makePage({ labels: { 0: { visibility: 0, x: null, y: null } } }).start();
  assert.ok(!page.key("Escape"));  // not handled while the help is hidden
  page.key("h");
  page.paint();
  assert.equal(page.els.help.hidden, false);
  assert.ok(page.key("Escape"));
  page.paint();
  assert.equal(page.els.help.hidden, true);
  assert.ok(!page.key("n", { ctrlKey: true }));
  page.key("ArrowRight"); page.key("ArrowRight");
  page.key("g");
  assert.equal(page.state.current, 1);
  page.key("b");
  page.paint();
  assert.ok(page.status.includes("SHOWING PREVIOUS FRAME"), page.status);
  page.fireWindow("blur");
  page.paint();
  assert.ok(!page.status.includes("SHOWING PREVIOUS FRAME"), page.status);
});

test("G says when every frame is labelled", async () => {
  const none = { visibility: 0, x: null, y: null };
  const page = await makePage({ frames: 2, labels: { 0: none, 1: none } }).start();
  assert.ok(page.status.includes("every frame is labelled"), page.status);
  page.key("ArrowRight");
  page.key("g");
  page.paint();
  assert.ok(page.status.includes("every frame is labelled"), page.status);
});

(async () => {
  const passed = [], failed = [];
  for (const [name, fn] of TESTS) {
    try {
      await fn();
      passed.push(name);
    } catch (err) {
      failed.push({ name, message: err.stack || String(err) });
    }
  }
  console.log(JSON.stringify({ passed, failed }));
})();
