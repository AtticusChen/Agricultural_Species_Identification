import test from "node:test";
import assert from "node:assert/strict";
import { LIMITS, batchSettings, buildCsv, csvCell, filterHistory, imageDataInfo, normalizePrediction, resultKind, safeFilename, summarize, supportedMime, validateFiles } from "../src/mikania_app/static/workspace-core.mjs";
const file = (name = "photo.jpg", size = 1024, type = "image/jpeg", lastModified = 1) => ({ name, size, type, lastModified });
const image = "data:image/jpeg;base64,/9j/2Q==";
const prediction = (overrides = {}) => ({ present: true, count: 0, score: 0.9, threshold: 0.88, annotated_image: image, latency_ms: 200, model: "v3", ...overrides });
const item = (status, result = null, name = "photo.jpg") => ({ status, result, file: file(name), completedAt: "2026-09-14T00:00:00.000Z", error: "=not a formula" });

test("supported image MIME or explicit fallback extension only", () => {
  assert.equal(supportedMime(file("PHOTO.JPG", 20, "")), "image/jpeg");
  assert.equal(supportedMime(file("photo.jpg", 20, "text/html")), null);
  assert.equal(supportedMime(file("photo.svg", 20, "")), null);
  assert.equal(supportedMime(file("photo.webp", 20, "image/webp")), "image/webp");
});
test("reject empty, oversized, unsupported and duplicate files", () => {
  const result = validateFiles([file()], [file(), file("zero.jpg", 0), file("huge.jpg", LIMITS.fileBytes + 1), file("x.svg", 2, "image/svg+xml"), file("ok.png", 2, "image/png")]);
  assert.equal(result.accepted.length, 1);
  assert.equal(result.rejected.length, 4);
  assert.equal(validateFiles([], [file(), file()]).accepted.length, 1);
});
test("enforce ten-file and eighty-megabyte bounds including existing queue", () => {
  const tiny = Array.from({ length: 11 }, (_, index) => file(`${index}.jpg`));
  assert.equal(validateFiles([], tiny).accepted.length, 10);
  const large = Array.from({ length: 5 }, (_, index) => file(`${index}.jpg`, LIMITS.fileBytes));
  assert.equal(validateFiles([], large).accepted.length, 4);
  assert.equal(validateFiles(large.slice(0, 3), large.slice(3)).accepted.length, 1);
});
test("recommended threshold is omitted, not slider-rounded; reset uses omission", () => {
  const health = { default_scan_mode: "precise", default_threshold: 0.87654321 };
  assert.deepEqual(batchSettings(health, false, "87.7"), { scan_mode: "precise" });
  assert.deepEqual(batchSettings(health, true, "87.7"), { scan_mode: "precise", threshold: 0.877 });
  assert.deepEqual(batchSettings(health, false, "87.7"), { scan_mode: "precise" });
  assert.throws(() => batchSettings(health, true, "NaN"));
  assert.throws(() => batchSettings(health, true, "101"));
});
test("safe image payloads only", () => {
  assert.equal(imageDataInfo(image).extension, "jpg");
  for (const bad of ["https://example.org/image.jpg", "data:text/html;base64,AAAA", "data:image/svg+xml;base64,AAAA", "data:image/png;base64,abc", "javascript:alert(1)", null]) assert.equal(imageDataInfo(bad), null);
});
test("positive with no regions remains positive and missing score remains unknown", () => {
  const normalized = normalizePrediction(prediction({ score: null, detections: [{ confidence: 0.99 }] }));
  assert.equal(normalized.present, true);
  assert.equal(normalized.count, 0);
  assert.equal(normalized.score, null);
  assert.equal(resultKind(item("done", normalized)), "positive");
});
test("malformed results throw instead of becoming negatives", () => {
  for (const overrides of [{ present: undefined }, { present: "false" }, { count: -1 }, { annotated_image: "data:text/html;base64,AAAA" }]) assert.throws(() => normalizePrediction(prediction(overrides)));
  assert.equal(resultKind(item("error")), "error");
  assert.equal(resultKind(item("pending")), "pending");
});
test("session counts distinguish errors and negatives, filters only completed entries", () => {
  const items = [item("done", prediction(), "UAV.jpg"), item("done", prediction({ present: false }), "ground.jpg"), item("error"), item("pending"), item("running")];
  assert.deepEqual(summarize(items), { total: 3, positive: 1, negative: 1, error: 1 });
  assert.equal(filterHistory(items).length, 3);
  assert.equal(filterHistory(items, "negative").length, 1);
  assert.equal(filterHistory(items, "all", " uav ").length, 1);
});
test("CSV BOM, quoting, formula escapes and empty scores are preserved", () => {
  for (const unsafe of ["=1+2", " +123", "-1", "@SUM(1)", "\t=1", "\rtest"]) assert.ok(csvCell(unsafe).startsWith('"\''));
  assert.equal(csvCell('a,"b"\nc'), '"a,""b""\nc"');
  const csv = buildCsv([item("done", prediction({ score: null }), "=HYPERLINK(1).jpg"), item("error")]);
  assert.ok(csv.startsWith("\uFEFF"));
  assert.ok(csv.includes('"\'=HYPERLINK(1).jpg"'));
  assert.ok(csv.includes('"0","","0.88"'));
  assert.ok(csv.includes("识别失败（未作判断）"));
  assert.ok(csv.includes('"\'=not a formula"'));
});
test("download names remove separators, controls and device names", () => {
  assert.equal(safeFilename("CON.jpg"), "image_识别结果.jpg");
  assert.equal(safeFilename("leaf.png", "png"), "leaf_识别结果.png");
  assert.ok(!safeFilename("../path\\x\n.jpg").match(/[\\/\n]/));
});
