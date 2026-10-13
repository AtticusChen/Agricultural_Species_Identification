export const LIMITS = Object.freeze({ count: 10, fileBytes: 20 * 1024 * 1024, totalBytes: 80 * 1024 * 1024 });
const MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const EXTENSION_TYPES = Object.freeze({ jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp" });

export function supportedMime(file) {
  if (file.type) return MIME_TYPES.has(file.type.toLowerCase()) ? file.type.toLowerCase() : null;
  return EXTENSION_TYPES[file.name.split(".").pop().toLowerCase()] || null;
}

export function fileKey(file) { return JSON.stringify([file.name, file.size, file.lastModified]); }

export function validateFiles(existing, incoming) {
  const accepted = [], rejected = [], keys = new Set(existing.map(fileKey));
  let total = existing.reduce((sum, file) => sum + file.size, 0);
  for (const file of incoming) {
    let reason = "";
    if (!supportedMime(file)) reason = "仅支持 JPG、PNG 或 WebP 图片";
    else if (!Number.isFinite(file.size) || file.size <= 0) reason = "图片为空，无法识别";
    else if (file.size > LIMITS.fileBytes) reason = "单张图片不能超过 20 MB";
    else if (keys.has(fileKey(file))) reason = "这张图片已在待识别列表中";
    else if (existing.length + accepted.length >= LIMITS.count) reason = "一次最多加入 10 张图片";
    else if (total + file.size > LIMITS.totalBytes) reason = "图片总大小不能超过 80 MB";
    if (reason) rejected.push({ name: file.name, reason });
    else { accepted.push(file); total += file.size; keys.add(fileKey(file)); }
  }
  return { accepted, rejected };
}

export function batchSettings(health, adjusted, sliderValue) {
  const settings = {};
  if (["precise", "fast"].includes(health?.default_scan_mode)) settings.scan_mode = health.default_scan_mode;
  if (adjusted) {
    const value = Number(sliderValue) / 100;
    if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error("判定阈值应在 0% 到 100% 之间。");
    settings.threshold = value;
  }
  return Object.freeze(settings);
}

export function imageDataInfo(value) {
  if (typeof value !== "string") return null;
  const match = /^data:image\/(jpeg|png|webp);base64,([A-Za-z0-9+/]+={0,2})$/.exec(value);
  if (!match || match[2].length % 4 !== 0) return null;
  return { mime: `image/${match[1]}`, extension: match[1] === "jpeg" ? "jpg" : match[1], base64: match[2] };
}

export function normalizePrediction(data) {
  if (!data || typeof data.present !== "boolean" || !Number.isInteger(data.count) || data.count < 0 || !imageDataInfo(data.annotated_image)) throw new Error("服务返回的识别结果不完整，请重试；本次不作阳性或阴性判断。");
  return { ...data, score: Number.isFinite(data.score) && data.score >= 0 && data.score <= 1 ? data.score : null, threshold: Number.isFinite(data.threshold) ? data.threshold : null, latency_ms: Number.isFinite(data.latency_ms) && data.latency_ms >= 0 ? data.latency_ms : null, model: typeof data.model === "string" ? data.model : "—" };
}

export function resultKind(item) {
  if (item.status === "error") return "error";
  if (item.status !== "done" || typeof item.result?.present !== "boolean") return "pending";
  return item.result.present ? "positive" : "negative";
}

export function summarize(items) {
  const counts = { total: 0, positive: 0, negative: 0, error: 0 };
  for (const item of items) { const kind = resultKind(item); if (kind !== "pending") { counts.total += 1; counts[kind] += 1; } }
  return counts;
}

export function filterHistory(items, filter = "all", query = "") {
  const search = query.trim().toLocaleLowerCase();
  return items.filter((item) => { const kind = resultKind(item); return kind !== "pending" && (filter === "all" || kind === filter) && (!search || item.file.name.toLocaleLowerCase().includes(search)); });
}

export function csvCell(value) {
  let text = value == null ? "" : String(value);
  if (/^[\s\uFEFF]*[=+@-]/u.test(text) || /^[\t\r\n]/u.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}

export function buildCsv(items) {
  const rows = [["文件名", "结果", "参考区域", "判定得分", "阈值", "耗时ms", "模型", "时间", "错误说明"]];
  for (const item of filterHistory(items)) {
    const kind = resultKind(item), data = item.result || {};
    rows.push([item.file.name, { positive: "疑似发现薇甘菊", negative: "本图未检出薇甘菊", error: "识别失败（未作判断）" }[kind], kind === "error" ? "" : data.count, kind === "error" ? "" : data.score, kind === "error" ? "" : data.threshold, kind === "error" ? "" : data.latency_ms, kind === "error" ? "" : data.model, item.completedAt || "", kind === "error" ? item.error : ""]);
  }
  return `\uFEFF${rows.map((row) => row.map(csvCell).join(",")).join("\r\n")}\r\n`;
}

export function safeFilename(name, extension = "jpg") {
  const ext = ["jpg", "png", "webp", "csv"].includes(extension) ? extension : "jpg";
  let base = String(name || "image").replace(/\.[^.]+$/, "");
  base = base.replace(/[<>:"/\\|?*\u0000-\u001f\u007f]/g, "_").replace(/^[.\s]+|[.\s]+$/g, "").slice(0, 100);
  if (!base || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(base)) base = "image";
  return `${base}_识别结果.${ext}`;
}
