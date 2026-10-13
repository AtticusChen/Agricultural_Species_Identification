import { LIMITS, batchSettings, buildCsv, filterHistory, imageDataInfo, normalizePrediction, resultKind, safeFilename, summarize, supportedMime, validateFiles } from "./workspace-core.mjs";

const $ = (id) => document.getElementById(id);
const state = { items: [], selectedId: null, health: null, ready: false, checking: false, running: false, stopping: false, thresholdAdjusted: false, display: "annotated", filter: "all", view: "workspace", sampleLoading: false, progress: null };
let nextId = 1;
let exampleCatalog = null;
const selected = () => state.items.find((item) => item.id === state.selectedId);
const formatScore = (value) => Number.isFinite(value) ? value.toFixed(3) : "—";
const formatTime = (value) => Number.isFinite(value) ? `${(value / 1000).toFixed(2)} s` : "—";
const formatThreshold = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const labels = { positive: "疑似检出", negative: "未检出", error: "识别失败", pending: "等待识别" };

function notify(message, tone = "info") {
  $("notice").textContent = message;
  $("notice").dataset.tone = tone;
}

function switchView(view) {
  if (!["workspace", "history", "guide"].includes(view)) return;
  state.view = view;
  for (const name of ["workspace", "history", "guide"]) $(`${name}View`).hidden = name !== view;
  for (const button of document.querySelectorAll("[data-view]")) {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  $("viewTitle").textContent = { workspace: "识别工作台", history: "本次识别记录", guide: "使用指南" }[view];
}

function renderControls() {
  const busy = state.running || state.sampleLoading;
  const pending = state.items.some((item) => ["pending", "error"].includes(item.status));
  for (const id of ["chooseFiles", "addFiles", "clearButton", "fileInput"]) $(id).disabled = busy;
  $("clearButton").disabled = busy || !state.items.length;
  $("startButton").disabled = busy || !pending || !state.ready;
  $("startLabel").textContent = state.running ? "正在识别…" : state.items.some((item) => item.status === "error") ? "开始 / 重试识别" : "开始识别";
  $("pauseButton").hidden = !state.running;
  $("pauseButton").disabled = state.stopping;
  $("pauseButton").textContent = state.stopping ? "当前张完成后暂停" : "完成当前张后暂停";
  $("threshold").disabled = busy || !state.ready;
  $("resetThreshold").disabled = busy || !state.ready || !state.thresholdAdjusted;
  $("reconnectButton").disabled = state.checking || state.running;
  $("dropZone").setAttribute("aria-disabled", String(busy));
  $("uploadPanel").classList.toggle("has-files", state.items.length > 0);
  $("queueCount").textContent = `${state.items.length} / ${LIMITS.count}`;
  for (const button of document.querySelectorAll("[data-sample]")) button.disabled = busy || state.items.length >= LIMITS.count;
  for (const button of document.querySelectorAll(".queue-remove")) button.disabled = busy;
  const counts = summarize(state.items);
  $("historyCount").textContent = String(counts.total);
  $("summaryTotal").textContent = String(counts.total);
  $("summaryPositive").textContent = String(counts.positive);
  $("summaryNegative").textContent = String(counts.negative);
  $("summaryErrors").textContent = String(counts.error);
  $("exportCsv").disabled = counts.total === 0;
  $("exportCsvHistory").disabled = counts.total === 0;
  $("progressWrap").hidden = !state.progress;
  if (state.progress) {
    const { finished, total } = state.progress;
    $("progressFill").style.width = `${total ? finished / total * 100 : 0}%`;
    $("progressBar").setAttribute("aria-valuemin", "0");
    $("progressBar").setAttribute("aria-valuemax", String(total));
    $("progressBar").setAttribute("aria-valuenow", String(finished));
    $("progressText").textContent = `${state.running ? "本轮进度" : finished < total ? "已暂停" : "本轮完成"} · ${finished} / ${total}`;
  }
}

function renderQueue() {
  const fragment = document.createDocumentFragment();
  for (const item of state.items) {
    const row = document.createElement("li");
    row.className = "queue-item";
    row.dataset.status = item.status;
    row.classList.toggle("selected", item.id === state.selectedId);
    const choose = document.createElement("button");
    choose.type = "button";
    choose.className = "queue-select";
    choose.setAttribute("aria-pressed", String(item.id === state.selectedId));
    const thumb = document.createElement("img");
    thumb.className = "queue-thumb";
    thumb.src = item.url;
    thumb.alt = "";
    const info = document.createElement("span");
    info.className = "queue-info";
    const name = document.createElement("strong");
    name.className = "queue-name";
    name.textContent = item.file.name;
    name.title = item.file.name;
    const status = document.createElement("span");
    status.className = "queue-status";
    status.textContent = item.status === "running" ? "正在识别…" : labels[resultKind(item)];
    info.append(name, status);
    choose.append(thumb, info);
    choose.addEventListener("click", () => {
      const restoreFocus = document.activeElement === choose;
      state.selectedId = item.id;
      renderQueue();
      renderResult();
      if (restoreFocus) document.querySelector(".queue-item.selected .queue-select")?.focus({ preventScroll: true });
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "queue-remove";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `移除 ${item.file.name}`);
    remove.disabled = state.running || state.sampleLoading;
    remove.addEventListener("click", () => removeItem(item.id));
    row.append(choose, remove);
    fragment.append(row);
  }
  $("queueList").replaceChildren(fragment);
}

function renderResult() {
  const item = selected();
  const step = state.running ? 1 : item && ["done", "error"].includes(item.status) ? 2 : 0;
  document.querySelectorAll(".workflow-steps li").forEach((element, index) => {
    const active = index === step;
    element.classList.toggle("current", active);
    if (active) element.setAttribute("aria-current", "step");
    else element.removeAttribute("aria-current");
  });
  $("resultEmpty").hidden = !!item;
  $("resultContent").hidden = !item;
  $("zoomButton").disabled = !item;
  $("downloadImage").disabled = item?.status !== "done";
  if (!item) {
    $("originalImage").removeAttribute("src");
    $("annotatedImage").removeAttribute("src");
    for (const button of document.querySelectorAll("[data-display]")) {
      button.disabled = true;
      const active = button.dataset.display === "annotated";
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    }
    return;
  }
  const done = item.status === "done";
  const kind = resultKind(item);
  const data = item.result;
  $("resultFilename").textContent = item.file.name;
  $("resultFilename").title = item.file.name;
  $("resultBadge").dataset.tone = kind;
  $("resultContent").dataset.tone = kind;
  $("resultBadge").textContent = item.status === "running" ? "识别中" : labels[kind];
  $("resultTitle").textContent = { positive: "疑似发现薇甘菊", negative: "本图未检出薇甘菊", error: "这张图片尚未完成判断", pending: item.status === "running" ? "正在查看叶片与藤蔓特征" : "照片已准备好" }[kind];
  $("resultExplanation").textContent = {
    positive: data?.count > 0 ? "绿色区域仅作定位参考，可能包含背景，不代表精确植株边界；判定得分不是正确率。" : "整图疑似含薇甘菊，但暂无达到展示条件的可靠区域；本次仅提供整图判断，判定得分不是正确率。",
    negative: "本图未达到当前判定阈值，不等于确认不存在薇甘菊。复杂背景或远距离航拍建议人工复核。",
    error: item.error || "识别失败，请重试。本次未作阳性或阴性判断。",
    pending: item.status === "running" ? "正在上传并分析，请稍候。暂停将在当前照片完成后生效。" : "点击“开始识别”，查看照片级判断与局部参考区域。",
  }[kind];
  $("statRegions").textContent = done ? String(data.count) : "—";
  $("statScore").textContent = done ? formatScore(data.score) : "—";
  $("statScore").title = done && !Number.isFinite(data.score) ? "当前模型接口未提供照片级得分" : "模型判定得分，不是识别准确率";
  $("statTime").textContent = done ? formatTime(data.latency_ms) : "—";
  $("statThreshold").textContent = done ? formatThreshold(data.threshold) : "—";
  $("statModel").textContent = done ? data.model : "—";
  $("originalImage").src = item.url;
  $("originalImage").alt = `原图：${item.file.name}`;
  if (done) { $("annotatedImage").src = data.annotated_image; $("annotatedImage").alt = `识别参考图：${item.file.name}`; }
  else $("annotatedImage").removeAttribute("src");
  const display = done ? state.display : "original";
  for (const button of document.querySelectorAll("[data-display]")) {
    const active = button.dataset.display === display;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = !done && button.dataset.display !== "original";
  }
  for (const mode of ["original", "annotated", "compare"]) $("resultCanvas").classList.toggle(`display-${mode}`, mode === display);
  $("originalFrame").hidden = display === "annotated";
  $("annotatedFrame").hidden = display === "original";
}

function renderHistory() {
  const rows = filterHistory(state.items, state.filter, $("historySearch").value).slice().reverse();
  $("historyEmpty").hidden = rows.length > 0;
  $("historyTable").hidden = rows.length === 0;
  const fragment = document.createDocumentFragment();
  for (const item of rows) {
    const row = document.createElement("tr");
    const kind = resultKind(item), data = item.result;
    const cells = [item.file.name, labels[kind], item.status === "done" ? data.count : "—", formatScore(data?.score), formatTime(data?.latency_ms)];
    for (const value of cells) { const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell); }
    row.dataset.tone = kind;
    const cell = document.createElement("td"), button = document.createElement("button");
    button.type = "button";
    button.className = "history-open";
    button.dataset.id = item.id;
    button.textContent = "查看";
    button.setAttribute("aria-label", `查看 ${item.file.name} 的识别记录`);
    button.addEventListener("click", () => { state.selectedId = item.id; switchView("workspace"); renderQueue(); renderResult(); $("resultTitle").focus({ preventScroll: true }); });
    cell.append(button);
    row.append(cell);
    fragment.append(row);
  }
  $("historyTableBody").replaceChildren(fragment);
  for (const button of document.querySelectorAll("[data-filter]")) { const active = state.filter === button.dataset.filter; button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active)); }
}

function render() { renderQueue(); renderControls(); renderResult(); renderHistory(); }

function addFiles(files) {
  if (state.running) return;
  const { accepted, rejected } = validateFiles(state.items.map((item) => item.file), Array.from(files));
  const added = [];
  for (const original of accepted) {
    // Empty browser MIME metadata is repaired only for an explicitly supported extension.
    const file = original.type ? original : new File([original], original.name, { type: supportedMime(original), lastModified: original.lastModified });
    const item = { id: String(nextId++), file, url: URL.createObjectURL(file), status: "pending", result: null, error: "", completedAt: null };
    state.items.push(item);
    added.push(item);
  }
  if (added.length) { state.selectedId = added[0].id; state.progress = null; switchView("workspace"); }
  const errors = rejected.slice(0, 3).map((entry) => `${entry.name}：${entry.reason}`).join("；");
  notify([added.length ? `已加入 ${added.length} 张图片，可开始识别。` : "", errors, rejected.length > 3 ? `另有 ${rejected.length - 3} 张未加入。` : ""].filter(Boolean).join(" "), rejected.length ? "warning" : "info");
  $("fileInput").value = "";
  render();
}

function closeZoom() { if ($("imageDialog").open) $("imageDialog").close(); $("dialogImage").removeAttribute("src"); }

function removeItem(id) {
  if (state.running || state.sampleLoading) return;
  const item = state.items.find((entry) => entry.id === id);
  if (!item) return;
  const restoreFocus = document.activeElement?.classList.contains("queue-remove");
  closeZoom();
  URL.revokeObjectURL(item.url);
  state.items = state.items.filter((entry) => entry.id !== id);
  if (state.selectedId === id) state.selectedId = state.items[0]?.id || null;
  state.progress = null;
  notify("已移除图片及其本次记录；原始文件不受影响。");
  render();
  if (restoreFocus) (document.querySelector(".queue-item.selected .queue-select") || $("chooseFiles")).focus({ preventScroll: true });
}

function resetThreshold() {
  const value = state.health?.default_threshold;
  if (Number.isFinite(value)) $("threshold").value = String(value * 100);
  $("thresholdValue").textContent = formatThreshold(value);
  $("settingStatus").textContent = "推荐设置";
  state.thresholdAdjusted = false;
  renderControls();
}

async function loadHealth() {
  if (state.checking || state.running) return;
  state.checking = true;
  state.ready = false;
  $("healthIndicator").dataset.state = "loading";
  $("healthText").textContent = "正在连接识别服务";
  renderControls();
  const controller = new AbortController(), timer = window.setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch("/api/v1/health", { signal: controller.signal, cache: "no-store" });
    if (!response.ok) throw new Error("识别服务暂不可用");
    const data = await response.json();
    state.health = data;
    state.ready = data.status === "ready";
    $("healthIndicator").dataset.state = state.ready ? "ready" : "offline";
    $("healthText").textContent = state.ready ? "识别服务已就绪" : "模型尚未准备完成";
    $("modelName").textContent = typeof data.model === "string" ? data.model : "—";
    $("modelNotice").hidden = !String(data.model || "").startsWith("mikania-ensemble-v4.1");
    $("connectionBanner").hidden = state.ready;
    if (!state.thresholdAdjusted) resetThreshold();
  } catch {
    $("healthIndicator").dataset.state = "offline";
    $("healthText").textContent = "暂时无法连接服务";
    $("connectionBanner").hidden = false;
  } finally { window.clearTimeout(timer); state.checking = false; renderControls(); }
}

async function loadSample(id) {
  if (state.running || state.sampleLoading) return;
  state.sampleLoading = true;
  renderControls();
  notify("正在读取真实示例图片…");
  const controller = new AbortController(), timer = window.setTimeout(() => controller.abort(), 20000);
  try {
    if (!exampleCatalog) {
      const response = await fetch("/api/v1/examples", { signal: controller.signal });
      if (!response.ok) throw new Error("示例图片暂时无法加载，请上传自己的照片。");
      const data = await response.json();
      if (!Array.isArray(data.examples)) throw new Error("示例列表不完整，请上传自己的照片。");
      exampleCatalog = data.examples;
    }
    const example = exampleCatalog.find((entry) => entry.id === id);
    if (!example || typeof example.url !== "string") throw new Error("未找到该示例图片，请上传自己的照片。");
    const url = new URL(example.url, window.location.href);
    if (url.origin !== window.location.origin) throw new Error("示例图片地址无效。");
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error("示例图片加载失败，请稍后重试。");
    const blob = await response.blob();
    addFiles([new File([blob], example.filename || `example-${id}.jpg`, { type: blob.type, lastModified: 0 })]);
  } catch (error) { notify(error.name === "AbortError" ? "示例图片加载超时，请重试或上传自己的照片。" : error.message, "error"); }
  finally { window.clearTimeout(timer); state.sampleLoading = false; renderControls(); }
}

async function startBatch() {
  if (state.running || state.sampleLoading || !state.ready) return;
  const batch = state.items.filter((item) => ["pending", "error"].includes(item.status));
  if (!batch.length) return;
  const settings = batchSettings(state.health, state.thresholdAdjusted, $("threshold").value);
  state.running = true;
  state.stopping = false;
  state.progress = { finished: 0, total: batch.length };
  switchView("workspace");
  let errors = 0;
  try {
    for (const item of batch) {
      if (state.stopping) break;
      item.status = "running";
      item.error = "";
      item.result = null;
      item.completedAt = null;
      state.selectedId = item.id;
      notify(`正在识别第 ${state.progress.finished + 1} / ${batch.length} 张：${item.file.name}`);
      render();
      const payload = new FormData();
      payload.append("file", item.file);
      if (settings.scan_mode) payload.append("scan_mode", settings.scan_mode);
      if (settings.threshold !== undefined) payload.append("threshold", String(settings.threshold));
      try {
        const response = await fetch("/api/v1/predict", { method: "POST", body: payload });
        let data;
        try { data = await response.json(); } catch { throw new Error("服务返回异常，请检查连接后重试。本次未作判断。"); }
        if (!response.ok) {
          if (response.status === 503) {
            state.ready = false;
            state.stopping = true;
            $("healthIndicator").dataset.state = "offline";
            $("healthText").textContent = "模型暂不可用";
            $("connectionBanner").hidden = false;
          }
          throw new Error(typeof data.detail === "string" ? data.detail : `识别未完成（${response.status}），请重试。`);
        }
        item.result = normalizePrediction(data);
        item.status = "done";
      } catch (error) {
        item.status = "error";
        item.error = error instanceof TypeError ? "网络连接中断，请检查连接并重新连接后重试；本次未作判断。" : error.message || "识别失败，请重试；本次未作判断。";
        errors += 1;
        if (error instanceof TypeError) {
          state.ready = false;
          state.stopping = true;
          $("healthIndicator").dataset.state = "offline";
          $("healthText").textContent = "识别连接已中断";
          $("connectionBanner").hidden = false;
        }
      }
      item.completedAt = new Date().toISOString();
      state.progress.finished += 1;
      render();
    }
  } finally {
    state.running = false;
    const paused = state.progress.finished < state.progress.total;
    notify(paused ? `本轮已暂停，完成 ${state.progress.finished} / ${state.progress.total} 张；剩余图片未提交。${errors ? ` 其中 ${errors} 张失败，可重试。` : ""}` : `本轮已完成 ${state.progress.finished} 张。${errors ? `其中 ${errors} 张识别失败，未作判断，可重新尝试。` : "结果和导出文件仅保留在当前页面。"}`, errors ? "warning" : "info");
    state.stopping = false;
    render();
  }
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob), anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadImage() {
  const item = selected();
  if (item?.status !== "done") return;
  const info = imageDataInfo(item.result.annotated_image);
  if (!info) { notify("结果图片格式异常，无法下载。", "error"); return; }
  try {
    const binary = atob(info.base64), bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    saveBlob(new Blob([bytes], { type: info.mime }), safeFilename(item.file.name, info.extension));
    notify("已准备好识别参考图，请在浏览器下载中查看。");
  } catch { notify("图片下载失败，请重新识别后重试。", "error"); }
}

function exportCsv() {
  if (!summarize(state.items).total) return;
  saveBlob(new Blob([buildCsv(state.items)], { type: "text/csv;charset=utf-8" }), `薇迹_本次识别记录_${new Date().toISOString().slice(0, 10)}.csv`);
  notify("已准备好本次识别记录，包含失败记录；得分不是准确率。");
}

document.addEventListener("click", (event) => { const navigation = event.target.closest("[data-view]"); if (navigation) switchView(navigation.dataset.view); });
for (const button of document.querySelectorAll("[data-display]")) button.addEventListener("click", () => { state.display = button.dataset.display; renderResult(); });
for (const button of document.querySelectorAll("[data-filter]")) button.addEventListener("click", () => { state.filter = button.dataset.filter; renderHistory(); });
for (const button of document.querySelectorAll("[data-sample]")) button.addEventListener("click", () => loadSample(button.dataset.sample));
for (const id of ["chooseFiles", "addFiles"]) $(id).addEventListener("click", (event) => { event.stopPropagation(); if (!state.running && !state.sampleLoading) $("fileInput").click(); });
$("fileInput").addEventListener("change", () => addFiles($("fileInput").files));
$("dropZone").addEventListener("click", (event) => { if (!event.target.closest("button") && !state.running && !state.sampleLoading) $("fileInput").click(); });
$("dropZone").addEventListener("keydown", (event) => {
  if (event.target !== $("dropZone") || !["Enter", " "].includes(event.key)) return;
  event.preventDefault();
  if (!state.running && !state.sampleLoading) $("fileInput").click();
});
for (const name of ["dragenter", "dragover"]) $("dropZone").addEventListener(name, (event) => { event.preventDefault(); if (!state.running && !state.sampleLoading) $("dropZone").classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) $("dropZone").addEventListener(name, (event) => { event.preventDefault(); $("dropZone").classList.remove("dragging"); });
$("dropZone").addEventListener("drop", (event) => { if (!state.running && !state.sampleLoading) addFiles(event.dataTransfer.files); });
// Dropping outside the upload target must not navigate away and lose this session.
window.addEventListener("dragover", (event) => { if (event.dataTransfer?.types.includes("Files")) event.preventDefault(); });
window.addEventListener("drop", (event) => { if (event.dataTransfer?.types.includes("Files")) event.preventDefault(); });
$("threshold").addEventListener("input", () => { state.thresholdAdjusted = true; $("thresholdValue").textContent = formatThreshold(Number($("threshold").value) / 100); $("settingStatus").textContent = "已自定义 · 降低阈值可能增加误报"; renderControls(); });
$("resetThreshold").addEventListener("click", resetThreshold);
$("reconnectButton").addEventListener("click", loadHealth);
$("startButton").addEventListener("click", startBatch);
$("pauseButton").addEventListener("click", () => { state.stopping = true; notify("将在当前图片完成后暂停；当前上传与服务器识别不会被取消。"); renderControls(); });
$("clearButton").addEventListener("click", () => {
  if (state.running || state.sampleLoading) return;
  if (summarize(state.items).total && !window.confirm("清空图片会同时清除本次识别记录。需要保留的结果请先下载。确认清空吗？")) return;
  closeZoom();
  for (const item of state.items) URL.revokeObjectURL(item.url);
  state.items = [];
  state.selectedId = null;
  state.progress = null;
  state.filter = "all";
  state.display = "annotated";
  $("historySearch").value = "";
  notify("本次图片与记录已清空，原始文件不受影响。");
  render();
  $("chooseFiles").focus({ preventScroll: true });
});
$("historySearch").addEventListener("input", renderHistory);
$("downloadImage").addEventListener("click", downloadImage);
$("exportCsv").addEventListener("click", exportCsv);
$("exportCsvHistory").addEventListener("click", exportCsv);
$("zoomButton").addEventListener("click", () => {
  const item = selected();
  if (!item) return;
  const showOriginal = item.status !== "done" || state.display === "original";
  $("dialogImage").src = showOriginal ? item.url : item.result.annotated_image;
  $("dialogImage").alt = `${showOriginal ? "原图" : "识别参考图"}：${item.file.name}`;
  $("dialogTitle").textContent = `${item.file.name} · ${showOriginal ? "原图" : "识别参考图"}`;
  $("imageDialog").showModal();
});
$("closeDialog").addEventListener("click", closeZoom);
$("imageDialog").addEventListener("click", (event) => { if (event.target === $("imageDialog")) closeZoom(); });
$("imageDialog").addEventListener("close", () => $("dialogImage").removeAttribute("src"));
window.addEventListener("beforeunload", (event) => { if (state.running) { event.preventDefault(); event.returnValue = ""; } });
window.addEventListener("pagehide", (event) => { if (!event.persisted) for (const item of state.items) URL.revokeObjectURL(item.url); });
switchView("workspace");
render();
loadHealth();
