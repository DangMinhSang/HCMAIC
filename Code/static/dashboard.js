(() => {
  "use strict";

  const api = window.AIC_CONFIG;
  const $ = (selector) => document.querySelector(selector);
  const state = {
    task: "kis", results: [], selected: new Set(), query: "", detailId: null, searchToken: 0,
  };
  const taskCopy = {
    kis: {
      kicker: "TEXTUAL KNOWN ITEM SEARCH",
      title: "Tìm đúng khoảnh khắc",
      help: "Mô tả sự kiện bằng tiếng Việt hoặc tiếng Anh. Hệ thống kết hợp CLIP, OCR và Qwen.",
      placeholder: "Ví dụ: Biển màu vàng có nội dung cảnh báo sạt lở nguy hiểm",
      hints: [
        ["Biển cảnh báo", "Biển màu vàng có nội dung cảnh báo sạt lở nguy hiểm"],
        ["Họp báo", "Một người đang phát biểu trước nhiều micro tại họp báo"],
      ],
    },
    qa: {
      kicker: "VISUAL QUESTION ANSWERING",
      title: "Tìm cảnh và trả lời",
      help: "Nhập mô tả sự kiện, sau đó ghi “Câu hỏi:” để tách phần cần trả lời.",
      placeholder: "Cảnh trao giải âm nhạc. Câu hỏi: Có bao nhiêu người trên sân khấu?",
      hints: [
        ["Đếm người", "Cảnh trao giải. Câu hỏi: Có bao nhiêu người trên sân khấu?"],
        ["Màu sắc", "Người phụ nữ cầm một chiếc ly. Câu hỏi: Chiếc ly màu gì?"],
      ],
    },
    trake: {
      kicker: "TEMPORAL RETRIEVAL & ALIGNMENT",
      title: "Căn chỉnh chuỗi sự kiện",
      help: "Mỗi dòng là một semantic keyframe theo đúng thứ tự thời gian. Hệ thống dùng alignment động và Qwen.",
      placeholder: "Vận động viên chạy đà\nVận động viên giậm nhảy\nVận động viên bay qua xà\nVận động viên tiếp đất",
      hints: [
        ["Nhảy cao", "Vận động viên chạy đà\nVận động viên giậm nhảy\nVận động viên bay qua xà\nVận động viên tiếp đất"],
        ["Mở laptop", "Người đặt laptop lên bàn\nNgười mở nắp laptop\nMàn hình laptop sáng lên"],
      ],
    },
  };

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    })[char]);
  }

  function toast(message) {
    const node = $("#toast");
    node.textContent = message;
    node.classList.add("show");
    clearTimeout(window.aicToastTimer);
    window.aicToastTimer = setTimeout(() => node.classList.remove("show"), 4200);
  }

  async function jsonResponse(response, fallback) {
    const body = await response.text();
    let data;
    try {
      data = JSON.parse(body);
    } catch (_) {
      if (response.status === 504) {
        throw new Error("Gateway tạm hết thời gian ở request trạng thái; hệ thống sẽ an toàn khi thử lại.");
      }
      const type = response.headers.get("content-type") || "không rõ";
      throw new Error(`API trả HTML thay vì JSON (HTTP ${response.status}, ${type}).`);
    }
    if (!response.ok) throw new Error(data?.error || fallback);
    return data;
  }

  const wait = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

  async function waitForSearch(statusUrl, token) {
    let transientFailures = 0;
    while (token === state.searchToken) {
      await wait(650);
      try {
        const response = await fetch(statusUrl, { cache: "no-store" });
        const data = await jsonResponse(response, "Không đọc được trạng thái query");
        transientFailures = 0;
        if (data.status === "complete") return data;
        if (data.status === "error") {
          const failure = new Error(data.error || "Truy vấn thất bại.");
          failure.queryFailed = true;
          throw failure;
        }
        const elapsed = Number(data.elapsed_ms) > 0 ? ` · ${(data.elapsed_ms / 1000).toFixed(1)}s` : "";
        $("#notice").textContent = `${data.stage || "Đang xử lý…"}${elapsed}`;
      } catch (error) {
        if (error.queryFailed) throw error;
        transientFailures += 1;
        if (transientFailures >= 5) throw error;
        $("#notice").textContent = `Mất kết nối tạm thời, đang thử lại (${transientFailures}/5)…`;
        await wait(900);
      }
    }
    throw new Error("Query đã được thay thế bởi một yêu cầu mới.");
  }

  function scorePercent(value) {
    const score = Number(value) || 0;
    return Math.max(0, Math.min(100, Math.round(score * 100)));
  }

  function scoreRow(label, value, kind = "") {
    const percent = scorePercent(value);
    return `<div class="score-row ${kind}"><span>${label}</span><b><i style="--score:${percent}%"></i></b><em>${percent}%</em></div>`;
  }

  function toggleItem(item) {
    const targets = state.task === "trake"
      ? state.results.filter((candidate) => candidate.rank === item.rank)
      : [item];
    const allSelected = targets.every((candidate) => state.selected.has(candidate.id));
    targets.forEach((candidate) => allSelected ? state.selected.delete(candidate.id) : state.selected.add(candidate.id));
    render();
    updateSelection();
  }

  function cardMarkup(item) {
    const selected = state.selected.has(item.id);
    const ai = item.ai_joint_score || item.ai_score;
    const title = item.title || "Không có metadata tiêu đề";
    return `<article class="result-card ${selected ? "selected" : ""} ${item.rank === 1 ? "rank-one" : ""}" data-id="${escapeHtml(item.id)}" title="Nhấp để chọn · nhấp đúp để xem chi tiết">
      <div class="image-wrap">
        <img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.video_id)} frame ${item.frame_id}" loading="lazy">
        <span class="rank-badge">${item.rank}</span>
        ${item.event ? `<span class="event-badge">Event ${item.event}</span>` : ""}
        <span class="selected-mark">✓</span>
      </div>
      <div class="card-body">
        <div class="card-id"><span>${escapeHtml(item.video_id)} · F${item.frame_id}</span><time>${escapeHtml(item.time)}</time></div>
        <p class="card-title">${escapeHtml(title)}</p>
        ${ai ? scoreRow("Qwen", ai, "ai") : ""}
        ${scoreRow("CLIP", item.clip)}
        ${item.ocr_score ? scoreRow("OCR", item.ocr_score, "ocr") : ""}
        ${item.ocr_text ? `<p class="ocr-snippet">${escapeHtml(item.ocr_text)}</p>` : ""}
        ${item.answer ? `<p class="answer-line"><strong>Answer:</strong> ${escapeHtml(item.answer)} · ${scorePercent(item.qa_confidence)}%</p>` : ""}
      </div>
    </article>`;
  }

  function bindCards() {
    document.querySelectorAll(".result-card").forEach((card) => {
      card.addEventListener("click", () => {
        const item = state.results.find((result) => result.id === card.dataset.id);
        if (item) toggleItem(item);
      });
      card.addEventListener("dblclick", (event) => {
        event.preventDefault();
        const item = state.results.find((result) => result.id === card.dataset.id);
        if (item) openInspector(item);
      });
    });
  }

  function render() {
    const grid = $("#grid");
    $("#empty").hidden = state.results.length > 0;
    if (!state.results.length) {
      grid.replaceChildren();
      return;
    }
    if (state.task !== "trake") {
      grid.innerHTML = state.results.map(cardMarkup).join("");
    } else {
      const groups = new Map();
      state.results.forEach((item) => {
        if (!groups.has(item.rank)) groups.set(item.rank, []);
        groups.get(item.rank).push(item);
      });
      grid.innerHTML = [...groups.entries()].map(([rank, items]) => `
        <section class="sequence-group">
          <header class="sequence-head"><h3>Chuỗi #${rank} · ${escapeHtml(items[0].video_id)}</h3><span>${items.length} semantic keyframe</span></header>
          <div class="sequence-cards">${items.map(cardMarkup).join("")}</div>
        </section>`).join("");
    }
    bindCards();
  }

  function updateSelection() {
    $("#export span").textContent = state.task === "trake"
      ? new Set(state.results.filter((item) => state.selected.has(item.id)).map((item) => item.rank)).size
      : state.selected.size;
  }

  function openInspector(item) {
    $("#detail-image").src = item.image_url;
    $("#detail-rank").textContent = `RANK ${item.rank}${item.event ? ` · EVENT ${item.event}` : ""}`;
    $("#detail-id").textContent = `${item.video_id} · frame ${item.frame_id}`;
    $("#detail-title").textContent = item.title || "Không có metadata tiêu đề.";
    const metrics = [
      ["Final", item.score], ["Qwen", item.ai_joint_score || item.ai_score],
      ["CLIP", item.clip], ["OCR", item.ocr_score],
      ["Metadata", item.metadata_score], ["Keyframe", item.keyframe_number],
    ];
    $("#detail-metrics").innerHTML = metrics.map(([label, value]) => `<div><dt>${label}</dt><dd>${typeof value === "number" ? value.toFixed(3) : escapeHtml(value)}</dd></div>`).join("");
    $("#detail-ocr").textContent = item.ocr_text || "Không có OCR.";
    $("#detail-objects").textContent = item.objects?.length ? item.objects.join(", ") : "Không có object metadata.";
    $("#detail-frame-input").value = item.frame_id;
    state.detailId = item.id;
    const videoLink = $("#detail-video");
    videoLink.hidden = !item.video_url;
    videoLink.href = item.video_url ? `${item.video_url}#t=${Math.max(0, Number(item.pts_time) || 0)}` : "";
    $("#inspector").classList.add("open");
    $("#inspector").setAttribute("aria-hidden", "false");
    $("#drawer-backdrop").classList.add("show");
  }

  function closeInspector() {
    $("#inspector").classList.remove("open");
    $("#inspector").setAttribute("aria-hidden", "true");
    $("#drawer-backdrop").classList.remove("show");
    state.detailId = null;
  }

  function options() {
    return {
      top_k: Math.max(1, Math.min(100, Number($("#topk").value) || 100)),
      dedupe: Math.max(0, Number($("#dedupe").value) || 0),
      max_per_video: Math.max(0, Number($("#pervideo").value) || 0),
      video_id: $("#video").value,
    };
  }

  async function search() {
    const query = $("#query").value.trim();
    if (!query) {
      toast("Hãy nhập truy vấn trước.");
      $("#query").focus();
      return;
    }
    const button = $("#search");
    const requestedTask = state.task;
    const token = ++state.searchToken;
    button.classList.add("loading");
    button.disabled = true;
    button.querySelector("span").textContent = "Đang phân tích…";
    $("#notice").className = "notice";
    $("#notice").textContent = "Recall CLIP/OCR → Qwen rerank → đa dạng hóa kết quả…";
    try {
      const response = await fetch(api.search, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: state.task, query, options: options() }),
      });
      const accepted = await jsonResponse(response, "Truy vấn thất bại");
      const data = accepted.status_url ? await waitForSearch(accepted.status_url, token) : accepted;
      if (requestedTask !== state.task || token !== state.searchToken) return;
      state.query = query;
      state.results = data.results || [];
      state.selected.clear();
      $("#count").textContent = `${state.results.length} kết quả · ${(data.elapsed_ms / 1000).toFixed(2)} giây`;
      $("#query-summary").hidden = false;
      $("#query-summary").textContent = query;
      $("#notice").textContent = data.notice || "Hoàn thành.";
      render();
      updateSelection();
    } catch (error) {
      if (token !== state.searchToken) return;
      state.results = [];
      state.selected.clear();
      $("#count").textContent = "Truy vấn chưa hoàn thành";
      $("#notice").className = "notice error";
      $("#notice").textContent = error.message;
      render();
      updateSelection();
      toast(error.message);
    } finally {
      if (token === state.searchToken) {
        button.classList.remove("loading");
        button.disabled = false;
        button.querySelector("span").textContent = "Tìm kiếm";
      }
    }
  }

  function setTask(task) {
    state.searchToken += 1;
    const searchButton = $("#search");
    searchButton.classList.remove("loading");
    searchButton.disabled = false;
    searchButton.querySelector("span").textContent = "Tìm kiếm";
    state.task = task;
    state.results = [];
    state.selected.clear();
    document.querySelectorAll(".task-tab").forEach((button) => button.classList.toggle("active", button.dataset.task === task));
    const copy = taskCopy[task];
    $("#task-kicker").textContent = copy.kicker;
    $("#task-title").textContent = copy.title;
    $("#task-help").textContent = copy.help;
    $("#query").placeholder = copy.placeholder;
    $("#query").value = "";
    $("#query-hints").innerHTML = copy.hints.map(([label, value]) => `<button data-example="${escapeHtml(value)}">${escapeHtml(label)}</button>`).join("");
    bindHints();
    $("#count").textContent = "Sẵn sàng tìm kiếm";
    $("#notice").textContent = task === "trake" ? "Nhập ít nhất hai event, mỗi event một dòng." : "Nhập mô tả để bắt đầu.";
    $("#query-summary").hidden = true;
    render();
    updateSelection();
  }

  function bindHints() {
    $("#query-hints").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
      $("#query").value = button.dataset.example;
      $("#query").focus();
    }));
  }

  function selectTop(limit) {
    state.selected.clear();
    state.results.forEach((item) => {
      if (item.rank <= limit) state.selected.add(item.id);
    });
    render();
    updateSelection();
  }

  async function exportCsv() {
    if (!state.selected.size) {
      toast("Hãy chọn ít nhất một kết quả.");
      return;
    }
    try {
      const response = await fetch(api.export, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected: [...state.selected],
          frame_overrides: Object.fromEntries(
            state.results.filter((item) => item.frameOverride && state.selected.has(item.id)).map((item) => [item.id, item.frame_id]),
          ),
        }),
      });
      if (!response.ok) await jsonResponse(response, "Xuất CSV thất bại");
      const blob = await response.blob();
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = state.task === "trake" ? "aic_trake.csv" : `aic_${state.task}.csv`;
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (error) {
      toast(error.message);
    }
  }

  async function loadHealth() {
    try {
      const response = await fetch(api.health);
      const data = await jsonResponse(response, "Không đọc được trạng thái hệ thống");
      $("#corpus").textContent = `${data.vector_count.toLocaleString("vi-VN")} vectors · ${data.video_count} videos · ${data.ocr_records.toLocaleString("vi-VN")} OCR`;
      const model = $("#model-status");
      model.classList.remove("waiting");
      model.classList.add(data.reranker_ready ? "ready" : "fallback");
      model.lastChild.textContent = data.reranker_ready ? "Qwen sẵn sàng" : "CLIP/OCR fallback";
      model.title = data.reranker_error || "Feature đã preload trong RAM";
      const select = $("#video");
      data.video_ids.forEach((id) => select.add(new Option(id, id)));
    } catch (error) {
      $("#model-status").classList.add("fallback");
      $("#model-status").lastChild.textContent = "Backend chưa sẵn sàng";
      $("#notice").className = "notice error";
      $("#notice").textContent = error.message;
    }
  }

  document.querySelectorAll(".task-tab").forEach((button) => button.addEventListener("click", () => setTask(button.dataset.task)));
  document.querySelectorAll("[data-select-top]").forEach((button) => button.addEventListener("click", () => selectTop(Number(button.dataset.selectTop))));
  $("#clear").addEventListener("click", () => { state.selected.clear(); render(); updateSelection(); });
  $("#search").addEventListener("click", search);
  $("#export").addEventListener("click", exportCsv);
  $("#query").addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") search();
  });
  $("#gap-presets").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    $("#dedupe").value = button.dataset.gap;
    $("#gap-presets").querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
  }));
  $("#inspector-close").addEventListener("click", closeInspector);
  $("#drawer-backdrop").addEventListener("click", closeInspector);
  $("#apply-frame").addEventListener("click", () => {
    const item = state.results.find((result) => result.id === state.detailId);
    const frame = Number($("#detail-frame-input").value);
    if (!item || !Number.isInteger(frame) || frame < 0) {
      toast("Frame nộp phải là số nguyên không âm.");
      return;
    }
    item.frame_id = frame;
    item.frameOverride = true;
    render();
    toast(`Đã override ${item.video_id} → frame ${frame}`);
  });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeInspector(); });

  bindHints();
  updateSelection();
  loadHealth();
})();
