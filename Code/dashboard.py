"""Dark, card-based AIC 2026 dashboard matching the competition workflow."""

from __future__ import annotations

import argparse
import csv
import io
import os
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request, send_file, session

from data_paths import DatasetNotFoundError
from ocr_index import OCRMemoryIndex
from qa import VQABaseline, split_qa_query
from retrieval import AICRetrievalEngine, SearchResult, TrakeVideoResult


app = Flask(__name__)
app.secret_key = os.environ.get("AIC_WEB_SECRET", secrets.token_urlsafe(32))

ENGINE: AICRetrievalEngine | None = None
VQA: VQABaseline | None = None
OCR_INDEX: OCRMemoryIndex | None = None
OCR_INDEX_LOADED = False
ENGINE_LOCK = threading.RLock()
SESSIONS: dict[str, "SearchSession"] = {}
SESSIONS_LOCK = threading.RLock()


@dataclass
class StoredResult:
    result: SearchResult
    group: str | None = None
    event_index: int | None = None


@dataclass
class SearchSession:
    task: str = "kis"
    results: dict[str, StoredResult] = field(default_factory=dict)
    trake_sequences: dict[str, TrakeVideoResult] = field(default_factory=dict)


def get_engine() -> AICRetrievalEngine:
    global ENGINE
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = AICRetrievalEngine.from_environment()
        return ENGINE


def get_vqa() -> VQABaseline:
    global VQA
    with ENGINE_LOCK:
        if VQA is None:
            VQA = VQABaseline()
        return VQA


def get_ocr_index() -> OCRMemoryIndex | None:
    """Load text-only OCR postings into RAM once, before the first query."""
    global OCR_INDEX, OCR_INDEX_LOADED
    with ENGINE_LOCK:
        if OCR_INDEX_LOADED:
            return OCR_INDEX
        OCR_INDEX_LOADED = True
        configured = os.environ.get("AIC_OCR_INDEX")
        default_path = Path("/kaggle/working/aic_ocr_index.jsonl.gz")
        index_path = Path(configured) if configured else default_path
        if index_path.is_file():
            OCR_INDEX = OCRMemoryIndex.load(index_path)
        return OCR_INDEX


def current_session() -> SearchSession:
    identifier = session.get("aic_session")
    if not identifier:
        identifier = uuid.uuid4().hex
        session["aic_session"] = identifier
    with SESSIONS_LOCK:
        return SESSIONS.setdefault(identifier, SearchSession())


def time_label(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0), 60)
    return f"{int(minutes):02d}:{rest:06.3f}"


def language_note(engine: AICRetrievalEngine) -> str:
    info = getattr(engine.encoder, "last_query", None)
    if info is None or info.language != "vi":
        return "Đã nhận dạng English."
    if info.translation_used:
        return f"VI → EN: {info.text_for_model}"
    return info.warning or "Đang dùng truy vấn tiếng Việt gốc."


def vector_count_compat(engine: AICRetrievalEngine) -> int:
    """Support a notebook process that still has an older retrieval module cached."""
    value = getattr(engine, "vector_count", None)
    if value is not None:
        return int(value)
    try:
        import numpy as np

        return sum(int(np.load(path, mmap_mode="r").shape[0]) for path in engine._features.values())
    except (AttributeError, OSError, ValueError):
        return 0


def as_payload(identifier: str, stored: StoredResult) -> dict[str, Any]:
    result = stored.result
    return {
        "id": identifier,
        "rank": result.rank,
        "video_id": result.video_id,
        "frame_id": result.frame_id,
        "time": time_label(result.pts_time),
        "score": round(result.score, 3),
        "clip": round(result.visual_score, 3),
        "title": result.title,
        "answer": getattr(result, "answer", ""),
        "ocr_score": round(getattr(result, "ocr_score", 0.0), 3),
        "ocr_text": getattr(result, "ocr_text", ""),
        "event": stored.event_index,
        # Relative URLs work both locally at / and when this Flask app is
        # mounted under /dashboard by the Kaggle Gradio share gateway.
        "image_url": f"media/{identifier}",
    }


def search_options(body: dict[str, Any]) -> tuple[int, int, int | None, str | None]:
    options = body.get("options") or {}
    top_k = max(1, min(int(options.get("top_k", 16)), 100))
    min_gap = max(0, min(int(options.get("dedupe", 0)), 600))
    max_per_video_raw = int(options.get("max_per_video", 3))
    max_per_video = max(1, min(max_per_video_raw, 100)) if max_per_video_raw else None
    video_id = str(options.get("video_id") or "").strip() or None
    return top_k, min_gap, max_per_video, video_id


def make_kis_results(engine: AICRetrievalEngine, query: str, body: dict[str, Any]) -> tuple[list[StoredResult], str]:
    top_k, min_gap, maximum, video_id = search_options(body)
    try:
        results = engine.search(
            query,
            top_k=top_k,
            min_frame_gap=min_gap,
            max_per_video=maximum,
            video_id=video_id,
            metadata_weight=0.10,
        )
    except TypeError as error:
        # A running Kaggle kernel can retain an older retrieval module in
        # sys.modules after git pull. Keep the dashboard usable until restart.
        if "unexpected keyword" not in str(error):
            raise
        results = engine.search(query, top_k=top_k, min_frame_gap=min_gap, metadata_weight=0.10)
        if video_id:
            results = [result for result in results if result.video_id == video_id]
        if maximum:
            counts: dict[str, int] = {}
            filtered = []
            for result in results:
                counts[result.video_id] = counts.get(result.video_id, 0) + 1
                if counts[result.video_id] <= maximum:
                    filtered.append(result)
            results = filtered
    ocr_index = get_ocr_index()
    if ocr_index is None:
        return [StoredResult(result) for result in results], language_note(engine) + " · OCR index chưa được nạp."

    hits = ocr_index.search(query, limit=max(top_k * 12, 100), video_id=video_id)
    combined = {(result.video_id, result.keyframe_number): result for result in results}
    for hit in hits:
        key = (hit.video_id, hit.keyframe_number)
        existing = combined.get(key)
        if existing is None:
            existing = engine.result_for_keyframe(
                hit.video_id,
                hit.keyframe_number,
                # Exact OCR text is a stronger signal than a generic scene
                # embedding for queries asking what a sign says.
                score=1.0 + hit.score,
                ocr_score=hit.score,
                ocr_text=hit.text,
            )
            if existing is None:
                continue
            combined[key] = existing
        else:
            existing.ocr_score = max(existing.ocr_score, hit.score)
            existing.ocr_text = hit.text
            existing.score += 1.0 + hit.score

    selected: list[SearchResult] = []
    frames_by_video: dict[str, list[int]] = {}
    for result in sorted(combined.values(), key=lambda item: item.score, reverse=True):
        nearby = frames_by_video.setdefault(result.video_id, [])
        if maximum is not None and len(nearby) >= maximum:
            continue
        if any(abs(result.frame_id - frame_id) <= min_gap for frame_id in nearby):
            continue
        nearby.append(result.frame_id)
        result.rank = len(selected) + 1
        selected.append(result)
        if len(selected) == top_k:
            break
    return [StoredResult(result) for result in selected], (
        language_note(engine) + f" · OCR RAM: {len(hits)} keyframe khớp chữ."
    )


def make_qa_results(engine: AICRetrievalEngine, query: str, body: dict[str, Any]) -> tuple[list[StoredResult], str]:
    event, question = split_qa_query(query)
    stored, note = make_kis_results(engine, event, body)
    results = [item.result for item in stored]
    try:
        predictions = get_vqa().predict(question, results)
        by_rank = {prediction.rank: prediction for prediction in predictions}
        for result in results:
            prediction = by_rank.get(result.rank)
            if prediction:
                result.answer = prediction.answer
                result.qa_confidence = prediction.confidence
        if predictions:
            best = predictions[0]
            for result in results:
                result.answer = result.answer or best.answer
            note += f" · VQA: {best.answer} ({best.confidence:.0%})"
        else:
            note += " · Không có ảnh để chạy VQA."
    except RuntimeError as error:
        # Retrieval must remain available if the optional VQA checkpoint is offline.
        note += f" · VQA chưa sẵn sàng: {error}"
    return stored, note


def make_trake_results(engine: AICRetrievalEngine, query: str, body: dict[str, Any]) -> tuple[list[StoredResult], dict[str, TrakeVideoResult], str]:
    events = [line.strip().lstrip("-0123456789. ") for line in query.splitlines() if line.strip()]
    top_k, _gap, _maximum, _video_id = search_options(body)
    sequences = engine.search_trake(events, top_videos=min(top_k, 50))
    stored: list[StoredResult] = []
    indexed: dict[str, TrakeVideoResult] = {}
    for sequence in sequences:
        group = f"trake-{sequence.rank}"
        indexed[group] = sequence
        for event_index, frame in enumerate(sequence.frames, start=1):
            stored.append(StoredResult(frame, group=group, event_index=event_index))
    return stored, indexed, "Mỗi card là một semantic keyframe; chọn một card để xuất cả chuỗi video TRAKE."


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    try:
        engine = get_engine()
        return jsonify(
            {
                "ok": True,
                "video_count": engine.video_count,
                "vector_count": vector_count_compat(engine),
                "video_ids": sorted(engine._features),
                "ocr_records": get_ocr_index().record_count if get_ocr_index() else 0,
            }
        )
    except (DatasetNotFoundError, OSError) as error:
        return jsonify({"ok": False, "error": str(error)}), 503


@app.post("/api/search")
def search():
    started = time.perf_counter()
    body = request.get_json(silent=True) or {}
    task = str(body.get("task") or "kis").lower()
    query = str(body.get("query") or "").strip()
    if task not in {"kis", "qa", "trake"}:
        return jsonify({"error": "Loại truy vấn không hợp lệ."}), 400
    if not query:
        return jsonify({"error": "Hãy nhập truy vấn."}), 400
    try:
        engine = get_engine()
        with ENGINE_LOCK:
            if task == "kis":
                stored, notice = make_kis_results(engine, query, body)
                sequences: dict[str, TrakeVideoResult] = {}
            elif task == "qa":
                stored, notice = make_qa_results(engine, query, body)
                sequences = {}
            else:
                stored, sequences, notice = make_trake_results(engine, query, body)
        state = current_session()
        state.task = task
        state.results.clear()
        state.trake_sequences = sequences
        payload: list[dict[str, Any]] = []
        for item in stored:
            identifier = uuid.uuid4().hex
            state.results[identifier] = item
            payload.append(as_payload(identifier, item))
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return jsonify({"results": payload, "elapsed_ms": elapsed_ms, "notice": notice, "task": task})
    except (DatasetNotFoundError, ValueError, RuntimeError, OSError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/media/<identifier>")
def media(identifier: str):
    stored = current_session().results.get(identifier)
    if stored is None or not stored.result.image_path:
        return "Not found", 404
    path = Path(stored.result.image_path)
    if not path.is_file():
        return "Not found", 404
    return send_file(path, conditional=True, max_age=3600)


@app.post("/api/export")
def export():
    body = request.get_json(silent=True) or {}
    selected = [str(value) for value in body.get("selected") or []]
    state = current_session()
    entries = [state.results[item] for item in selected if item in state.results]
    if not entries:
        return jsonify({"error": "Chọn ít nhất một kết quả trước khi xuất CSV."}), 400

    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    if state.task == "trake":
        groups = {entry.group for entry in entries if entry.group}
        sequences = [state.trake_sequences[group] for group in groups if group in state.trake_sequences]
        sequences.sort(key=lambda item: item.rank)
        width = max((len(item.frames) for item in sequences), default=0)
        writer.writerow(["video_id", *[f"frame_id_{index}" for index in range(1, width + 1)]])
        for item in sequences:
            writer.writerow([item.video_id, *[frame.frame_id for frame in item.frames]])
        name = "aic_trake.csv"
    else:
        entries.sort(key=lambda item: item.result.rank)
        is_qa = state.task == "qa"
        writer.writerow(["video_id", "frame_id", "answer"] if is_qa else ["video_id", "frame_id"])
        for entry in entries:
            row = [entry.result.video_id, entry.result.frame_id]
            if is_qa:
                row.append(entry.result.answer)
            writer.writerow(row)
        name = f"aic_{state.task}.csv"
    return Response(
        stream.getvalue().encode("utf-8-sig"),
        content_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIC26 Retrieval</title><style>
:root{--bg:#0d1117;--panel:#151c25;--panel2:#10161e;--line:#293545;--muted:#91a1b5;--text:#edf3fa;--teal:#35d1c0;--blue:#16a7df;--orange:#f6a313;--danger:#ec5b61}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:16px}.shell{min-height:100vh;display:grid;grid-template-columns:480px minmax(0,1fr)}aside{background:var(--panel);border-right:1px solid var(--line);min-height:100vh;position:sticky;top:0;height:100vh;overflow-y:auto}.brand{height:72px;display:flex;align-items:center;padding:0 24px;gap:12px;border-bottom:1px solid var(--line)}.dot{height:13px;width:13px;border-radius:50%;background:#21d56e}.brand b{font-size:25px;letter-spacing:.3px}.corpus{color:#8da1ba;font-size:15px}.side-body{padding:26px 24px}.eyebrow{color:#91a5c2;font-weight:800;font-size:14px;letter-spacing:.4px}.tabs{display:flex;gap:9px;margin:17px 0}.tab{height:50px;flex:1;border:1px solid #314052;border-radius:11px;color:#fff;background:#1a222d;font-size:16px;cursor:pointer}.tab.active{border-color:var(--teal);background:var(--teal);color:#062d2b;font-weight:800}.help{color:#b9c5d5;line-height:1.55;min-height:50px;margin:0 0 14px}.query{width:100%;height:112px;resize:vertical;background:#1c2632;border:1px solid #314052;border-radius:12px;padding:16px;color:#f1f6fb;font:inherit;outline:none}.query:focus,input:focus,select:focus{border-color:var(--teal);box-shadow:0 0 0 3px #35d1c022}.search{margin-top:26px;width:100%;height:64px;border:0;border-radius:12px;background:var(--teal);color:#062b2b;font-size:21px;font-weight:850;cursor:pointer}.search.loading{opacity:.65;cursor:wait}.options{margin-top:22px;border-top:1px solid var(--line);padding-top:23px}.option-title{color:#a6b4c7;font-weight:750}.changed{color:var(--orange)}.field{display:grid;grid-template-columns:138px 1fr;gap:12px;align-items:center;margin-top:12px;color:#aab8cc}.field input,.field select{height:56px;width:100%;border:1px solid #314052;border-radius:11px;background:#1b2430;color:#eff5fb;font:inherit;padding:0 20px}.fence{margin-top:24px;color:#aab8cc}.seg{display:flex;gap:9px;margin-top:11px}.seg button{height:43px;flex:1;background:#1b2430;border:1px solid #314052;border-radius:10px;color:#dce6f2;cursor:pointer;font:inherit}.seg button.active{background:var(--teal);border-color:var(--teal);color:#08312f;font-weight:800}main{min-width:0;display:flex;flex-direction:column}.main-head{height:72px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;padding:0 23px;background:#141a22;gap:15px}.result-count{color:#8fa1b8}.actions{display:flex;gap:17px}.small-btn{border:1px solid #2d3a4a;background:#1b2430;border-radius:11px;color:#eef4fb;padding:12px 20px;font:inherit;cursor:pointer}.export{background:var(--blue);border-color:var(--blue);color:#03273d;font-weight:800}.content{padding:24px 21px 32px}.chip{display:none;width:max-content;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border:1px solid #2b394a;background:#1a2430;border-radius:9px;color:#93a9c3;padding:7px 13px;margin-bottom:14px}.notice{min-height:21px;color:#9fb2c8;margin:0 0 14px}.notice.error{color:#ff9299}.empty{border:1px dashed #344356;border-radius:14px;color:#91a1b5;text-align:center;padding:80px 20px;margin-top:30px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:8px}.card{height:181px;background:#17212d;border:3px solid #070b0f;border-radius:12px;position:relative;overflow:hidden;cursor:pointer;isolation:isolate}.card:hover{border-color:#536981}.card.selected{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal)}.card img{width:100%;height:100%;object-fit:cover;display:block}.rank{position:absolute;top:7px;left:7px;background:#102a2d;color:#2fe8d7;border-radius:6px;padding:3px 7px;font-weight:850;z-index:1}.event{position:absolute;right:7px;top:7px;background:#172535e8;color:#aeeef0;border-radius:6px;padding:3px 7px;font-size:12px}.ocr{position:absolute;right:7px;top:7px;background:#2a2610e8;color:#ffe18a;border-radius:6px;padding:3px 7px;font-size:12px;font-weight:800}.event+.ocr{top:37px}.card-foot{position:absolute;left:0;right:0;bottom:0;padding:17px 9px 7px;display:flex;justify-content:space-between;align-items:end;background:linear-gradient(transparent,#06080bd9);font-size:15px;font-weight:800;text-shadow:0 1px 2px #000}.card-foot span:last-child{font-weight:500}.toast{position:fixed;right:26px;bottom:25px;max-width:440px;padding:13px 16px;border:1px solid #38506b;background:#1a2634;border-radius:10px;color:#e9f2fa;box-shadow:0 12px 42px #0008;display:none;z-index:5}.toast.show{display:block}@media(max-width:1050px){.shell{grid-template-columns:340px minmax(0,1fr)}.side-body{padding:20px 17px}.field{grid-template-columns:105px 1fr}.grid{grid-template-columns:repeat(auto-fill,minmax(210px,1fr))}}@media(max-width:760px){.shell{display:block}aside{position:relative;height:auto;min-height:0}.brand{height:58px}.side-body{padding:18px}main{min-height:100vh}.main-head{padding:0 12px}.actions{gap:7px}.small-btn{padding:9px 10px}.content{padding:16px 10px}.grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.card{height:135px}.card-foot{font-size:11px}.field{grid-template-columns:100px 1fr}}
</style></head><body><div class="shell"><aside><div class="brand"><span class="dot"></span><b>AIC26</b><span class="corpus" id="corpus">đang tải dữ liệu…</span></div><div class="side-body"><div class="eyebrow">BÀI TOÁN</div><div class="tabs"><button class="tab active" data-task="kis">KIS</button><button class="tab" data-task="qa">Q&amp;A</button><button class="tab" data-task="trake">TRAKE</button></div><p class="help" id="help">Nộp một khung hình bất kỳ nằm trong đoạn video chứa sự kiện.</p><textarea id="query" class="query" placeholder="cảnh báo về cá mập"></textarea><button id="search" class="search">TÌM KIẾM</button><div class="options"><div class="option-title">▼ TÙY CHỌN <span class="changed">— đã đổi: top-k, khử trùng</span></div><label class="field">Video<select id="video"><option value="">tất cả</option></select></label><label class="field">Top-K<input id="topk" type="number" min="1" max="100" value="16"></label><label class="field">Khử trùng<input id="dedupe" type="number" min="0" max="600" value="0"></label><label class="field">Trần/video<input id="pervideo" type="number" min="1" max="100" value="3"></label><div class="fence">Rào frame<div class="seg" id="fence"><button data-gap="0">Tắt</button><button data-gap="30">Hẹp</button><button class="active" data-gap="90">Chuẩn</button><button data-gap="180">Rộng</button></div></div></div></div></aside><main><header class="main-head"><div class="result-count" id="count">Sẵn sàng</div><div class="actions"><button class="small-btn" id="selectall">Chọn tất cả</button><button class="small-btn" id="clear">Bỏ chọn</button><button class="small-btn export" id="export">Xuất CSV (0)</button></div></header><div class="content"><div class="chip" id="chip"></div><p class="notice" id="notice">Nhập truy vấn rồi chọn TÌM KIẾM.</p><div class="empty" id="empty">Kết quả keyframe sẽ xuất hiện tại đây.</div><div class="grid" id="grid"></div></div></main></div><div class="toast" id="toast"></div><script>
const state={task:'kis',results:[],selected:new Set(),gap:90};const q=s=>document.querySelector(s);const qa={kis:{help:'Nộp một khung hình bất kỳ nằm trong đoạn video chứa sự kiện.',placeholder:'cảnh báo về cá mập'},qa:{help:'Nhập mô tả sự kiện và câu hỏi. Ví dụ: Cảnh trao giải. Câu hỏi: Có bao nhiêu người?',placeholder:'Cảnh trao giải. Câu hỏi: Có bao nhiêu người trên sân khấu?'},trake:{help:'Mỗi dòng là một mốc semantic, theo đúng thứ tự thời gian.',placeholder:'Vận động viên chạy đà\nVận động viên giậm nhảy\nVận động viên bay qua xà\nVận động viên tiếp đất'}};
function toast(message){const e=q('#toast');e.textContent=message;e.classList.add('show');clearTimeout(window.toastTimer);window.toastTimer=setTimeout(()=>e.classList.remove('show'),4200)}function selectedCount(){q('#export').textContent=`Xuất CSV (${state.selected.size})`}function render(){const grid=q('#grid');grid.replaceChildren();q('#empty').style.display=state.results.length?'none':'block';for(const item of state.results){const card=document.createElement('article');card.className='card'+(state.selected.has(item.id)?' selected':'');card.title=`${item.video_id} · frame ${item.frame_id}${item.ocr_text?' · OCR: '+item.ocr_text:''}${item.title?' · '+item.title:''}`;card.onclick=()=>{state.selected.has(item.id)?state.selected.delete(item.id):state.selected.add(item.id);render();selectedCount()};const image=document.createElement('img');image.src=item.image_url;image.loading='lazy';image.alt=item.title||item.video_id;image.onerror=()=>{image.style.opacity='.2'};const rank=document.createElement('span');rank.className='rank';rank.textContent=item.rank;card.append(image,rank);if(item.event){const event=document.createElement('span');event.className='event';event.textContent='Event '+item.event;card.append(event)}if(item.ocr_score){const ocr=document.createElement('span');ocr.className='ocr';ocr.textContent=`OCR ${Math.round(item.ocr_score*100)}%`;card.append(ocr)}const foot=document.createElement('div');foot.className='card-foot';const left=document.createElement('span');left.textContent=`${item.video_id} · F${item.frame_id}`;const right=document.createElement('span');right.textContent=`${item.time} · ${item.score.toFixed(3)}`;foot.append(left,right);card.append(foot);grid.append(card)}}
function options(){return{top_k:+q('#topk').value||16,dedupe:Math.max(+q('#dedupe').value||0,state.gap),max_per_video:+q('#pervideo').value||3,video_id:q('#video').value}}async function search(){const query=q('#query').value.trim();if(!query){toast('Hãy nhập truy vấn.');q('#query').focus();return}const button=q('#search');button.classList.add('loading');button.textContent='ĐANG TÌM…';q('#notice').className='notice';q('#notice').textContent='Đang truy xuất toàn bộ feature…';try{const response=await fetch('api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:state.task,query,options:options()})});const data=await response.json();if(!response.ok)throw Error(data.error||'Truy vấn thất bại');state.results=data.results;state.selected.clear();q('#count').textContent=`${data.results.length} kết quả · ${data.elapsed_ms} ms`;q('#chip').style.display='block';q('#chip').textContent=`${query} · ${query.trim().split(/\s+/).length} từ`;q('#notice').textContent=data.notice||'';render();selectedCount()}catch(error){state.results=[];state.selected.clear();render();selectedCount();q('#count').textContent='Không có kết quả';q('#notice').className='notice error';q('#notice').textContent=error.message;toast(error.message)}finally{button.classList.remove('loading');button.textContent='TÌM KIẾM'}}
document.querySelectorAll('.tab').forEach(button=>button.onclick=()=>{state.task=button.dataset.task;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===button));q('#help').textContent=qa[state.task].help;q('#query').placeholder=qa[state.task].placeholder;state.results=[];state.selected.clear();render();selectedCount();q('#count').textContent='Sẵn sàng';q('#chip').style.display='none';q('#notice').className='notice';q('#notice').textContent='Nhập truy vấn rồi chọn TÌM KIẾM.'});q('#search').onclick=search;q('#query').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')search()});q('#selectall').onclick=()=>{state.results.forEach(x=>state.selected.add(x.id));render();selectedCount()};q('#clear').onclick=()=>{state.selected.clear();render();selectedCount()};q('#fence').querySelectorAll('button').forEach(button=>button.onclick=()=>{state.gap=+button.dataset.gap;q('#fence').querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===button));q('#dedupe').value=state.gap});q('#export').onclick=async()=>{if(!state.selected.size){toast('Chọn ít nhất một kết quả trước khi xuất CSV.');return}try{const response=await fetch('api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({selected:[...state.selected]})});if(!response.ok){const data=await response.json();throw Error(data.error)}const blob=await response.blob();const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=state.task==='trake'?'aic_trake.csv':`aic_${state.task}.csv`;link.click();URL.revokeObjectURL(url)}catch(error){toast(error.message)}};
fetch('api/health').then(async response=>{const data=await response.json();if(!response.ok)throw Error(data.error);q('#corpus').textContent=`${data.vector_count.toLocaleString('vi-VN')} vector · ${data.video_count} video${data.ocr_records?` · OCR ${data.ocr_records.toLocaleString('vi-VN')}`:''}`;const select=q('#video');for(const id of data.video_ids){const option=document.createElement('option');option.value=id;option.textContent=id;select.append(option)}}).catch(error=>{q('#corpus').textContent='chưa gắn dataset';q('#notice').className='notice error';q('#notice').textContent=error.message});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="AIC26 custom retrieval dashboard")
    parser.add_argument("--host", default=os.environ.get("AIC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIC_PORT", "7860")))
    arguments = parser.parse_args()
    app.run(host=arguments.host, port=arguments.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
