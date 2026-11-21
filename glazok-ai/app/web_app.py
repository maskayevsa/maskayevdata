import os
import sys
import threading
import cv2
import uvicorn
import yaml
from datetime import datetime
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from database import init_db, get_connection
from cv_engine import GlazokCVEngine
from catchup import find_missed_videos, analyze_video_file

CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
TEMPLATE_PATH = os.path.join(APP_DIR, "template.html")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

for key, val in CFG.get("paths", {}).items():
    if isinstance(val, str) and not os.path.isabs(val):
        CFG["paths"][key] = os.path.normpath(os.path.join(BASE_DIR, val))

DB_PATH = CFG["paths"]["db_path"]
SNAPSHOTS_DIR = CFG["paths"]["snapshots_dir"]
REC_DIR = CFG["paths"]["rec_dir"]
RTSP_HD = CFG["streams"]["rtsp_hd"]

init_db(DB_PATH)

app = FastAPI(title="Glazok AI")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")

_cv_engine = None
_cv_lock = threading.Lock()
_task_status = {"running": False, "message": "Ожидание"}


def get_cv_engine():
    global _cv_engine
    with _cv_lock:
        if _cv_engine is None:
            _cv_engine = GlazokCVEngine(db_path=DB_PATH)
        else:
            _cv_engine.reload_embeddings()
        return _cv_engine


def dir_label(code):
    return {"in": "Вход ↘", "out": "Выход ↗", "pass": "Проход ↔", "unknown": "Неизвестно"}.get(code, code)


@app.get("/", response_class=HTMLResponse)
def index(tab: str = "events"):
    conn = get_connection(DB_PATH)
    persons = conn.execute(
        "SELECT p.id, p.name, p.note, COUNT(fe.id) as vec_count "
        "FROM persons p LEFT JOIN face_embeddings fe ON p.id = fe.person_id "
        "GROUP BY p.id ORDER BY p.name"
    ).fetchall()

    is_sbx = 1 if tab == "sandbox" else 0
    events = conn.execute(
        "SELECT e.*, p.name as person_name FROM events e "
        "LEFT JOIN persons p ON e.person_id = p.id "
        "WHERE e.is_sandbox = ? ORDER BY e.timestamp DESC, e.id DESC LIMIT 100",
        (is_sbx,)
    ).fetchall()
    conn.close()

    rec_files = []
    if tab == "tools":
        for root, _, files in os.walk(REC_DIR):
            for fn in sorted(files, reverse=True):
                if fn.endswith(".mkv"):
                    rec_files.append(os.path.relpath(os.path.join(root, fn), REC_DIR))
                    if len(rec_files) >= 60:
                        break
            if len(rec_files) >= 60:
                break

    person_options = "".join([f'<option value="{p["id"]}">{p["name"]}</option>' for p in persons])
    cards = []

    for ev in events:
        title = ev["person_name"] if ev["person_name"] else "Неизвестный"
        badge_cls = "known" if ev["person_name"] else "unknown"
        snap_url = "/snapshots/" + ev["snapshot_path"].replace("\\", "/")
        conf_str = f"{int(ev['confidence']*100)}%" if ev["confidence"] > 0 else "—"
        vid_info = f"{ev['video_file']} (+{int(ev['video_offset_sec'])}с)" if ev["video_file"] else "Ручной снимок"

        card = (
            f'<div class="card">'
            f'<a href="{snap_url}" target="_blank"><img src="{snap_url}" loading="lazy"></a>'
            f'<div class="info">'
            f'<div class="row"><span class="badge {badge_cls}">{title}</span><span class="dir">{dir_label(ev["direction"])}</span></div>'
            f'<div class="meta">🕒 {ev["timestamp"]} | Сходство: <b>{conf_str}</b></div>'
            f'<div class="meta">🎞 {vid_info}</div>'
            f'<form method="post" action="/assign" class="assign-form">'
            f'<input type="hidden" name="event_id" value="{ev["id"]}">'
            f'<input type="hidden" name="tab" value="{tab}">'
            f'<select name="existing_person_id"><option value="">-- Выбрать из базы --</option>{person_options}</select>'
            f'<input type="text" name="new_name" placeholder="Или новое ФИО...">'
            f'<button type="submit">Запомнить</button></form>'
            f'<form method="post" action="/delete_event" style="margin-top:6px;" onsubmit="return confirm(\'Удалить это событие?\');">'
            f'<input type="hidden" name="event_id" value="{ev["id"]}">'
            f'<input type="hidden" name="tab" value="{tab}">'
            f'<button type="submit" class="btn-danger" style="font-size:12px; padding:4px 8px;">Удалить событие</button></form>'
            f'</div></div>'
        )
        cards.append(card)

    if tab in ("events", "sandbox"):
        clear_btn = '<form method="post" action="/clear_sandbox" style="margin-bottom:12px;"><button type="submit" class="btn-danger">Очистить результаты песочницы</button></form>' if tab == "sandbox" else ""
        content = clear_btn + ("".join(cards) if cards else "<p class='meta'>Событий пока нет.</p>")
    elif tab == "persons":

        p_rows = []
        for p in persons:
            p_rows.append(
                f'<div class="person-row" style="flex-wrap:wrap; gap:8px;">'
                f'<div><b>{p["name"]}</b> (ID: {p["id"]}) <span class="meta">(ракурсов в базе: {p["vec_count"]})</span></div>'
                f'<div style="display:flex; gap:6px;">'
                f'<form method="post" action="/rename_person" style="display:flex; gap:6px;">'
                f'<input type="hidden" name="person_id" value="{p["id"]}">'
                f'<input type="text" name="new_name" value="{p["name"]}" required>'
                f'<button type="submit">Переименовать / Объединить</button></form>'
                f'<form method="post" action="/delete_person" onsubmit="return confirm(\'Удалить профиль?\');">'
                f'<input type="hidden" name="person_id" value="{p["id"]}"><button type="submit" class="btn-danger">Удалить</button></form>'
                f'</div></div>'
            )
        content = f"<h3>Известные люди ({len(persons)})</h3>" + ("".join(p_rows) if p_rows else "<p class='meta'>База лиц пуста.</p>")
    else:
        f_opts = "".join([f'<option value="{rf}">{rf}</option>' for rf in rec_files])
        content = (
            f'<div class="tool-box"><h3>Статус фоновых задач</h3><p><b>{_task_status["message"]}</b></p><hr>'
            f'<h3>1. Догон пропущенного архива</h3><p class="meta">Проверит пропущенные 5-минутные отрезки и добавит события без дублей.</p>'
            f'<form method="post" action="/run_catchup"><button type="submit">Проверить пропущенный архив</button></form><hr>'
            f'<h3>2. Песочница (тест по файлу)</h3><p class="meta">Прогонит выбранный файл на GPU в изолированную ленту.</p>'
            f'<form method="post" action="/run_sandbox"><select name="rel_video_path" style="width:100%; margin-bottom:8px;">{f_opts}</select>'
            f'<input type="text" name="custom_path" placeholder="Или полный путь к файлу..." style="width:100%; margin-bottom:8px;">'
            f'<button type="submit">Запустить в песочнице</button></form></div>'
        )

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("{{ACT_EVENTS}}", "active" if tab == "events" else "")
    html = html.replace("{{ACT_PERSONS}}", "active" if tab == "persons" else "")
    html = html.replace("{{ACT_TOOLS}}", "active" if tab == "tools" else "")
    html = html.replace("{{ACT_SANDBOX}}", "active" if tab == "sandbox" else "")
    html = html.replace("{{PERSONS_COUNT}}", str(len(persons)))
    html = html.replace("{{CONTENT_BLOCK}}", content)
    return html


@app.post("/assign")
def assign_person(event_id: int = Form(...), existing_person_id: str = Form(""), new_name: str = Form(""), tab: str = Form("events")):
    conn = get_connection(DB_PATH)
    ev = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if not ev:
        conn.close()
        return RedirectResponse(f"/?tab={tab}", status_code=303)

    target_id = None
    if new_name.strip():
        cur = conn.cursor()
        cur.execute("INSERT INTO persons (name, created_at) VALUES (?, ?)", (new_name.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        target_id = cur.lastrowid
    elif existing_person_id.isdigit():
        target_id = int(existing_person_id)

    if target_id:
        emb_bytes = ev["raw_embedding"]
        if emb_bytes is None:
            img_path = os.path.join(SNAPSHOTS_DIR, ev["snapshot_path"])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    for d in get_cv_engine().process_frame(img, persist_tracker=False):
                        if d["embedding"] is not None:
                            emb_bytes = d["embedding"].tobytes()
                            break

        if emb_bytes is not None:
            conn.execute(
                "INSERT INTO face_embeddings (person_id, embedding, source_snapshot, created_at) VALUES (?, ?, ?, ?)",
                (target_id, emb_bytes, ev["snapshot_path"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
        conn.execute("UPDATE events SET person_id = ?, confidence = 1.0 WHERE id = ?", (target_id, event_id))
        conn.commit()

    conn.close()
    return RedirectResponse(f"/?tab={tab}", status_code=303)


@app.post("/delete_person")
def delete_person(person_id: int = Form(...)):
    conn = get_connection(DB_PATH)
    conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()
    return RedirectResponse("/?tab=persons", status_code=303)


@app.post("/delete_event")
def delete_event(event_id: int = Form(...), tab: str = Form("events")):
    conn = get_connection(DB_PATH)
    ev = conn.execute("SELECT snapshot_path FROM events WHERE id = ?", (event_id,)).fetchone()
    if ev and ev["snapshot_path"]:
        full_snap = os.path.join(SNAPSHOTS_DIR, ev["snapshot_path"])
        if os.path.exists(full_snap):
            try:
                os.remove(full_snap)
            except OSError:
                pass
    conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()
    return RedirectResponse(f"/?tab={tab}", status_code=303)


@app.post("/rename_person")
def rename_person(person_id: int = Form(...), new_name: str = Form(...)):
    clean_name = new_name.strip()
    if not clean_name:
        return RedirectResponse("/?tab=persons", status_code=303)

    conn = get_connection(DB_PATH)
    existing = conn.execute("SELECT id FROM persons WHERE name = ? AND id != ?", (clean_name, person_id)).fetchone()

    if existing:
        target_id = existing["id"]
        conn.execute("UPDATE face_embeddings SET person_id = ? WHERE person_id = ?", (target_id, person_id))
        conn.execute("UPDATE events SET person_id = ? WHERE person_id = ?", (target_id, person_id))
        conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    else:
        conn.execute("UPDATE persons SET name = ? WHERE id = ?", (clean_name, person_id))

    conn.commit()
    conn.close()
    return RedirectResponse("/?tab=persons", status_code=303)

@app.post("/manual_snapshot")
def manual_snapshot():
    now_dt = datetime.now()
    d_folder = now_dt.strftime("%Y-%m-%d")
    save_dir = os.path.join(SNAPSHOTS_DIR, d_folder)
    os.makedirs(save_dir, exist_ok=True)
    fn = now_dt.strftime("%H-%M-%S") + "_manual.jpg"
    full_path = os.path.join(save_dir, fn)

    # Захват через ffmpeg с пропуском первых 25 кадров (ожидание опорного I-frame)
    ffmpeg_exe = CFG["paths"]["ffmpeg_exe"]
    cmd = [
        ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", RTSP_HD,
        "-vf", "select=gte(n\\,25)",
        "-vframes", "1",
        "-q:v", "2",
        full_path
    ]
    import subprocess
    subprocess.run(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, timeout=10)

    if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
        conn = get_connection(DB_PATH)
        conn.execute(
            "INSERT INTO events (timestamp, person_id, confidence, direction, snapshot_path, source_type, is_sandbox) "
            "VALUES (?, NULL, 0.0, 'unknown', ?, 'manual', 0)",
            (now_dt.strftime("%Y-%m-%d %H:%M:%S"), os.path.join(d_folder, fn))
        )
        conn.commit()
        conn.close()
    return RedirectResponse("/?tab=events", status_code=303)


@app.post("/run_catchup")
def run_catchup():
    if _task_status["running"]:
        return RedirectResponse("/?tab=tools", status_code=303)

    def worker():
        _task_status["running"] = True
        try:
            missed = find_missed_videos(REC_DIR, DB_PATH, CFG["recorder"]["segment_seconds"])
            if not missed:
                _task_status["message"] = f"[{datetime.now().strftime('%H:%M:%S')}] Пропущенных отрезков не найдено."
                return
            engine = get_cv_engine()
            total = 0
            for idx, path in enumerate(missed, 1):
                _task_status["message"] = f"Догон: файл {idx}/{len(missed)} ({os.path.basename(path)})..."
                total += analyze_video_file(path, engine, is_sandbox=False, db_path=DB_PATH, snapshots_dir=SNAPSHOTS_DIR)
            _task_status["message"] = f"[{datetime.now().strftime('%H:%M:%S')}] Догон завершён! Файлов: {len(missed)}, событий: {total}."
        except Exception as e:
            _task_status["message"] = f"Ошибка догона: {e}"
        finally:
            _task_status["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return RedirectResponse("/?tab=tools", status_code=303)


@app.post("/run_sandbox")
def run_sandbox(rel_video_path: str = Form(""), custom_path: str = Form("")):
    if _task_status["running"]:
        return RedirectResponse("/?tab=tools", status_code=303)

    target = custom_path.strip() if custom_path.strip() else os.path.join(REC_DIR, rel_video_path)
    if not os.path.exists(target):
        _task_status["message"] = f"Файл не найден: {target}"
        return RedirectResponse("/?tab=tools", status_code=303)

    def worker():
        _task_status["running"] = True
        try:
            _task_status["message"] = f"Песочница: анализ {os.path.basename(target)}..."
            cnt = analyze_video_file(target, get_cv_engine(), is_sandbox=True, db_path=DB_PATH, snapshots_dir=SNAPSHOTS_DIR)
            _task_status["message"] = f"[{datetime.now().strftime('%H:%M:%S')}] Песочница завершена! Найдено событий: {cnt}."
        except Exception as e:
            _task_status["message"] = f"Ошибка песочницы: {e}"
        finally:
            _task_status["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return RedirectResponse("/?tab=tools", status_code=303)


@app.post("/clear_sandbox")
def clear_sandbox():
    conn = get_connection(DB_PATH)
    conn.execute("DELETE FROM events WHERE is_sandbox = 1")
    conn.commit()
    conn.close()
    sbx_dir = os.path.join(SNAPSHOTS_DIR, "sandbox")
    if os.path.exists(sbx_dir):
        for fn in os.listdir(sbx_dir):
            try:
                os.remove(os.path.join(sbx_dir, fn))
            except OSError:
                pass
    return RedirectResponse("/?tab=sandbox", status_code=303)


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["web"]["host"], port=CFG["web"]["port"], log_level="warning")

