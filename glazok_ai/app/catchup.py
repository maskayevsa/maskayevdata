import os
import cv2
from datetime import datetime, timedelta
from database import get_connection
from cv_engine import GlazokCVEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_DIR = os.path.join(BASE_DIR, "Rec")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "snapshots")
DB_PATH = os.path.join(BASE_DIR, "data", "events.db")


def parse_video_start_dt(video_path):
    """Извлекает дату и время начала файла из пути вида ...\2026-11-13\18-15-00.mkv"""
    folder = os.path.basename(os.path.dirname(video_path))
    filename = os.path.splitext(os.path.basename(video_path))[0]
    try:
        return datetime.strptime(f"{folder} {filename}", "%Y-%m-%d %H-%M-%S")
    except ValueError:
        return None


def find_missed_videos(rec_dir=REC_DIR, db_path=DB_PATH, segment_sec=300):
    """Находит все .mkv файлы, которые не были покрыты онлайн-анализатором."""
    conn = get_connection(db_path)
    intervals = conn.execute("SELECT start_ts, end_ts FROM analyzed_intervals").fetchall()
    conn.close()

    parsed_intervals = []
    for row in intervals:
        try:
            s = datetime.strptime(row["start_ts"], "%Y-%m-%d %H:%M:%S")
            e = datetime.strptime(row["end_ts"], "%Y-%m-%d %H:%M:%S")
            parsed_intervals.append((s, e))
        except ValueError:
            pass

    missed = []
    now = datetime.now()

    for root, _, files in os.walk(rec_dir):
        for fn in sorted(files):
            if not fn.endswith(".mkv"):
                continue
            full_path = os.path.join(root, fn)
            start_dt = parse_video_start_dt(full_path)
            if not start_dt:
                continue
            end_dt = start_dt + timedelta(seconds=segment_sec)

            if end_dt > now - timedelta(seconds=15):
                continue

            covered_sec = 0
            for iv_s, iv_e in parsed_intervals:
                overlap_start = max(start_dt, iv_s)
                overlap_end = min(end_dt, iv_e)
                if overlap_end > overlap_start:
                    covered_sec += (overlap_end - overlap_start).total_seconds()

            if covered_sec < segment_sec * 0.8:
                missed.append(full_path)

    return missed


def analyze_video_file(video_path, engine: GlazokCVEngine, is_sandbox=False, db_path=DB_PATH, snapshots_dir=SNAPSHOTS_DIR):
    """Прогоняет один видеофайл через CV-движок и записывает события в БД."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    step = max(1, int(round(fps / 4.0)))
    base_dt = parse_video_start_dt(video_path) or datetime.fromtimestamp(os.path.getmtime(video_path))
    rel_video_path = os.path.relpath(video_path, REC_DIR) if video_path.startswith(REC_DIR) else video_path

    active_tracks = {}
    events_created = 0
    frame_idx = 0

    def flush_track(tr):
        nonlocal events_created
        ev_dt = base_dt + timedelta(seconds=tr["start_sec"])
        date_folder = ev_dt.strftime("%Y-%m-%d")
        sub_dir = "sandbox" if is_sandbox else date_folder
        save_dir = os.path.join(snapshots_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)

        fn = ev_dt.strftime("%Y%m%d_%H-%M-%S") + f"_f{tr['track_id']}.jpg"
        full_snap = os.path.join(save_dir, fn)
        rel_snap = os.path.join(sub_dir, fn)
        cv2.imwrite(full_snap, tr["best_frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        person_id, confidence = None, 0.0
        raw_emb = None
        if tr["best_embedding"] is not None:
            person_id, _, confidence = engine.match_face(tr["best_embedding"])
            raw_emb = tr["best_embedding"].tobytes()

        direction = engine.estimate_direction(tr["areas"])
        src_type = "sandbox" if is_sandbox else "catchup"

        conn = get_connection(db_path)
        conn.execute("""
            INSERT INTO events (timestamp, person_id, confidence, direction, snapshot_path, video_file, video_offset_sec, source_type, is_sandbox, raw_embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ev_dt.strftime("%Y-%m-%d %H:%M:%S"),
            person_id,
            round(confidence, 3),
            direction,
            rel_snap,
            rel_video_path,
            round(tr["start_sec"], 1),
            src_type,
            1 if is_sandbox else 0,
            raw_emb,
        ))
        conn.commit()
        conn.close()
        events_created += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            cur_sec = frame_idx / fps
            detections = engine.process_frame(frame, persist_tracker=True)

            for det in detections:
                tid = det["track_id"]
                if tid not in active_tracks:
                    active_tracks[tid] = {
                        "track_id": tid,
                        "start_sec": cur_sec,
                        "last_sec": cur_sec,
                        "areas": [det["area"]],
                        "best_score": det["face_score"] if det["face_found"] else (det["sharpness"] * 0.01),
                        "best_frame": frame.copy(),
                        "best_embedding": det["embedding"],
                    }
                else:
                    tr = active_tracks[tid]
                    tr["last_sec"] = cur_sec
                    tr["areas"].append(det["area"])
                    score = det["face_score"] if det["face_found"] else (det["sharpness"] * 0.01)
                    if det["face_found"] and tr["best_embedding"] is None:
                        tr["best_score"] = score
                        tr["best_frame"] = frame.copy()
                        tr["best_embedding"] = det["embedding"]
                    elif (det["face_found"] == (tr["best_embedding"] is not None)) and score > tr["best_score"]:
                        tr["best_score"] = score
                        tr["best_frame"] = frame.copy()
                        if det["embedding"] is not None:
                            tr["best_embedding"] = det["embedding"]

            finished = [tid for tid, tr in active_tracks.items() if cur_sec - tr["last_sec"] > 3.0]
            for tid in finished:
                flush_track(active_tracks.pop(tid))

        frame_idx += 1

    for tid in list(active_tracks.keys()):
        flush_track(active_tracks.pop(tid))

    cap.release()

    if not is_sandbox and base_dt:
        end_dt = base_dt + timedelta(seconds=frame_idx / fps)
        conn = get_connection(db_path)
        conn.execute(
            "INSERT INTO analyzed_intervals (start_ts, end_ts, mode) VALUES (?, ?, 'catchup')",
            (base_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()

    return events_created
