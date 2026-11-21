import os
import sys
import time
import signal
import threading
import logging
from datetime import datetime
import cv2
import yaml

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from database import init_db, get_connection
from cv_engine import GlazokCVEngine

CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("cv_worker")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str) and not os.path.isabs(val):
            cfg["paths"][key] = os.path.normpath(os.path.join(BASE_DIR, val))
    return cfg



class RTSPFrameGrabber:
    """Читает RTSP в фоновом потоке, всегда отдавая самый свежий кадр без лага буфера."""
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                logger.info(f"Подключение к потоку {self.rtsp_url}...")
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    time.sleep(3)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Потеря кадра RTSP, переподключение...")
                self.cap.release()
                self.cap = None
                time.sleep(2)
                continue

            with self.lock:
                self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class OnlineCVWorker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db_path = cfg["paths"]["db_path"]
        self.snapshots_dir = cfg["paths"]["snapshots_dir"]
        self.rtsp_url = cfg["streams"]["rtsp_hd"]
        self.running = False

        init_db(self.db_path)
        logger.info("Загрузка моделей компьютерного зрения на целевую GPU...")
        self.engine = GlazokCVEngine(db_path=self.db_path)
        self.grabber = RTSPFrameGrabber(self.rtsp_url)

        # Активные визиты: {track_id: dict}
        self.active_tracks = {}
        self.track_timeout_sec = 3.0

        # Учёт непрерывного интервала работы для реестра "догона"
        self.session_start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.interval_row_id = self._create_interval_record()

    def _create_interval_record(self):
        conn = get_connection(self.db_path)
        cur = conn.cursor()
        now_str = self.session_start_ts
        cur.execute(
            "INSERT INTO analyzed_intervals (start_ts, end_ts, mode) VALUES (?, ?, ?)",
            (now_str, now_str, "realtime")
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def _heartbeat_interval(self):
        conn = get_connection(self.db_path)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE analyzed_intervals SET end_ts = ? WHERE id = ?", (now_str, self.interval_row_id))
        conn.commit()
        conn.close()

    def _save_visit_event(self, track_data):
        now_dt = track_data["first_seen_dt"]
        date_folder = now_dt.strftime("%Y-%m-%d")
        save_dir = os.path.join(self.snapshots_dir, date_folder)
        os.makedirs(save_dir, exist_ok=True)

        filename = now_dt.strftime("%H-%M-%S") + f"_t{track_data['track_id']}.jpg"
        full_snap_path = os.path.join(save_dir, filename)
        rel_snap_path = os.path.join(date_folder, filename)

        # Сохраняем лучший кадр визита
        cv2.imwrite(full_snap_path, track_data["best_frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        person_id = None
        person_name = "Неизвестный"
        confidence = 0.0

        if track_data["best_embedding"] is not None:
            person_id, person_name, confidence = self.engine.match_face(track_data["best_embedding"])

        direction = self.engine.estimate_direction(track_data["areas"])

        # Определяем имя текущего 5-минутного файла архива для быстрой привязки
        minute_block = (now_dt.minute // 5) * 5
        video_file = os.path.join(date_folder, f"{now_dt.hour:02d}-{minute_block:02d}-00.{self.cfg['recorder']['container']}")
        offset_sec = (now_dt.minute - minute_block) * 60 + now_dt.second

        raw_emb = track_data["best_embedding"].tobytes() if track_data["best_embedding"] is not None else None

        conn = get_connection(self.db_path)
        conn.execute("""
            INSERT INTO events (timestamp, person_id, confidence, direction, snapshot_path, video_file, video_offset_sec, source_type, is_sandbox, raw_embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'realtime', 0, ?)
        """, (
            now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            person_id,
            round(confidence, 3),
            direction,
            rel_snap_path,
            video_file,
            float(offset_sec),
            raw_emb,
        ))
        conn.commit()
        conn.close()

        logger.info(f"СОБЫТИЕ: {person_name} (сходство: {confidence:.2f}) | Направление: {direction} | Кадр: {rel_snap_path}")

    def run(self):
        self.running = True
        self.grabber.start()
        logger.info("Онлайн-анализатор запущен (частота ~4-10 FPS).")

        last_heartbeat = time.time()
        last_reload = time.time()

        while self.running:
            loop_start = time.time()

            # Раз в 30 сек обновляем реестр обработанного времени и подтягиваем новые лица из БД
            if loop_start - last_heartbeat >= 30:
                self._heartbeat_interval()
                last_heartbeat = loop_start
            if loop_start - last_reload >= 15:
                self.engine.reload_embeddings()
                last_reload = loop_start

            frame = self.grabber.get_frame()
            if frame is not None:
                detections = self.engine.process_frame(frame, persist_tracker=True)
                now_ts = time.time()
                now_dt = datetime.now()

                for det in detections:
                    tid = det["track_id"]
                    if tid not in self.active_tracks:
                        self.active_tracks[tid] = {
                            "track_id": tid,
                            "first_seen_dt": now_dt,
                            "last_seen": now_ts,
                            "areas": [det["area"]],
                            "best_score": det["face_score"] if det["face_found"] else (det["sharpness"] * 0.01),
                            "best_frame": frame.copy(),
                            "best_embedding": det["embedding"],
                        }
                    else:
                        tr = self.active_tracks[tid]
                        tr["last_seen"] = now_ts
                        tr["areas"].append(det["area"])

                        # Приоритет отдаём кадру с лицом; если лица не было — берём самый резкий силуэт
                        current_score = det["face_score"] if det["face_found"] else (det["sharpness"] * 0.01)
                        if det["face_found"] and tr["best_embedding"] is None:
                            tr["best_score"] = current_score
                            tr["best_frame"] = frame.copy()
                            tr["best_embedding"] = det["embedding"]
                        elif (det["face_found"] == (tr["best_embedding"] is not None)) and current_score > tr["best_score"]:
                            tr["best_score"] = current_score
                            tr["best_frame"] = frame.copy()
                            if det["embedding"] is not None:
                                tr["best_embedding"] = det["embedding"]

            # Проверка завершившихся визитов
            now_ts = time.time()
            finished_ids = [tid for tid, tr in self.active_tracks.items() if now_ts - tr["last_seen"] > self.track_timeout_sec]
            for tid in finished_ids:
                tr = self.active_tracks.pop(tid)
                self._save_visit_event(tr)

            # Ограничение скорости цикла до ~4 кадров в секунду (0.25 сек на итерацию)
            elapsed = time.time() - loop_start
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    def stop(self):
        self.running = False
        self._heartbeat_interval()
        for tid in list(self.active_tracks.keys()):
            self._save_visit_event(self.active_tracks.pop(tid))
        self.grabber.stop()
        logger.info("Онлайн-анализатор остановлен.")


def main():
    cfg = load_config()
    worker = OnlineCVWorker(cfg)

    def shutdown(signum, frame):
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    worker.run()


if __name__ == "__main__":
    main()
