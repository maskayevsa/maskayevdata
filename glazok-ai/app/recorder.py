import os
import sys
import time
import shutil
import signal
import subprocess
import threading
import logging
from datetime import datetime, timedelta
import yaml

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str) and not os.path.isabs(val):
            cfg["paths"][key] = os.path.normpath(os.path.join(BASE_DIR, val))
    return cfg


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("recorder")


class ArchiveCleaner:
    def __init__(self, rec_dir, max_size_gb, max_days, interval_sec):
        self.rec_dir = rec_dir
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.max_days = max_days
        self.interval_sec = interval_sec
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _get_files_sorted(self):
        files = []
        for root, _, filenames in os.walk(self.rec_dir):
            for fn in filenames:
                if fn.endswith((".mkv", ".mp4")):
                    full_path = os.path.join(root, fn)
                    try:
                        stat = os.stat(full_path)
                        files.append((stat.st_mtime, stat.st_size, full_path))
                    except OSError:
                        pass
        files.sort(key=lambda x: x[0])  # От самых старых к новым
        return files

    def _remove_empty_dirs(self):
        for entry in os.scandir(self.rec_dir):
            if entry.is_dir():
                try:
                    if not os.listdir(entry.path):
                        os.rmdir(entry.path)
                        logger.info(f"Удалена пустая папка архива: {entry.name}")
                except OSError:
                    pass

    def cleanup_once(self):
        if not os.path.exists(self.rec_dir):
            return

        cutoff_ts = (datetime.now() - timedelta(days=self.max_days)).timestamp()
        files = self._get_files_sorted()

        # 1. Удаление по сроку давности (старше max_days)
        remaining_files = []
        for mtime, size, path in files:
            if mtime < cutoff_ts:
                try:
                    os.remove(path)
                    logger.info(f"Удалён устаревший файл (> {self.max_days} дн.): {os.path.basename(path)}")
                except OSError as e:
                    logger.error(f"Ошибка удаления {path}: {e}")
            else:
                remaining_files.append((mtime, size, path))

        # 2. Удаление по превышению общего лимита объёма (max_size_gb)
        total_size = sum(f[1] for f in remaining_files)
        while total_size > self.max_size_bytes and remaining_files:
            mtime, size, oldest_path = remaining_files.pop(0)
            try:
                os.remove(oldest_path)
                total_size -= size
                logger.info(f"Удалён старый файл по лимиту {self.max_size_bytes // (1024**3)} ГБ: {os.path.basename(oldest_path)}")
            except OSError as e:
                logger.error(f"Ошибка удаления {oldest_path}: {e}")

        self._remove_empty_dirs()

    def _loop(self):
        while self.running:
            try:
                self.cleanup_once()
            except Exception as e:
                logger.error(f"Сбой при очистке архива: {e}")
            for _ in range(self.interval_sec):
                if not self.running:
                    break
                time.sleep(1)


class StreamRecorder:
    def __init__(self, cfg):
        self.ffmpeg_exe = cfg["paths"]["ffmpeg_exe"]
        self.rtsp_url = cfg["streams"]["rtsp_hd"]
        self.rec_dir = cfg["paths"]["rec_dir"]
        self.segment_sec = cfg["recorder"]["segment_seconds"]
        self.container = cfg["recorder"]["container"]
        self.process = None
        self.running = False

    def _ensure_date_dirs(self):
        # Создаём папки на сегодня и завтра, чтобы ffmpeg не упал в 00:00:00 при переходе суток
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        os.makedirs(os.path.join(self.rec_dir, today), exist_ok=True)
        os.makedirs(os.path.join(self.rec_dir, tomorrow), exist_ok=True)

    def _dir_watcher(self):
        while self.running:
            self._ensure_date_dirs()
            time.sleep(60)

    def run(self):
        self.running = True
        self._ensure_date_dirs()
        threading.Thread(target=self._dir_watcher, daemon=True).start()

        output_pattern = os.path.join(self.rec_dir, "%Y-%m-%d", f"%H-%M-%S.{self.container}")

        cmd = [
            self.ffmpeg_exe,
            "-hide_banner",
            "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "copy",
            "-an",
            "-f", "segment",
            "-segment_time", str(self.segment_sec),
            "-segment_atclocktime", "1",
            "-reset_timestamps", "1",
            "-strftime", "1",
            output_pattern,
        ]

        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        while self.running:
            logger.info(f"Старт записи потока {self.rtsp_url} в {self.rec_dir}...")
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creation_flags,
                )
                for line in self.process.stdout:
                    clean = line.strip()
                    if clean:
                        logger.warning(f"ffmpeg: {clean}")

                self.process.wait()
                if self.running:
                    logger.warning("Процесс записи ffmpeg остановлен. Перезапуск через 3 сек...")
                    time.sleep(3)
            except Exception as e:
                if self.running:
                    logger.error(f"Ошибка рекордера: {e}. Повтор через 3 сек...")
                    time.sleep(3)

    def stop(self):
        self.running = False
        if self.process and self.process.poll() is None:
            logger.info("Остановка записи ffmpeg...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def main():
    cfg = load_config()
    cleaner = ArchiveCleaner(
        rec_dir=cfg["paths"]["rec_dir"],
        max_size_gb=cfg["recorder"]["max_size_gb"],
        max_days=cfg["recorder"]["max_days"],
        interval_sec=cfg["recorder"]["cleanup_interval_sec"],
    )
    recorder = StreamRecorder(cfg)

    def shutdown(signum, frame):
        recorder.stop()
        cleaner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    cleaner.start()
    recorder.run()


if __name__ == "__main__":
    main()
