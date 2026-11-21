import os
import sys
import time
import signal
import subprocess
import threading
import logging
from logging.handlers import RotatingFileHandler
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


def setup_logger(name, log_file, console=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class ProcessRunner:
    def __init__(self, name, cmd, cwd, env, logger, restart_delay=3):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.logger = logger
        self.restart_delay = restart_delay
        self.process = None
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        while self.running:
            self.logger.info(f"Запуск процесса {self.name}...")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.process = subprocess.Popen(
                    self.cmd,
                    cwd=self.cwd,
                    env=self.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creation_flags,
                )

                for line in self.process.stdout:
                    clean_line = line.strip()
                    if clean_line:
                        self.logger.info(clean_line)

                self.process.wait()
                exit_code = self.process.returncode
                if self.running:
                    self.logger.warning(f"Процесс {self.name} завершился (код {exit_code}). Перезапуск через {self.restart_delay} сек...")
                    time.sleep(self.restart_delay)

            except Exception as e:
                if self.running:
                    self.logger.error(f"Ошибка при работе {self.name}: {e}. Повтор через {self.restart_delay} сек...")
                    time.sleep(self.restart_delay)

    def stop(self):
        self.running = False
        if self.process and self.process.poll() is None:
            self.logger.info(f"Остановка процесса {self.name}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def main():
    cfg = load_config()
    logs_dir = cfg["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)

    sys_logger = setup_logger("supervisor", os.path.join(logs_dir, "supervisor.log"), console=True)
    go2rtc_logger = setup_logger("go2rtc", os.path.join(logs_dir, "go2rtc.log"), console=False)

    # Настройка изоляции целевой второй видеокарты (GPU 1) для всех дочерних процессов
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = str(cfg["hardware"]["cuda_device_order"])
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["hardware"]["gpu_id"])

    sys_logger.info(f"Инициализация супервизора Glazok. Целевой GPU ID: {env['CUDA_VISIBLE_DEVICES']}")

    go2rtc_runner = ProcessRunner(
        name="go2rtc",
        cmd=[cfg["paths"]["go2rtc_exe"]],
        cwd=cfg["paths"]["go2rtc_dir"],
        env=env,
        logger=go2rtc_logger,
        restart_delay=3,
    )

    runners = [go2rtc_runner]

    # Точка подключения будущих воркеров (Шаг 3: recorder.py и далее CV / Web)
    recorder_script = os.path.join(cfg["paths"]["base_dir"], "app", "recorder.py")
    if cfg["modules"].get("recorder_enabled") and os.path.exists(recorder_script):
        rec_logger = setup_logger("recorder", os.path.join(logs_dir, "recorder.log"), console=False)
        runners.append(
            ProcessRunner(
                name="recorder",
                cmd=[sys.executable, recorder_script],
                cwd=cfg["paths"]["base_dir"],
                env=env,
                logger=rec_logger,
                restart_delay=5,
            )
        )

    cv_script = os.path.join(cfg["paths"]["base_dir"], "app", "cv_worker.py")
    if cfg["modules"].get("cv_realtime_enabled") and os.path.exists(cv_script):
        cv_logger = setup_logger("cv_worker", os.path.join(logs_dir, "cv_worker.log"), console=True)
        runners.append(
            ProcessRunner(
                name="cv_worker",
                cmd=[sys.executable, cv_script],
                cwd=cfg["paths"]["base_dir"],
                env=env,
                logger=cv_logger,
                restart_delay=5,
            )
        )

    web_script = os.path.join(cfg["paths"]["base_dir"], "app", "web_app.py")
    if cfg["modules"].get("web_enabled") and os.path.exists(web_script):
        web_logger = setup_logger("web_app", os.path.join(logs_dir, "web_app.log"), console=False)
        runners.append(
            ProcessRunner(
                name="web_app",
                cmd=[sys.executable, web_script],
                cwd=cfg["paths"]["base_dir"],
                env=env,
                logger=web_logger,
                restart_delay=5,
            )
        )

    for r in runners:
        r.start()

    def shutdown_handler(signum, frame):
        sys_logger.info("Получен сигнал завершения работы...")
        for r in reversed(runners):
            r.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()
