import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from database import get_connection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "events.db")


class GlazokCVEngine:
    def __init__(self, models_dir=DEFAULT_MODELS_DIR, db_path=DEFAULT_DB_PATH, similarity_threshold=0.45):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        os.makedirs(models_dir, exist_ok=True)


        # 1. Загрузка детектора людей YOLOv8n на изолированную GPU (cuda:0 внутри процесса)
        yolo_path = os.path.join(models_dir, "yolov8n.pt")
        self.yolo = YOLO(yolo_path)
        self.yolo.to("cuda:0")

        # 2. Загрузка анализатора лиц InsightFace на CUDA
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            root=models_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 3. Загрузка эталонных векторов лиц из БД
        self.known_embeddings = []  # Список кортежей: (person_id, person_name, norm_vector)
        self.reload_embeddings()

    def reload_embeddings(self):
        """Обновляет кэш векторов из SQLite (вызывается при добавлении/исправлении людей)."""
        conn = get_connection(self.db_path)
        rows = conn.execute("""
            SELECT fe.person_id, p.name, fe.embedding
            FROM face_embeddings fe
            JOIN persons p ON p.id = fe.person_id
        """).fetchall()
        conn.close()

        loaded = []
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                loaded.append((row["person_id"], row["name"], vec / norm))
        self.known_embeddings = loaded

    def match_face(self, embedding):
        """Сравнивает вектор лица с базой по косинусному сходству."""
        if not self.known_embeddings:
            return None, "Неизвестный", 0.0

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None, "Неизвестный", 0.0
        target = embedding / norm

        best_score = -1.0
        best_id = None
        best_name = "Неизвестный"

        for person_id, name, known_vec in self.known_embeddings:
            score = float(np.dot(target, known_vec))
            if score > best_score:
                best_score = score
                best_id = person_id
                best_name = name

        if best_score >= self.similarity_threshold:
            return best_id, best_name, best_score
        return None, "Неизвестный", best_score

    @staticmethod
    def _calc_sharpness(image):
        """Оценка резкости кадра через дисперсию Лапласиана (чем выше, тем чётче)."""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def estimate_direction(area_history):
        """Определяет направление движения по изменению площади рамки человека."""
        if len(area_history) < 2:
            return "unknown"
        start_area = np.mean(area_history[:max(1, len(area_history) // 3)])
        end_area = np.mean(area_history[-max(1, len(area_history) // 3):])
        ratio = end_area / max(start_area, 1.0)
        if ratio > 1.25:
            return "in"    # Приближается к двери (Вход)
        elif ratio < 0.75:
            return "out"   # Удаляется от двери (Выход)
        return "pass"      # Прошёл мимо / стоял на месте

    def process_frame(self, frame, persist_tracker=True):
        """
        Анализирует один кадр:
        Возвращает список обнаруженных людей с координатами, track_id, качеством и вектором лица.
        """
        if persist_tracker:
            results = self.yolo.track(frame, classes=[0], conf=0.45, persist=True, verbose=False)
        else:
            results = self.yolo.predict(frame, classes=[0], conf=0.45, verbose=False)

        detections = []
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        h, w = frame.shape[:2]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            area = (x2 - x1) * (y2 - y1)
            track_id = int(boxes.id[i].item()) if (persist_tracker and boxes.id is not None) else i

            # Вырезаем область человека с небольшим запасом для поиска лица
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            person_crop = frame[cy1:cy2, cx1:cx2]

            faces = self.face_app.get(person_crop)
            best_face_emb = None
            face_score = 0.0
            sharpness = self._calc_sharpness(person_crop)

            if faces:
                # Берём самое крупное лицо в рамке человека
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                top_face = faces[0]
                best_face_emb = top_face.embedding
                face_area = (top_face.bbox[2] - top_face.bbox[0]) * (top_face.bbox[3] - top_face.bbox[1])
                # Комбинированный рейтинг качества: наличие лица + размер лица + резкость
                face_score = float(face_area) * (1.0 + sharpness / 500.0)

            detections.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "sharpness": sharpness,
                "face_found": best_face_emb is not None,
                "face_score": face_score,
                "embedding": best_face_emb,
            })

        return detections
