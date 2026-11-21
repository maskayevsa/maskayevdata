import re
import os
import shutil
import ollama
import chromadb
from chromadb.config import Settings
import time

# ===== КОНФИГУРАЦИЯ =====
INPUT_FILES = ["sample_normative.txt"]
CHROMA_DIR = "./chroma_normative_db"
COLLECTION_NAME = "normative_docs"
EMBEDDING_MODEL = "nomic-embed-text"
EXPECTED_EMBEDDING_DIM = 768
MAX_CHUNK_LENGTH = 5000  # обрезка текста для предотвращения ошибок длины

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
def clean_text(text: str) -> str:
    """Удаляет теги и обрезает слишком длинные чанки."""
    text = re.sub(r'\[\[.*?\]\]', '', text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    if len(cleaned) > MAX_CHUNK_LENGTH:
        cleaned = cleaned[:MAX_CHUNK_LENGTH] + "\n… [обрезано]"
    return cleaned

def parse_document(file_path: str):
    """
    Парсит размеченный документ, корректно обрабатывая SECTION / END_SECTION.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Глобальные метаданные документа
    doc_match = re.search(r'\[\[DOC_START\s+name="([^"]+)"\s+id="([^"]+)"\]\]', content)
    if not doc_match:
        raise ValueError(f"В файле {file_path} не найден тег DOC_START")
    doc_name = doc_match.group(1)
    doc_id = doc_match.group(2)

    chunks = []
    current_section = None  # (num, title)

    i = 0
    length = len(content)
    while i < length:
        # ---- ЗАКРЫТИЕ СЕКЦИИ ----
        end_sec = re.match(r'\[\[END_SECTION\]\]', content[i:])
        if end_sec:
            current_section = None
            i += end_sec.end()
            continue

        # ---- ОТКРЫТИЕ СЕКЦИИ ----
        sec_start = re.match(
            r'\[\[\s*SECTION\s+num="([^"]+)"(?:\s+title="([^"]*)")?\s*\]\]',
            content[i:],
            re.IGNORECASE
        )
        if sec_start:
            sec_num = sec_start.group(1)
            sec_title = sec_start.group(2) if sec_start.group(2) else ""
            current_section = (sec_num, sec_title)
            i += sec_start.end()
            continue

        # ---- ОТКРЫТИЕ ПОДРАЗДЕЛА ----
        # Устойчивая регулярка: допускает кавычки " или ', их отсутствие, лишние пробелы
        sub_start = re.search(
            r'\[\s*\[\s*SUBSECTION\s+num\s*=\s*["\']?([^"\'\]\s]+)["\']?'
            r'(?:\s+part\s*=\s*["\']?([^"\'\]\s]*)["\']?)?'
            r'(?:\s+title\s*=\s*["\']?([^"\'\]\s]*)["\']?)?\s*\]\s*\]',
            content[i:],
            re.IGNORECASE
        )
        if sub_start:
            start_pos = i
            sub_num = sub_start.group(1)
            sub_part = sub_start.group(2) if sub_start.group(2) else ""
            sub_title = sub_start.group(3) if sub_start.group(3) else ""

            # Ищем закрывающий END_SUBSECTION
            end_match = re.search(r'\[\[END_SUBSECTION\]\]', content[start_pos:])
            if not end_match:
                print(f"⚠️ {file_path}: нет END_SUBSECTION для {sub_num}, пропуск")
                i += 1
                continue

            end_pos = start_pos + end_match.end()
            inner_content = content[start_pos + sub_start.end() : start_pos + end_match.start()]
            clean_inner = clean_text(inner_content)

            metadata = {
                "doc_name": doc_name,
                "doc_id": doc_id,
                "section_num": current_section[0] if current_section else "",
                "section_title": current_section[1] if current_section else "",
                "subsection_num": sub_num,
                "subsection_part": sub_part,
                "subsection_title": sub_title
            }

            safe_doc_id = re.sub(r'\W+', '_', doc_id)
            chunk_id = f"{safe_doc_id}_{current_section[0] if current_section else ''}_{sub_num}"
            if sub_part:
                chunk_id += f"_{sub_part}"

            chunks.append({
                "id": chunk_id,
                "text": clean_inner,
                "metadata": metadata
            })

            i = end_pos
            continue

        i += 1

    print(f"📄 {file_path}: извлечено {len(chunks)} подразделов")
    return chunks

# ===== ОСНОВНОЙ СКРИПТ =====
def main():
    # 0. Очистка старой базы
    if os.path.exists(CHROMA_DIR):
        print(f"Удаление старой векторной базы: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    # 1. Парсинг всех документов
    all_chunks = []
    for file in INPUT_FILES:
        if not os.path.exists(file):
            print(f"Файл {file} не найден, пропускаем.")
            continue
        try:
            chunks = parse_document(file)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Ошибка при обработке {file}: {e}")

    if not all_chunks:
        print("Нет чанков для индексации.")
        return

    # 2. Инициализация ChromaDB
    print("\nИнициализация ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Старая коллекция '{COLLECTION_NAME}' удалена.")
    except:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Нормативные документы (разделы/подразделы)"},
        embedding_function=None
    )

    # 3. Генерация эмбеддингов и добавление в базу
    print(f"\nГенерация эмбеддингов через '{EMBEDDING_MODEL}'...")
    success_count = 0
    failed_ids = []
    dimension_errors = []

    for idx, chunk in enumerate(all_chunks):
        text = chunk["text"]
        if not text.strip():
            print(f"⚠️ Пустой текст для {chunk['id']}, пропуск")
            failed_ids.append(chunk["id"])
            continue

        try:
            response = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=text
            )
            embedding = response["embedding"]
            actual_dim = len(embedding)

            if actual_dim != EXPECTED_EMBEDDING_DIM:
                print(f"!!! {chunk['id']}: размерность {actual_dim}, ожидалось {EXPECTED_EMBEDDING_DIM} — пропуск")
                dimension_errors.append(chunk["id"])
                failed_ids.append(chunk["id"])
                continue

            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[chunk["metadata"]],
                ids=[chunk["id"]]
            )
            success_count += 1

        except Exception as e:
            print(f"Ошибка при обработке {chunk['id']}: {e}")
            failed_ids.append(chunk["id"])

        if (idx + 1) % 20 == 0:
            print(f"Обработано: {idx + 1}/{len(all_chunks)} чанков")

        time.sleep(0.05)

    # 4. Итоговый отчёт
    print("\n" + "="*50)
    print("ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    print("="*50)
    print(f"✅ Успешно: {success_count}/{len(all_chunks)}")
    print(f"❌ Не удалось: {len(failed_ids)}")
    if dimension_errors:
        print(f"   из них ошибки размерности: {len(dimension_errors)}")

    # 5. Тестовый поиск
    if success_count > 0:
        print("\n🔍 Тестовый поиск: 'требования к оформлению приказов'")
        try:
            q_emb = ollama.embeddings(model=EMBEDDING_MODEL, prompt="требования к оформлению приказов")["embedding"]
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            if results['documents'] and results['documents'][0]:
                print("\n--- Топ-3 результата ---")
                for i, (doc, meta, dist) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    print(f"\n{i+1}. [Расстояние: {dist:.2f}]")
                    print(f"   Документ: {meta.get('doc_name')} ({meta.get('doc_id')})")
                    print(f"   Пункт: {meta.get('subsection_num')}")
                    preview = doc[:150] + "..." if len(doc) > 150 else doc
                    print(f"   Текст: {preview}")
            else:
                print("   Нет результатов.")
        except Exception as e:
            print(f"   Ошибка тестового поиска: {e}")

    print(f"\n💾 База сохранена в: {CHROMA_DIR}")

if __name__ == "__main__":
    main()