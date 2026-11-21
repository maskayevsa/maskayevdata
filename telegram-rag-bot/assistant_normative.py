import re
import chromadb
from chromadb.config import Settings
import ollama
from typing import List, Dict, Tuple, Optional

# ===== КОНФИГУРАЦИЯ =====
CHROMA_DIR = "./chroma_normative_db"
COLLECTION_NAME = "normative_docs"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "saiga-rag-custom:latest"  # или другая локальная модель

# Настройки поиска
TOP_K_SEMANTIC = 30          # берём больше кандидатов из векторного поиска
TOP_K_FINAL = 10             # сколько отдаём в контекст LLM
MAX_CONTEXT_LENGTH = 6000    # максимальная длина текста контекста (в символах)

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

def extract_citation_request(query: str) -> Optional[Dict]:
    """
    Анализирует запрос на предмет явной просьбы процитировать пункт.
    Возвращает словарь с метаданными для точного поиска или None.
    """
    query_lower = query.lower()
    # Ищем паттерны: "пункт 19.23", "п. 5.1", "цитата из пункта 2.3"
    patterns = [
        r'(?:пункт|п\.)\s*([\d\.]+)',           # пункт 19.23, п.5.1
        r'процитируй\s+(?:пункт|п\.)\s*([\d\.]+)', # процитируй пункт 19.23
        r'цитату?\s+(?:пункта|п\.)\s*([\d\.]+)'    # цитату пункта 5.1
    ]
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            subsection_num = match.group(1)
            return {"subsection_num": subsection_num}
    return None


def extract_document_filter(query: str) -> Dict:
    """
    Извлекает из запроса возможные фильтры по документу, разделу и т.п.
    Возвращает словарь для ChromaDB where.
    """
    filters = {}
    query_lower = query.lower()

    # Поиск номера документа (например, "ЦА-24", "№99")
    doc_id_match = re.search(r'(?:№?\s*([А-Я]{1,3}-\d{2,3})|([№]\s*\d+))', query)
    if doc_id_match:
        doc_id = doc_id_match.group(1) or doc_id_match.group(2)
        if doc_id:
            filters["doc_id"] = doc_id.strip()

    # Поиск номера пункта (если ещё не извлекли как цитату)
    subsection_match = re.search(r'(?:пункт|п\.)\s*([\d\.]+)', query_lower)
    if subsection_match:
        filters["subsection_num"] = subsection_match.group(1)

    # Можно добавить поиск по названию документа, но пока ограничимся ID
    return filters


# ===== 1. ТОЧНЫЙ ПОИСК ПО МЕТАДАННЫМ =====
def exact_search_by_metadata(metadata_filter: Dict) -> List[Dict]:
    """
    Выполняет точный поиск по метаданным в ChromaDB.
    Возвращает список чанков, соответствующих фильтру.
    """
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)

    try:
        results = collection.get(
            where=metadata_filter,
            include=["documents", "metadatas"]
        )
    except Exception as e:
        print(f"Ошибка точного поиска: {e}")
        return []

    chunks = []
    if results['ids']:
        for i in range(len(results['ids'])):
            chunks.append({
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i],
                'source': 'exact',
                'score': 1.0  # максимальный балл для точных совпадений
            })
    return chunks


# ===== 2. СЕМАНТИЧЕСКИЙ ПОИСК =====
def semantic_search(query: str, top_k: int = TOP_K_SEMANTIC) -> List[Dict]:
    """
    Выполняет семантический поиск по векторной базе.
    Возвращает top_k ближайших чанков (без отбрасывания по порогу).
    """
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)

    # Получаем эмбеддинг запроса
    try:
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=query
        )
        query_embedding = response["embedding"]
    except Exception as e:
        print(f"Ошибка получения эмбеддинга: {e}")
        return []

    # Поиск
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if results['documents'] and results['documents'][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Сохраняем расстояние как score (меньше = лучше)
            chunks.append({
                'id': results['ids'][0][i],
                'text': doc,
                'metadata': meta,
                'source': 'semantic',
                'score': dist  # L2 расстояние, чем меньше, тем релевантнее
            })
    return chunks


# ===== 3. ГИБРИДНЫЙ ПОИСК =====
def hybrid_search(query: str) -> List[Dict]:
    """
    Основная функция поиска:
    1. Пытается найти точные совпадения по метаданным.
    2. Выполняет семантический поиск.
    3. Объединяет результаты, отдавая приоритет точным совпадениям.
    """
    results = []

    # 1. Извлекаем фильтры из запроса
    filters = extract_document_filter(query)
    if filters:
        exact_results = exact_search_by_metadata(filters)
        results.extend(exact_results)

    # 2. Семантический поиск
    semantic_results = semantic_search(query)
    
    # Добавляем семантические результаты, избегая дубликатов по id
    existing_ids = {r['id'] for r in results}
    for sr in semantic_results:
        if sr['id'] not in existing_ids:
            results.append(sr)

    # 3. Сортировка: точные выше, затем по расстоянию (меньше = лучше)
    results.sort(key=lambda x: (x['source'] != 'exact', x.get('score', float('inf'))))
    
    # 4. Ограничиваем финальное количество
    return results[:TOP_K_FINAL]


# ===== 4. ФОРМИРОВАНИЕ КОНТЕКСТА ДЛЯ LLM =====
def format_context(chunks: List[Dict]) -> str:
    """
    Формирует текст контекста из найденных чанков, 
    добавляет информацию об источнике и обрезает при необходимости.
    """
    context_parts = []
    total_length = 0

    for i, chunk in enumerate(chunks, 1):
        meta = chunk['metadata']
        
        # Формируем строку источника
        source_parts = []
        if meta.get('doc_name'):
            # Укорачиваем название, если оно слишком длинное
            doc_name = meta['doc_name']
            if len(doc_name) > 50:
                doc_name = doc_name[:47] + "..."
            source_parts.append(doc_name)
        if meta.get('doc_id'):
            source_parts.append(f"(№{meta['doc_id']})")
        if meta.get('section_num'):
            source_parts.append(f"раздел {meta['section_num']}")
        if meta.get('subsection_num'):
            source_parts.append(f"пункт {meta['subsection_num']}")
        if meta.get('subsection_part'):
            source_parts.append(f"часть {meta['subsection_part']}")
        
        source_str = ", ".join(source_parts) if source_parts else "Неизвестный источник"
        
        # Текст чанка
        chunk_text = chunk['text'].strip()
        
        # Формируем блок контекста
        block = f"[{i}] Текст:\n{chunk_text}\nИсточник: {source_str}\n"
        
        # Если превышаем лимит, обрезаем этот блок или пропускаем?
        if total_length + len(block) > MAX_CONTEXT_LENGTH:
            # Если это первый блок — обрежем его, иначе остановимся
            if i == 1:
                block = block[:MAX_CONTEXT_LENGTH] + "...\n[контекст обрезан]\n"
                context_parts.append(block)
            break
        else:
            context_parts.append(block)
            total_length += len(block)

    return "\n".join(context_parts)


# ===== 5. ЗАПРОС К LLM =====
def ask_llm(query: str, context: str) -> str:
    """
    Отправляет промт с контекстом в локальную LLM и возвращает ответ.
    """
    system_prompt = """Ты — интеллектуальный помощник по нормативной документации компании. 

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного ниже контекста. Не используй свои знания.
2. Каждое утверждение подкрепляй ссылкой на источник в формате: [Документ, пункт].
3. Если пользователь явно запросил цитату, приведи точный текст пункта в кавычках.
4. Если в контексте нет информации для ответа, скажи: "В предоставленных документах не удалось найти информацию по вашему запросу."
5. Отвечай на русском языке, кратко и по существу.

КОНТЕКСТ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ОТВЕТ:"""

    final_prompt = system_prompt.format(context=context, query=query)

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': 'Ты полезный помощник по нормативным документам.'},
                {'role': 'user', 'content': final_prompt}
            ],
            options={
                'temperature': 0.1,
                'num_ctx': 8192,
                'num_predict': 1024
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"Ошибка при обращении к модели: {e}"


# ===== 6. ОСНОВНАЯ ФУНКЦИЯ ПОМОЩНИКА =====
def ask_assistant(query: str) -> str:
    """
    Главная функция: обрабатывает запрос пользователя.
    """
    print(f"\n{'='*60}")
    print(f"ЗАПРОС: {query}")
    print(f"{'='*60}")

    # Шаг 1: Проверяем, не является ли запрос явным требованием цитаты
    citation_request = extract_citation_request(query)
    if citation_request:
        print("🔍 Обнаружен запрос на цитирование пункта.")
        exact_chunks = exact_search_by_metadata(citation_request)
        if exact_chunks:
            chunk = exact_chunks[0]
            meta = chunk['metadata']
            source = f"{meta.get('doc_name', 'Документ')} (№{meta.get('doc_id', '')}), пункт {meta.get('subsection_num', '')}"
            if meta.get('subsection_part'):
                source += f", часть {meta.get('subsection_part')}"
            answer = f"Цитата из {source}:\n\n> {chunk['text'].strip()}"
            return answer
        else:
            return f"Пункт {citation_request.get('subsection_num')} не найден в базе документов."

    # Шаг 2: Гибридный поиск
    chunks = hybrid_search(query)
    if not chunks:
        return "По вашему запросу ничего не найдено в базе нормативных документов."

    # Шаг 3: Формируем контекст
    context = format_context(chunks)

    # Шаг 4: Запрос к LLM
    answer = ask_llm(query, context)
    return answer


# ===== 7. ТЕСТИРОВАНИЕ =====
if __name__ == "__main__":
    # Примеры запросов
    test_queries = [
        "как правильно указывать даты в тексте телеграммы?",
        "процитируй пункт 1.15",
        "п. 5.1 требования к шрифтам",
        "ЦА-24 оформление приказов",
        "дата",
    ]
    for q in test_queries:
        response = ask_assistant(q)
        print(f"\n📌 ОТВЕТ:\n{response}")
        print("\n" + "-"*50)