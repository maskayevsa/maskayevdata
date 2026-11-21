import sqlite3
import chromadb
from chromadb.config import Settings
import ollama
import re
from typing import List, Dict, Tuple
import time

# ===== КОНФИГУРАЦИЯ =====
DB_FILE = "employees.db"
CHROMA_COLLECTION_NAME = "employees_collection"
EMBEDDING_MODEL = "nomic-embed-text"

# ===== 1. ТОЧНЫЙ ПОИСК (SQL) =====
def exact_search(query: str, limit: int = 50) -> List[Dict]:  # Лимит уже увеличен до 50
    # ПОЛНОСТЬЮ ПРОПУСКАЕМ ТОЧНЫЙ ПОИСК ДЛЯ ЗАПРОСОВ О ЗАМЕСТИТЕЛЯХ
    query_lower = query.lower()
    if "заместитель" in query_lower and "департамент" in query_lower:
        print("[ВНИМАНИЕ] Для запроса о заместителях - ПРОПУСКАЕМ точный поиск")
        return []  # Возвращаем пустой список сразу
        
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    keywords = extract_keywords(query)
    

    # ===== СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ ДОЛЖНОСТЕЙ =====
    job_titles = ['заместитель начальника департамента', 'первый заместитель начальника департамента', 
                  'начальник отдела', 'главный специалист', 'эксперт', 'ведущий специалист']
    
    query_lower = query.lower()
    for job_title in job_titles:
        if job_title in query_lower:
            # Ищем точное совпадение по должности
            sql = """
                SELECT * FROM employees 
                WHERE LOWER("Должность") LIKE ?
                ORDER BY 
                    CASE 
                        WHEN "Должность" LIKE ? THEN 1  -- Точное совпадение
                        ELSE 2
                    END,
                    "ФИО"
                LIMIT ?
            """
            # Параметры: 1) для LIKE, 2) для точного совпадения, 3) лимит
            cursor.execute(sql, [f"%{job_title}%", f"%{job_title}%", limit])
            results = [dict(row) for row in cursor.fetchall()]
            # ДОБАВЬ ОТЛАДКУ
            print(f"[DEBUG] SQL для '{job_title}': {sql}")
            print(f"[DEBUG] Параметры: LIKE='%{job_title}%', точное='%{job_title}%', limit={limit}")
            print(f"[DEBUG] Найдено записей: {len(results)}")
            for i, res in enumerate(results[:10], 1):
                print(f"[DEBUG] {i}. ID {res.get('id')}: Должность='{res.get('Должность')}'")
            
            conn.close()
            
            # Быстрая обработка и возврат
            for res in results:
                res['source'] = 'exact'
                res['exact_relevance_score'] = 30  # Максимальный приоритет для точных должностей
                # Для первого заместителя - дополнительный бонус
                if 'первый заместитель' in res.get('Должность', '').lower():
                    res['exact_relevance_score'] = 50
            
            print(f"[Точный поиск] Найдено {len(results)} записей по должности: '{job_title}'")
            # Добавь перед return results:
            print(f"[DEBUG] Проверяем что есть в базе...")
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('SELECT id, "Должность" FROM employees WHERE "Должность" LIKE "%заместитель%"')
            all_zam = cursor.fetchall()
            print(f"[DEBUG] Всего записей с 'заместитель': {len(all_zam)}")
            for row in all_zam:
                print(f"[DEBUG] ID {row[0]}: {row[1]}")
            conn.close()
            
            
            return results  # Возвращаем сразу, не выполняя обычный поиск
    
    # ===== ОБЫЧНЫЙ ПОИСК (если не сработала специальная обработка) =====
    if not keywords:
        return []
    
    conditions = []
    params = []
    
    for kw in keywords:
        # Проверяем, является ли ключевое слово номером телефона
        if re.match(r'^[\d\-]+$', kw):
            phone_clean = kw.replace('-', '')
            conditions.append("""
                (REPLACE("Рабочий телефон", '-', '') LIKE ? OR 
                 "Рабочий телефон" LIKE ?)
            """)
            params.extend([f"%{phone_clean}%", f"%{kw}%"])

        else:
            # Для текстовых полей
            conditions.append("""
                ("ФИО" LIKE ? OR "ФИО" LIKE ? OR 
                 "Должность" LIKE ? OR "Должность" LIKE ? OR
                 "Отдел (фактический)" LIKE ? OR
                 "Отдел (по штатному расписанию)" LIKE ? OR
                 "Основные функции (общими словами)" LIKE ? OR
                 "Кратко основные функции (фактически выполняемые)" LIKE ? OR
                 "Основные функции (из Должностной инструкции)" LIKE ?)
            """)
            params.extend([f"%{kw.capitalize()}%", f"%{kw.lower()}%", 
                           f"%{kw.capitalize()}%", f"%{kw.lower()}%",
                           f"%{kw}%", f"%{kw}%",
                           f"%{kw}%", f"%{kw}%", f"%{kw}%"])  # Три новых поля функций
    
    if not conditions:
        return []
    
    # ПРОСТОЙ И РАБОЧИЙ SQL-ЗАПРОС
    sql = f"""
        SELECT * FROM employees 
        WHERE {' OR '.join(conditions)}
        LIMIT ?
    """
    params.append(limit)
    
    cursor.execute(sql, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # ДОБАВЛЯЕМ ПРИОРИТИЗАЦИЮ В PYTHON
    if keywords and results:
        primary_keyword = keywords[0]
        for res in results:
            score = 0
            fio = res.get('ФИО', '')
            dolzhnost = res.get('Должность', '')
            
            # ВЫСШИЙ ПРИОРИТЕТ: точное совпадение номера телефона
            # Проверяем, является ли ключевое слово номером (содержит цифры/дефисы)
            if re.match(r'^[\d\-]+$', primary_keyword):
                phone_clean = primary_keyword.replace('-', '')
                user_phone = res.get('Рабочий телефон', '').replace('-', '')
                if phone_clean in user_phone:
                    score += 15  # Максимальный приоритет для точного совпадения номера
            
            # Проверка телефона для ЛЮБОГО ключевого слова (не только номеров)
            # Если ключевое слово (например, "32-92") содержится в номере телефона
            user_phone_clean = res.get('Рабочий телефон', '').replace('-', '')
            if primary_keyword.replace('-', '') in user_phone_clean:
                score += 12  # Высокий приоритет для частичного совпадения номера
            
            # Высший приоритет: точное совпадение ФИО с ключевым словом (регистр)
            if primary_keyword.lower() in fio.lower():
                score += 10
                # Дополнительные баллы за точное совпадение начала фамилии
                if fio.lower().startswith(primary_keyword.lower()):
                    score += 5
            
            # Приоритет для должности
            if primary_keyword.lower() in dolzhnost.lower():
                score += 3
            
            # Приоритет для отдела
            if primary_keyword.lower() in res.get('Отдел (фактический)', '').lower():
                score += 2
            
            res['exact_relevance_score'] = score
    
    # СОРТИРУЕМ по нашему счёту релевантности
    if keywords and results:
        results.sort(key=lambda x: x.get('exact_relevance_score', 0), reverse=True)
    
    # ДОБАВЛЯЕМ ИСТОЧНИК
    for res in results:
        res['source'] = 'exact'
    
    print(f"[Точный поиск] Найдено {len(results)} записей по ключам: {keywords}")
    
    # Отладочный вывод первых 3 результатов и их счёта
    for i, res in enumerate(results[:3]):
        print(f"  Результат {i+1}: ID={res.get('id')}, ФИО={res.get('ФИО', '')[:20]}..., Очки={res.get('exact_relevance_score', 0)}")
    
    return results

def extract_keywords(query: str) -> List[str]:
    """
    Извлекает потенциальные ключевые слова из запроса.
    Пока простейшая реализация - разбиваем по пробелам и фильтруем стоп-слова.
    """
    # 1. Сначала извлекаем номера телефонов (формат XX-XX-XX или X-XX-XX)
    phone_pattern = r'\b\d{1,2}-\d{2}-\d{2}\b'
    phones = re.findall(phone_pattern, query)
    
    # Удаляем найденные номера из запроса для дальнейшей обработки
    text_without_phones = re.sub(phone_pattern, '', query)
    
    # 2. Извлекаем слова из оставшегося текста
    words = re.findall(r'\w+', text_without_phones.lower())
    
    # 3. Стоп-слова
    stop_words = {'какой', 'какая', 'какое', 'какие', 'кто', 'где', 'как', 'найти', 
                  'номер', 'телефон', 'сотрудник', 'сотрудники', 'отдел', 'департамент',
                  'занимается', 'занимаются'}
    
    # 4. Фильтруем: оставляем слова длиннее 2 ИЛИ цифровые последовательности
    keywords = []
    for word in words:
        if word in stop_words:
            continue
        # Оставляем слова длиннее 2 ИЛИ чисто цифровые последовательности
        if len(word) > 2 or word.isdigit():
            keywords.append(word)
    
    # 5. Добавляем номера телефонов
    keywords.extend(phones)
    
    return keywords

# ===== 2. СЕМАНТИЧЕСКИЙ ПОИСК (ChromaDB) =====
def semantic_search(query: str, limit: int = 10) -> List[Dict]:
    """
    Ищет семантически близкие записи в ChromaDB.
    Возвращает список словарей с результатами.
    """
    # Подключаемся к ChromaDB
    chroma_client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
    
    # Получаем эмбеддинг для запроса
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query
    )
    query_embedding = response["embedding"]
    
    # Ищем в ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["documents", "metadatas", "distances"]
    )
    
    # Преобразуем результаты в удобный формат
    semantic_results = []
    if results['documents']:
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Из документа извлекаем ID для связи с SQLite
            # В документе у нас есть строка "id: X", находим её
            id_match = re.search(r'id:\s*(\d+)', doc)
            record_id = id_match.group(1) if id_match else metadata.get('sqlite_id', '')
            
            score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
            semantic_results.append({
                'id': record_id,
                'content': doc,
                'metadata': metadata,
                'score': score,
                'source': 'semantic'
            })
    
    print(f"[Семантический поиск] Найдено {len(semantic_results)} записей")
    return semantic_results

# ===== 3. ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ (RRF) =====
def reciprocal_rank_fusion(exact_results: List[Dict], 
                          semantic_results: List[Dict],
                          k_exact: int = 20,   # Меньше = больше вес
                          k_semantic: int = 150) -> List[Dict]:  # Больше = меньше вес
    """
    Объединяет результаты с разными весами для точного и семантического поиска.
    """
    scores = {}
    
    # Точный поиск получает БОЛЬШЕ вес (k меньше)
    for rank, item in enumerate(exact_results, 1):
        item_id = str(item.get('id', ''))
        if item_id:
            scores[item_id] = scores.get(item_id, 0) + 1.0 / (k_exact + rank)
    
    # Семантический поиск получает МЕНЬШЕ вес (k больше)
    for rank, item in enumerate(semantic_results, 1):
        item_id = str(item.get('id', ''))
        if item_id:
            scores[item_id] = scores.get(item_id, 0) + 1.0 / (k_semantic + rank)
    
    # Сортируем по убыванию итогового скора
    sorted_items = []
    for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        # Находим оригинальную запись (предпочтительно из exact, так как там полные данные)
        original_item = None
        for item in exact_results:
            if str(item.get('id', '')) == item_id:
                original_item = item
                break
        
        if not original_item:
            for item in semantic_results:
                if str(item.get('id', '')) == item_id:
                    original_item = item
                    break
        
        if original_item:
            original_item['fusion_score'] = score
            sorted_items.append(original_item)
    
    return sorted_items[:10]  # Возвращаем топ-10 результатов

# ===== 4. ГЛАВНАЯ ФУНКЦИЯ ГИБРИДНОГО ПОИСКА =====
def hybrid_search(query: str) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"ГИБРИДНЫЙ ПОИСК: '{query}'")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    query_lower = query.lower()
    
    # СПЕЦИАЛЬНАЯ ОБРАБОТКА ДЛЯ ЗАПРОСОВ О ЗАМЕСТИТЕЛЯХ
    if "заместитель" in query_lower and "департамент" in query_lower:
        print("[СПЕЦИАЛЬНЫЙ РЕЖИМ] Только семантический поиск с увеличенным лимитом")
        
        # ТОЛЬКО СЕМАНТИЧЕСКИЙ ПОИСК
        semantic_results = semantic_search(query, limit=60)
        
        # БЕРЁМ БОЛЬШЕ РЕЗУЛЬТАТОВ
        results = semantic_results[:30]
        
        # СОРТИРУЕМ по семантическому скору
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # ДОБАВЛЯЕМ fusion_score
        for i, res in enumerate(results):
            res['fusion_score'] = res.get('score', 0)
            res['source'] = 'semantic_only'
        
        elapsed = time.time() - start_time
        print(f"\nПоиск выполнен за {elapsed:.2f} секунд")
        print(f"Итоговых результатов: {len(results)}")
        
        return results
    
    # ОБЫЧНАЯ ЛОГИКА ДЛЯ ДРУГИХ ЗАПРОСОВ
    exact_results = exact_search(query, limit=50)
    semantic_results = semantic_search(query, limit=40)
    
    # Объединяем результаты
    fused_results = reciprocal_rank_fusion(exact_results, semantic_results)
    
    elapsed = time.time() - start_time
    print(f"\nПоиск выполнен за {elapsed:.2f} секунд")
    print(f"Итоговых результатов: {len(fused_results)}")
    
    return fused_results

# ===== 5. ТЕСТИРОВАНИЕ =====
if __name__ == "__main__":
    # Примеры тестовых запросов
    test_queries = [
        "Фамилия1 телефон",
        "123-45-67",
        "наименование отдела",
        "Фамилия2 Имя",
        "кто занимается графиками производства работ",
        "заместитель начальника департамента"
    ]
    
    for query in test_queries:
        results = hybrid_search(query)
        
        print(f"\nТОП-3 РЕЗУЛЬТАТА ДЛЯ '{query}':")
        for i, item in enumerate(results[:3], 1):
            print(f"\n{i}. [ID:{item.get('id')}] Скоринг: {item.get('fusion_score', 0):.4f}")
            print(f"   Источник: {item.get('source', 'unknown')}")
            
            # Краткий вывод (первые 150 символов)
            if 'content' in item:
                preview = item['content'][:150] + "..." if len(item['content']) > 150 else item['content']
                print(f"   Данные: {preview}")
            elif 'ФИО' in item:
                print(f"   ФИО: {item.get('ФИО', '')}")
                print(f"   Должность: {item.get('Должность', '')}")
        
        print("\n" + "-"*60)

# ===== 6. ИНТЕГРАЦИЯ С LLM (Ollama) =====
def ask_assistant(query: str, max_candidates: int = 20) -> str:
    print(f"\n{'='*60}")
    print(f"ЗАПРОС ПОМОЩНИКУ: '{query}'")
    print(f"{'='*60}")
    
    query_lower = query.lower()
    
    # ЕСЛИ ЕСТЬ КОНКРЕТНОЕ ФИО В ЗАПРОСЕ - ИЩЕМ ЧЕЛОВЕКА
    fio_pattern = r'[А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+|[А-Я][а-я]+\s+[А-Я][а-я]+'
    fio_matches = re.findall(fio_pattern, query)
    
    # ВЫПОЛНЯЕМ ПОИСК ОДИН РАЗ
    candidates = hybrid_search(query)
    
    if fio_matches:
        print(f"[РЕЖИМ] Поиск конкретного человека: {fio_matches[0]}")
        # Обычный поиск (не фильтруем по заместителям)
        candidates = candidates[:max_candidates]
 
    # ИНАЧЕ ЕСЛИ ЗАПРОС О ВСЕХ ЗАМЕСТИТЕЛЯХ
    elif "заместитель" in query_lower and "департамент" in query_lower:
        print("[РЕЖИМ] Список всех заместителей департамента")
        
        # ФИЛЬТРУЕМ и упрощаем контекст
        filtered_candidates = []
        for cand in candidates:
            if 'ФИО' in cand:
                dolzhnost = cand.get('Должность', '').lower()
                if 'заместитель' in dolzhnost:
                    # СОХРАНЯЕМ ТОЛЬКО ФИО и должность
                    simple_cand = {
                        'ФИО': cand.get('ФИО'),
                        'Должность': cand.get('Должность'),
                        'source': cand.get('source')
                    }
                    filtered_candidates.append(simple_cand)
            elif 'content' in cand:
                content = cand['content'].lower()
                if 'заместитель' in content:
                    # Упрощаем семантические результаты
                    fio_match = re.search(r'ФИО:\s*(.+)', cand['content'])
                    dolzhnost_match = re.search(r'Должность:\s*(.+)', cand['content'])
                    if fio_match and dolzhnost_match:
                        simple_cand = {
                            'ФИО': fio_match.group(1).strip(),
                            'Должность': dolzhnost_match.group(1).strip(),
                            'source': 'semantic'
                        }
                        filtered_candidates.append(simple_cand)
        
        candidates = filtered_candidates[:max_candidates]
        print(f"Найдено {len(candidates)} заместителей")
    
    else:
        # ОБЫЧНЫЙ РЕЖИМ
        candidates = candidates[:max_candidates]
        
        # ДОПОЛНИТЕЛЬНАЯ ФИЛЬТРАЦИЯ ДЛЯ ЗАПРОСОВ О ФУНКЦИЯХ
        function_keywords = ['занимается', 'отвечает', 'функци', 'делает', 'график', 'производств', 'работ']
        if any(keyword in query_lower for keyword in function_keywords):
            print(f"[РЕЖИМ] Поиск по функциям/обязанностям")
            
            filtered_candidates = []
            for cand in candidates:
                # Проверяем наличие функций в записи
                has_functions = False
                
                if 'ФИО' in cand:
                    # Проверяем поля с функциями
                    functions_fields = [
                        cand.get('Кратко основные функции (фактически выполняемые)', ''),
                        cand.get('Основные функции (общими словами)', ''),
                        cand.get('Основные функции (из Должностной инструкции)', '')
                    ]
                    
                    for func_field in functions_fields:
                        if func_field and func_field.strip() and func_field.strip() != '-':
                            # Ищем ключевые слова из запроса в функциях
                            func_lower = func_field.lower()
                            query_words = [word for word in query_lower.split() if len(word) > 3]
                            
                            for word in query_words:
                                if word in func_lower:
                                    has_functions = True
                                    break
                            if has_functions:
                                break
                
                elif 'content' in cand:
                    # Для семантических результатов
                    content_lower = cand['content'].lower()
                    query_words = [word for word in query_lower.split() if len(word) > 3]
                    
                    for word in query_words:
                        if word in content_lower:
                            has_functions = True
                            break
                
                if has_functions:
                    filtered_candidates.append(cand)
            
            if filtered_candidates:
                candidates = filtered_candidates
                print(f"После фильтрации по функциям: {len(candidates)} записей")
                
    if not candidates:
        return "В базе данных нет информации, соответствующей вашему запросу."
    
    # 2. Формируем текстовый контекст из результатов
    print("\nФормирование контекста для LLM...")
    context_parts = []
    
    for i, cand in enumerate(candidates, 1):
                # Для результатов точного поиска (есть ключ 'ФИО')
        if 'ФИО' in cand:
            # Собираем ключевые поля в удобочитаемую строку
            fio = cand.get('ФИО', 'не указано')
            dolzhnost = cand.get('Должность', 'не указана')
            phone = cand.get('Рабочий телефон', 'не указан')
            otdel_fact = cand.get('Отдел (фактический)', 'не указан')
            otdel_shtat = cand.get('Отдел (по штатному расписанию)', 'не указан')
            
            # Используем фактический отдел, если он есть, иначе штатный
            otdel = otdel_fact if otdel_fact != 'не указан' else otdel_shtat
            
            context_line = f"{i}. {fio} - {dolzhnost}"
            if otdel and otdel != 'не указан':
                context_line += f". Отдел: {otdel}"
            if phone and phone != 'не указан':
                context_line += f". Тел: {phone}"
            
            # Добавляем краткие фактические функции, если они есть
            functions = cand.get('Кратко основные функции (фактически выполняемые)', '')
            if functions and functions.strip() and functions.strip() != '-':
                # Ограничиваем длину, чтобы не перегружать контекст
                func_preview = functions[:150] + "..." if len(functions) > 150 else functions
                context_line += f" Функции: {func_preview}"
            
            context_parts.append(context_line)
        
        # Для результатов семантического поиска (есть ключ 'content')
        elif 'content' in cand:
            # Извлекаем основные данные из семантического результата
            content = cand['content']
            
            # Пытаемся найти ФИО в контенте
            fio_match = re.search(r'ФИО:\s*(.+)', content)
            fio = fio_match.group(1).strip() if fio_match else "не указано"
            
            # Пытаемся найти должность
            dolzhnost_match = re.search(r'Должность:\s*(.+)', content)
            dolzhnost = dolzhnost_match.group(1).strip() if dolzhnost_match else "не указана"
            
            # Пытаемся найти телефон
            phone_match = re.search(r'Телефон:\s*(.+)', content)
            phone = phone_match.group(1).strip() if phone_match else "не указан"
            
            context_line = f"{i}. {fio} - {dolzhnost}"
            if phone and phone != 'не указан':
                context_line += f". Тел: {phone}"
            
            # Добавляем краткий превью контента
            content_preview = content[:100] + "..." if len(content) > 100 else content
            context_line += f" [Данные: {content_preview}]"
            
            context_parts.append(context_line)
        else:
            continue
            
    context_text = "\n".join(context_parts)
    print(f"Сформирован контекст из {len(context_parts)} записей")
    
    # В функции ask_assistant, перед формированием system_prompt:
    if "заместитель" in query_lower and "департамент" in query_lower:
        # ДОБАВЬ ЭТО ДЛЯ ОТЛАДКИ:
        print(f"Длина контекста: {len(context_text)} символов")
        print("\n--- ПЕРВЫЕ 8 ЗАПИСЕЙ В КОНТЕКСТЕ ---")
        for i, line in enumerate(context_parts[:8], 1):
            print(f"{i}. {line[:120]}...")
        print("--- КОНЕЦ ---\n")
        
        # ОСОБЫЙ ПРОМТ ДЛЯ ЗАМЕСТИТЕЛЕЙ
        system_prompt = """Ты — электронный помощник по информационной базе сотрудников компании.

ВАЖНО: Это запрос о заместителях начальника департамента. 

ТВОИ ПРАВИЛА ДЛЯ ЭТОГО ЗАПРОСА:
1. ПЕРЕЧИСЛИ ВСЕХ заместителей начальника департамента из данных ниже.
2. Укажи всех: и "Первый заместитель начальника департамента", и "Заместитель начальника департамента".
3. Сгруппируй по типам: сначала первых заместителей, потом обычных заместителей.
4. Для каждого укажи: ФИО, должность.
5. Отвечай ТОЛЬКО на основе предоставленных данных.
6. Укажи, что в ответ могли попасть не все руководители и предложи поиск по имени.

ДАННЫЕ ИЗ БАЗЫ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ОТВЕТ (перечисли всех заместителей):"""
    else:
        # Обычный промт
        system_prompt = """Ты — электронный помощник по информационной базе сотрудников компании.

ТВОИ ПРАВИЛА:
1. ОТВЕЧАЙ ТОЛЬКО на основе предоставленных ниже данных. Ничего не выдумывай.
2. Если в данных нет точного ответа на вопрос, честно говори: "В предоставленных данных нет точной информации".
3. Будь кратким, конкретным и полезным. Для сотрудников указывай: ФИО, должность, отдел, телефон.
4. Если запрос о сотруднике — найди его в данных и перечисли ключевую информацию.
5. Если запрос об отделе — перечисли сотрудников этого отдела.
6. Если запрос о функциях — укажи, кто ими занимается.
7. Говори ТОЛЬКО на русском языке.

ДАННЫЕ ИЗ БАЗЫ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ОТВЕТ (основывайся только на данных выше):"""
    
    # 4. Формируем финальный промт, подставляя контекст и запрос
    final_prompt = system_prompt.format(context=context_text, query=query)
    
    # 5. Отправляем запрос в Ollama
    print("Отправка запроса к модели saiga-rag-custom...")
    try:
        response = ollama.chat(
            model='saiga-rag-custom:latest',  # Ваша модель
            messages=[
                {'role': 'system', 'content': 'Ты полезный помощник по базе сотрудников.'},
                {'role': 'user', 'content': final_prompt}
            ],
            options={
                'temperature': 0.1,      # Низкая "творческость" для точности
                'num_ctx': 8192,         # Достаточно для контекста
                'num_predict': 1024       # Ограничение длины ответа
            }
        )
        
        answer = response['message']['content']
        print("Ответ от LLM получен.")
        return answer
        
    except Exception as e:
        error_msg = f"Ошибка при обращении к Ollama: {e}"
        print(error_msg)
        return error_msg

# ===== 7. ТЕСТИРОВАНИЕ ПОМОЩНИКА =====
if __name__ == "__main__":
    # Примеры тестовых запросов
    test_queries = [
        "Фамилия1 телефон",
        "123-45-67",
        "наименование отдела",
        "Фамилия2 Имя",
        "кто занимается графиками производства работ"
    ]
    
    for query in test_queries:
        answer = ask_assistant(query)
        print(f"\n📋 ОТВЕТ НА ЗАПРОС '{query}':")
        print(f"{'─'*50}")
        print(answer)
        print(f"{'─'*50}\n")