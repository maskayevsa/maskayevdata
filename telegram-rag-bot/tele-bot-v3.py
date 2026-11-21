import os
import asyncio
import logging
from datetime import datetime
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
# Импортируем оба помощника с разными алиасами
from assistant import ask_assistant as ask_employee_assistant
from assistant_normative import ask_assistant as ask_normative_assistant

# ==================== НАСТРОЙКИ ====================
# Версия бота
__version__ = "3.0.0"
__last_modified__ = "2026-02-16"
# Конфигурация
BOT_TOKEN = '1234567890:AAA-BBB'
ALLOWED_USER_IDS = [1234567890, 1234567890, ...]  # Белый список ID пользователей

# ДОБАВЛЕНО: справочник имён для приветствия
USER_NAMES = {
    1234567890: "Пользователь 1",
    1234567890: "Пользователь 2",
    ...,
    # если нужно различать двух пользователей с одинаковым ID,
    # добавьте второй с другим ключом, но ID должен быть уникальным
}

# Устанавливаем protect_content=True по умолчанию для всех сообщений
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(protect_content=True)
)
dp = Dispatcher()

# Настройка логирования
def setup_logger():
    """Настройка логгера с ротацией по дням"""
    
    # Создаем директорию для логов, если её нет
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Создаем логгер
    logger = logging.getLogger('bot_logger')
    logger.setLevel(logging.INFO)
    
    # Формат логов: дата время - уровень - сообщение
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # Создаем обработчик для файла с текущей датой
    current_date = datetime.now().strftime('%Y%m%d')
    log_filename = f'log_{current_date}.txt'
    log_filepath = os.path.join(log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8', mode='a')
    file_handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)
    
    return logger

# Инициализация логгера
logger = setup_logger()

# Хранилище выбранного режима для каждого пользователя
# 'employees' - справочник сотрудников, 'normative' - нормативные документы
user_modes = {}

def log_user_action(user_id, username, first_name, last_name, action_type, message_data, response_text=None):
    """
    Логирование действий пользователя
    message_data может быть: текстом, типом сообщения (для не текстовых), или командой
    """
    # Формируем базовую информацию о пользователе
    name_parts = []
    if first_name:
        name_parts.append(first_name)
    if last_name:
        name_parts.append(last_name)
    full_name = " ".join(name_parts) if name_parts else "Неизвестно"
    
    username_str = f"@{username}" if username else "без username"
    user_info = f"User ID: {user_id}, Username: {username_str}, Имя: {full_name}"
    
    # Проверка доступа
    is_allowed = user_id in ALLOWED_USER_IDS
    access_status = "✅ РАЗРЕШЕН" if is_allowed else "❌ ЗАБЛОКИРОВАН"
    
    # Форматируем сообщение в зависимости от типа
    if action_type == "CONNECTION":
        log_message = f"[{access_status}] {user_info} - 🔌 Подключение к боту"
        
    elif action_type == "COMMAND":
        log_message = f"[{access_status}] {user_info} - ⌨️ Команда: {message_data}"
        
    elif action_type == "TEXT_MESSAGE":
        # Текстовое сообщение - обрезаем если длинное
        short_msg = message_data[:200] + "..." if len(message_data) > 200 else message_data
        log_message = f"[{access_status}] {user_info} - 💬 Текст: {short_msg}"
        
    elif action_type == "NON_TEXT_MESSAGE":
        # Не текстовое сообщение - логируем его тип
        log_message = f"[{access_status}] {user_info} - 📎 {message_data}"
        
    elif action_type == "MODE_CHANGE":
        log_message = f"[{access_status}] {user_info} - 🔄 Смена режима: {message_data}"
        
    elif action_type == "RESPONSE":
        # Ответ бота
        if response_text:
            short_response = response_text[:200] + "..." if len(response_text) > 200 else response_text
            log_message = f"[{access_status}] {user_info} - 🤖 Ответ: {short_response}"
        else:
            log_message = f"[{access_status}] {user_info} - 🤖 Отправлен ответ"
        
    elif action_type == "ERROR":
        log_message = f"[{access_status}] {user_info} - ⚠️ Ошибка: {message_data}"
        
    elif action_type == "ACCESS_DENIED":
        log_message = f"[{access_status}] {user_info} - 🚫 Попытка доступа (заблокирован)"
        
    else:
        log_message = f"[{access_status}] {user_info} - {message_data}"
    
    # Логируем с соответствующим уровнем
    if action_type == "ERROR":
        logger.error(log_message)
    elif action_type == "ACCESS_DENIED":
        logger.warning(log_message)
    else:
        logger.info(log_message)

def is_user_allowed(user_id):
    """Проверяет, есть ли пользователь в белом списке."""
    return user_id in ALLOWED_USER_IDS

def get_mode_keyboard():
    """Создаёт клавиатуру с кнопками выбора режима."""
    # ДОБАВЛЕНО: кнопка "Старт" на всю первую строку
    btn_start = KeyboardButton(text='🔝 В начало / помощь')
    btn1 = KeyboardButton(text='👥 Справочник о сотрудниках')
    btn2 = KeyboardButton(text='📄 Помощь в делопроизводстве')
    
    markup = ReplyKeyboardMarkup(
        keyboard=[
            [btn_start],           # первая строка для кнопки "Старт"
            [btn1, btn2]            # вторая строка с двумя кнопками
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )
    return markup

# ДОБАВЛЕНО: безопасные асинхронные функции для отправки сообщений
async def safe_send_message(chat_id, text, **kwargs):
    """Безопасная отправка сообщения с обработкой ошибок."""
    try:
        return await bot.send_message(chat_id, text, **kwargs)
    except Exception as e:
        logger.error(f"Ошибка отправки сообщения: {e}")
        return None

async def safe_reply_to(message: Message, text, **kwargs):
    """Безопасный ответ на сообщение с обработкой ошибок."""
    try:
        return await message.reply(text, **kwargs)
    except Exception as e:
        logger.error(f"Ошибка ответа на сообщение: {e}")
        return await safe_send_message(message.chat.id, text, **kwargs)

async def safe_edit_message(chat_id, message_id, text, **kwargs):
    """Безопасное редактирование сообщения с обработкой ошибок."""
    try:
        await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Ошибка редактирования сообщения: {e}")
        return False

@dp.message(Command('start', 'help', 'mode'))
async def send_welcome(message: Message):
    """Приветственное сообщение и предложение выбрать режим."""
    user = message.from_user
    user_id = user.id
    command = message.text
    
    # ДОБАВЛЕНО: получаем имя из справочника или используем имя из Telegram
    greeting_name = USER_NAMES.get(user_id, user.first_name)
    
    # Логируем команду
    log_user_action(
        user_id, 
        user.username, 
        user.first_name, 
        user.last_name, 
        "COMMAND", 
        command
    )
    
    if not is_user_allowed(user_id):
        # Логируем попытку доступа заблокированного пользователя
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "ACCESS_DENIED", 
            "Попытка использования команды"
        )
        await bot.send_message(message.chat.id, "⛔ Доступ запрещён.")
        return

    # Логируем успешное подключение
    log_user_action(
        user_id, 
        user.username, 
        user.first_name, 
        user.last_name, 
        "CONNECTION", 
        command
    )

    welcome_text = f"""
👋 Здравствуйте, {greeting_name}!

Я — помощник по базе сотрудников и нормативной документации компании.

Выберите режим работы с помощью кнопок ниже:

👥 Справочник о сотрудниках — поиск сотрудников, телефонов, отделов, функциональных обязанностей.

📄 Помощь в делопроизводстве — консультации по нормативным документам, инструкциям, правилам оформления.

После выбора режима просто задайте вопрос обычным языком.
    """
    
    # ИСПРАВЛЕНО: используем safe_send_message
    response = await safe_send_message(
        message.chat.id,
        welcome_text,
        parse_mode='Markdown',
        reply_markup=get_mode_keyboard()
    )
    
    # Логируем отправку ответа
    log_user_action(
        user_id, 
        user.username, 
        user.first_name, 
        user.last_name, 
        "RESPONSE", 
        command,
        "Приветственное сообщение"
    )

@dp.message()
async def handle_all_messages(message: Message):
    """Обрабатывает все сообщения, включая не текстовые."""
    user = message.from_user
    user_id = user.id
    
    # ОПРЕДЕЛЯЕМ ТИП СООБЩЕНИЯ
    message_type = None
    message_type_desc = "Неизвестный тип"
    
    if message.text:
        message_type = "text"
        message_type_desc = f"Текстовое сообщение: {message.text[:50]}..."
    elif message.photo:
        message_type = "photo"
        message_type_desc = f"Фото (ID: {message.photo[-1].file_id})"
    elif message.video:
        message_type = "video"
        message_type_desc = f"Видео (ID: {message.video.file_id})"
    elif message.document:
        message_type = "document"
        message_type_desc = f"Документ: {message.document.file_name} (ID: {message.document.file_id})"
    elif message.audio:
        message_type = "audio"
        message_type_desc = f"Аудио: {message.audio.file_name or 'без названия'}"
    elif message.voice:
        message_type = "voice"
        message_type_desc = "Голосовое сообщение"
    elif message.video_note:
        message_type = "video_note"
        message_type_desc = "Видеосообщение (кружок)"
    elif message.sticker:
        message_type = "sticker"
        message_type_desc = f"Стикер: {message.sticker.emoji or 'без эмодзи'}"
    elif message.animation:
        message_type = "animation"
        message_type_desc = "GIF/анимация"
    elif message.location:
        message_type = "location"
        message_type_desc = f"Геолокация: {message.location.latitude}, {message.location.longitude}"
    elif message.contact:
        message_type = "contact"
        message_type_desc = f"Контакт: {message.contact.first_name} {message.contact.last_name or ''}"
    elif message.poll:
        message_type = "poll"
        message_type_desc = f"Опрос: {message.poll.question}"
    elif message.dice:
        message_type = "dice"
        message_type_desc = f"🎲 Кубик: {message.dice.emoji} = {message.dice.value}"
    elif message.game:
        message_type = "game"
        message_type_desc = f"Игра: {message.game.title}"
    
    # 1. ЛОГИРУЕМ ВСЁ, ДАЖЕ ОТ ЗАБЛОКИРОВАННЫХ ПОЛЬЗОВАТЕЛЕЙ
    if message_type == "text":
        # Текстовое сообщение
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "TEXT_MESSAGE", 
            message.text
        )
    else:
        # Не текстовое сообщение
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "NON_TEXT_MESSAGE", 
            message_type_desc
        )
    
    # 2. ПРОВЕРЯЕМ ДОСТУП
    if not is_user_allowed(user_id):
        # Дополнительно логируем попытку доступа (если хотим отдельно)
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "ACCESS_DENIED", 
            f"Попытка отправить {message_type_desc}"
        )
        
        # Для заблокированных пользователей не отвечаем на не текстовые сообщения
        if message_type == "text":
            await bot.send_message(message.chat.id, "⛔ Доступ запрещён.")
        return
    
    # 3. ДЛЯ РАЗРЕШЁННЫХ ПОЛЬЗОВАТЕЛЕЙ - ИГНОРИРУЕМ НЕ ТЕКСТОВЫЕ СООБЩЕНИЯ
    if message_type != "text":
        # Можно отправить подсказку (опционально)
        await safe_send_message(
            message.chat.id,
            "📝 Пожалуйста, отправляйте только текстовые сообщения с вопросами.",
            reply_markup=get_mode_keyboard()
        )
        return
    
    # 4. ДАЛЬШЕ ОБРАБОТКА ТЕКСТОВЫХ СООБЩЕНИЙ ОТ РАЗРЕШЁННЫХ ПОЛЬЗОВАТЕЛЕЙ
    message_text = message.text

    # --- Обработка нажатия кнопок выбора режима ---
    # ДОБАВЛЕНО: обработка кнопки "Старт"
    if message_text == '🔝 В начало / помощь':
        # Логируем нажатие кнопки "Старт"
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "COMMAND", 
            "Возврат в начало"
        )
        await send_welcome(message)
        return
    
    if message_text == '👥 Справочник о сотрудниках':
        user_modes[user_id] = 'employees'
        
        # Логируем смену режима
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "MODE_CHANGE", 
            "Справочник о сотрудниках"
        )
        
        # ИСПРАВЛЕНО: используем safe_send_message
        response = await safe_send_message(
            message.chat.id,
            "✅ Выбран режим: Справочник о сотрудниках\n\nЗадайте ваш вопрос.",
            parse_mode='Markdown',
            reply_markup=get_mode_keyboard()
        )
        
        # Логируем отправку ответа
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "RESPONSE", 
            message_text,
            "Подтверждение выбора режима"
        )
        return

    elif message_text == '📄 Помощь в делопроизводстве':
        user_modes[user_id] = 'normative'
        
        # Логируем смену режима
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "MODE_CHANGE", 
            "Помощь в делопроизводстве"
        )
        
        # ИСПРАВЛЕНО: используем safe_send_message
        response = await safe_send_message(
            message.chat.id,
            "✅ Выбран режим: Помощь в делопроизводстве\n\nЗадайте ваш вопрос.",
            parse_mode='Markdown',
            reply_markup=get_mode_keyboard()
        )
        
        # Логируем отправку ответа
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "RESPONSE", 
            message_text,
            "Подтверждение выбора режима"
        )
        return

    # --- Если режим ещё не выбран ---
    if user_id not in user_modes:
        # ИСПРАВЛЕНО: используем safe_send_message
        response = await safe_send_message(
            message.chat.id,
            "⚠️ Пожалуйста, сначала выберите режим работы, нажав соответствующую кнопку.",
            reply_markup=get_mode_keyboard()
        )
        
        # Логируем отправку ответа
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "RESPONSE", 
            message_text,
            "Предупреждение о выборе режима"
        )
        return

    # --- Отправка запроса в выбранный помощник ---
    # ИСПРАВЛЕНО: используем safe_reply_to
    thinking_msg = await safe_reply_to(message, "🤔 Ищу информацию...")
    if not thinking_msg:
        # Логируем ошибку при отправке промежуточного сообщения
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "ERROR", 
            "Не удалось отправить сообщение о начале поиска"
        )
        return

    try:
        if user_modes[user_id] == 'employees':
            answer = ask_employee_assistant(message_text)
            assistant_type = "Справочник сотрудников"
        else:  # normative
            answer = ask_normative_assistant(message_text)
            assistant_type = "Нормативные документы"

        # ИСПРАВЛЕНО: используем safe_edit_message
        success = await safe_edit_message(
            chat_id=message.chat.id,
            message_id=thinking_msg.message_id,
            text=f"📋 Ответ:\n{answer}",
            parse_mode='Markdown'
        )
        
        if success:
            # Логируем успешный ответ от помощника
            log_user_action(
                user_id, 
                user.username, 
                user.first_name, 
                user.last_name, 
                "RESPONSE", 
                message_text,
                f"[{assistant_type}] {answer[:100]}..."
            )
        else:
            # Если не удалось отредактировать, отправляем новое сообщение
            await safe_send_message(
                message.chat.id,
                f"📋 Ответ:\n{answer}",
                parse_mode='Markdown'
            )
            log_user_action(
                user_id, 
                user.username, 
                user.first_name, 
                user.last_name, 
                "RESPONSE", 
                message_text,
                f"[{assistant_type}] {answer[:100]}... (отправлено новым сообщением)"
            )
        
    except Exception as e:
        error_message = f"⚠️ Произошла ошибка: {str(e)}"
        
        # Логируем ошибку
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "ERROR", 
            f"Ошибка при получении ответа: {str(e)}"
        )
        
        # ИСПРАВЛЕНО: используем safe_edit_message
        if not await safe_edit_message(
            chat_id=message.chat.id,
            message_id=thinking_msg.message_id,
            text=error_message
        ):
            await safe_send_message(
                message.chat.id,
                error_message
            )

@dp.message(Command('status'))
async def show_status(message: Message):
    """Показывает текущий выбранный режим."""
    user = message.from_user
    user_id = user.id
    command = message.text
    
    # Логируем команду
    log_user_action(
        user_id, 
        user.username, 
        user.first_name, 
        user.last_name, 
        "COMMAND", 
        command
    )
    
    if not is_user_allowed(user_id):
        # Логируем попытку доступа заблокированного пользователя
        log_user_action(
            user_id, 
            user.username, 
            user.first_name, 
            user.last_name, 
            "ACCESS_DENIED", 
            command
        )
        await bot.send_message(message.chat.id, "⛔ Доступ запрещён.")
        return

    if user_id in user_modes:
        mode_name = "Справочник о сотрудниках" if user_modes[user_id] == 'employees' else "Помощь в делопроизводстве"
        response_text = f"🔹 Текущий режим: {mode_name}"
    else:
        response_text = "🔸 Режим ещё не выбран. Воспользуйтесь кнопками ниже."
    
    # ИСПРАВЛЕНО: используем safe_send_message
    response = await safe_send_message(
        message.chat.id,
        response_text,
        parse_mode='Markdown',
        reply_markup=get_mode_keyboard() if user_id not in user_modes else None
    )
    
    # Логируем отправку ответа
    log_user_action(
        user_id, 
        user.username, 
        user.first_name, 
        user.last_name, 
        "RESPONSE", 
        command,
        response_text
    )

# ДОБАВЛЕНО: функция запуска с обработкой таймаутов (адаптировано под asyncio)
async def run_bot():
    """Запуск бота с автоматическим восстановлением после сбоев."""
    logger.info("=" * 50)
    logger.info("ЗАПУСК БОТА")
    logger.info("=" * 50)
    
    retry_count = 0
    max_retries = 10
    base_delay = 5  # начальная задержка в секундах
    
    while True:
        try:
            logger.info(f"🟢 Попытка подключения #{retry_count + 1}")
            
            # Запускаем polling
            await dp.start_polling(bot)
            
            # Если дошли сюда - всё хорошо, сбрасываем счётчик ошибок
            retry_count = 0
            
        except Exception as e:
            # Проверяем, является ли ошибка таймаутом
            if "timed out" in str(e).lower():
                logger.warning(f"⏱ Read timeout (таймаут чтения): {e}")
            elif "connection" in str(e).lower():
                logger.warning(f"🔌 Ошибка соединения: {e}")
            else:
                logger.error(f"❌ Неожиданная ошибка: {e}")
            
            retry_count += 1
            
            # Экспоненциальная задержка перед перезапуском
            if retry_count > 0:
                delay = min(base_delay * (2 ** (retry_count - 1)), 300)  # макс 5 минут
                
                if retry_count >= max_retries:
                    logger.critical(f"🔴 Слишком много ошибок ({retry_count}). Пауза {delay} сек...")
                    await asyncio.sleep(delay)
                    retry_count = 0  # сбрасываем после большой паузы
                else:
                    logger.info(f"🔄 Перезапуск через {delay} сек... (попытка {retry_count}/{max_retries})")
                    await asyncio.sleep(delay)

async def main():
    await run_bot()

if __name__ == "__main__":
    print("🤖 Бот запущен с поддержкой двух режимов...")
    print(f"📝 Логи сохраняются в директорию 'logs/'")
    
    # Логируем запуск бота
    logger.info("="*50)
    logger.info("БОТ ЗАПУЩЕН")
    logger.info("="*50)
    
    # ИСПРАВЛЕНО: используем asyncio.run()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Бот остановлен пользователем")
        print("👋 Бот остановлен пользователем")
    except Exception as e:
        logger.critical(f"Критическая ошибка при работе бота: {str(e)}")
        print(f"❌ Критическая ошибка: {str(e)}")
    finally:
        logger.info("БОТ ОСТАНОВЛЕН")
        logger.info("="*50)