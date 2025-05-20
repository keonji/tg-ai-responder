import os
import sys
import logging
import wave
import json
import subprocess
import asyncio
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, List, Any

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import httpx
from tzlocal import get_localzone

from llama_handler import LlamaOptimized, ChatLogger, SummaryProcessor, SummaryStateManager

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Класс конфигурации
# -----------------------------------------------------------------------------
class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    VOSK_MODEL_PATH = Path(os.getenv("VOSK_MODEL_PATH", "./vosk-model"))
    AUDIO_TEMP_DIR = Path(os.getenv("AUDIO_TEMP_DIR", "./temp_audio"))
    DAILY_SUMMARY_TIME = os.getenv("DAILY_SUMMARY_TIME", "23:00")
    LLAMA_SERVER_PATH = os.getenv("LLAMA_SERVER_PATH", "./llama.cpp/build/bin/llama-server")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.gguf")
    LOGS_DIR = Path(os.getenv("LOGS_DIR", "./chat_logs"))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Создаём необходимые директории
Config.AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Инициализация модели Vosk
try:
    vosk_model = Model(str(Config.VOSK_MODEL_PATH))
except Exception as e:
    logger.error(f"Ошибка инициализации модели Vosk: {e}")
    sys.exit(1)

# Инициализация объектов для llama, логирования и суммаризации
llama = LlamaOptimized()
chat_logger = ChatLogger(Config.LOGS_DIR)
summary_state = SummaryStateManager()
summary_processor = SummaryProcessor()

# Глобальная переменная для процесса llama-server
llama_server_process = None

# -----------------------------------------------------------------------------
# Вспомогательные функции
# -----------------------------------------------------------------------------
def clean_response(text: str) -> str:
    """Удаляет внутренние теги (например, <thinking>, <system>, <assistant>, <user>, <summary>) для чистого вывода."""
    text = re.sub(r'<(thinking|think)[^>]*>.*?</\1>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?(user|assistant|system|summary)[^>]*>', '', text)
    text = re.sub(r'<\|end_of_turn\|>', '', text)
    return text.strip()

def is_bot_mentioned(update: Update, bot_username: str) -> bool:
    """Проверяет, упомянут ли бот (по @username) в сообщении группового чата."""
    if update.message.entities:
        for entity in update.message.entities:
            if entity.type == "mention":
                mention = update.message.text[entity.offset: entity.offset + entity.length]
                if mention.lower() == f"@{bot_username.lower()}":
                    return True
    return False

def remove_old_logs(logs_dir: Path, days_to_keep: int = 7):
    """
    Удаляет лог-файлы из директории logs_dir, которым больше days_to_keep дней.
    Вызывается в рамках ежедневной задачи.
    """
    now = datetime.now()
    for file in logs_dir.iterdir():
        if file.is_file():
            created = datetime.fromtimestamp(file.stat().st_ctime)
            if (now - created) >= timedelta(days=days_to_keep):
                try:
                    file.unlink()
                    logger.info(f"Удалён старый лог: {file.name}")
                except Exception as e:
                    logger.error(f"Ошибка при удалении лога {file.name}: {e}")

# -----------------------------------------------------------------------------
# Обработчики команд
# -----------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды /start – выводит справку по командам."""
    text = (
        "🎙️ Голосовой ассистент готов к работе!\n"
        "🔊 Отправляйте голосовые сообщения или текст.\n"
        "📊 Используйте /summarize для получения дневной сводки.\n\n"
        "Доступные команды:\n"
        "• /summarize [enable|disable] — включение/отключение сбора сообщений для сводки\n"
        "• /summarize — получение сводки за текущий день\n"
        "• /clear — очистка контекста модели\n"
        "• /restart — перезапуск сервера LLaMA\n"
    )
    await update.message.reply_text(text, parse_mode='HTML')

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /summarize:
    - С аргументом enable/disable включает или отключает суммаризацию для чата
    - Без аргументов формируется сводка за текущий день (если суммаризация включена)
    """
    chat = update.message.chat
    args = context.args

    if args:
        subcommand = args[0].lower()
        if subcommand == "enable":
            summary_state.enable(chat.id)
            await update.message.reply_text("Суммаризация сообщений включена для этого чата.")
            return
        elif subcommand == "disable":
            summary_state.disable(chat.id)
            await update.message.reply_text("Суммаризация сообщений отключена для этого чата.")
            return

    # Проверяем, включена ли суммаризация для этого чата
    if not summary_state.is_enabled(chat.id):
        await update.message.reply_text("Суммаризация отключена для этого чата. Используйте /summarize enable для включения.")
        return

    summary = await summary_processor.summarize(chat_id=chat.id, is_manual=True)
    await update.message.reply_text(f"Сводка за сегодня:\n{summary}")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /clear — очищает контекст (память) модели для данного чата."""
    chat = update.message.chat
    ctx_key = str(chat.id) if chat.type != "private" else f"{chat.id}_{update.message.from_user.id}"
    llama.clear_context(ctx_key)
    await update.message.reply_text("Контекст модели очищен.")

async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /restart — перезапускает процесс llama-server (на случай «галлюцинаций» модели)."""
    global llama_server_process
    await update.message.reply_text("Перезапуск llama-server, подождите...")
    try:
        await stop_llama_server(llama_server_process)
        llama_server_process = await start_llama_server()
        await update.message.reply_text("llama-server перезапущен успешно.")
    except Exception as e:
        logger.error(f"Ошибка при перезапуске llama-server: {e}")
        await update.message.reply_text("Ошибка при перезапуске llama-server. Проверьте логи.")

# -----------------------------------------------------------------------------
# Обработчики сообщений
# -----------------------------------------------------------------------------
async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    При получении голосового сообщения:
    – происходит его конвертация (OGG→WAV) и распознавание через Vosk,
    – результат логируется (вне зависимости от упоминания бота),
    – пользователю возвращается расшифрованный текст.
    """
    user = update.message.from_user
    chat = update.message.chat
    ctx_key = str(chat.id) if chat.type != "private" else f"{chat.id}_{user.id}"
    try:
        file = await context.bot.get_file(update.message.voice.file_id)
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
        ogg_path = Config.AUDIO_TEMP_DIR / f"{timestamp_str}.ogg"
        wav_path = ogg_path.with_suffix(".wav")
        await file.download_to_drive(ogg_path)

        # Конвертация OGG в WAV
        cmd = ['ffmpeg', '-y', '-i', str(ogg_path), '-ar', '16000', '-ac', '1', str(wav_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Распознавание аудио
        with wave.open(str(wav_path), 'rb') as wf:
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            results = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    results.append(res.get('text', ''))
            final_res = json.loads(rec.FinalResult())
            results.append(final_res.get('text', ''))
            recognized_text = ' '.join(results).strip()

        # Удаляем временные файлы
        ogg_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

        if not recognized_text:
            raise ValueError("Пустой результат распознавания")

        # Логирование голосового сообщения (всегда)
        chat_logger.log_message(chat_id=chat.id, user_id=user.id,
                                  message_id=update.message.message_id, text=recognized_text)

        # Вывод результата пользователю
        await update.message.reply_text(
            f"<b>{user.full_name} (расшифровка):</b>\n{clean_response(recognized_text)}",
            parse_mode='HTML'
        )

    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}")
        await update.message.reply_text(
            f"<b>{user.full_name}:</b>\n[Ошибка обработки аудио]",
            parse_mode='HTML'
        )

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка текстовых сообщений:
    – сообщение логируется (без проверки упоминания бота),
    – в группах отвечает только при явном упоминании бота,
    – генерируется ответ через llama-server.
    """
    if not update.message:
        return
        
    user = update.message.from_user
    chat = update.message.chat
    text = update.message.text

    # Логирование каждого текстового сообщения
    chat_logger.log_message(chat_id=chat.id, user_id=user.id,
                              message_id=update.message.message_id, text=text)

    # В групповых чатах отвечаем только если бот упомянут
    if chat.type != "private":
        bot_username = context.bot.username
        if not is_bot_mentioned(update, bot_username):
            return

    await context.bot.send_chat_action(chat.id, "typing")
    ctx_key = str(chat.id) if chat.type != "private" else f"{chat.id}_{user.id}"
    prompt = f"<user>{text}</user>"
    response = await llama.generate(prompt, ctx_key=ctx_key, tag="user") or "Не удалось сгенерировать ответ"
    cleaned = clean_response(response)
    await update.message.reply_text(
        f"<b>{user.full_name}:</b>\n{cleaned}",
        parse_mode='HTML',
        disable_web_page_preview=True
    )

# -----------------------------------------------------------------------------
# Управление процессом llama-server
# -----------------------------------------------------------------------------
async def read_process_output(process):
    """Асинхронное чтение stdout и stderr процесса llama-server."""
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        logger.info(f"llama-server stdout: {line.decode().rstrip()}")
    while True:
        line = await process.stderr.readline()
        if not line:
            break
        logger.error(f"llama-server stderr: {line.decode().rstrip()}")

async def start_llama_server():
    """Запуск процесса llama-server с нужными параметрами."""
    cmd = [
        str(Config.LLAMA_SERVER_PATH),
        "-m", Config.MODEL_PATH,
        "--ctx-size", os.getenv("MAX_CTX_SIZE", "2048"),
        "--threads", os.getenv("LLAMA_THREADS", "8"),
        "--batch-size", os.getenv("BATCH_SIZE", "512"),
        "--temp", os.getenv("GENERATION_TEMP", "0.7"),
        "--repeat-penalty", os.getenv("REPEAT_PENALTY", "1.1"),
        "--log-disable"
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    logger.info(f"llama-server запущен: {' '.join(cmd)}")
    asyncio.create_task(read_process_output(process))
    return process

async def stop_llama_server(process):
    """Остановка процесса llama-server."""
    if process and process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
        logger.info("llama-server остановлен.")

# -----------------------------------------------------------------------------
# Ежедневная задача: сводка и очистка старых логов
# -----------------------------------------------------------------------------
async def daily_summary_task(context: ContextTypes.DEFAULT_TYPE):
    """Формирует сводку по логам за сегодняшний день и удаляет устаревшие файлы логов."""
    remove_old_logs(Config.LOGS_DIR, days_to_keep=7)
    
    # Получаем список лог-файлов (имя файла: YYYY-MM-DD_<chat_id>.log)
    processed_chats = set()
    for log_file in Config.LOGS_DIR.glob("*_*.log"):
        try:
            chat_id_str = log_file.stem.split("_")[-1]
            chat_id = int(chat_id_str)
            if chat_id in processed_chats:
                continue
            processed_chats.add(chat_id)
            
            # Проверяем, включена ли суммаризация для этого чата
            if not summary_state.is_enabled(chat_id):
                continue
            
            # Получаем сообщения за день
            messages = chat_logger.get_messages_today(chat_id)
            if not messages:
                logger.info(f"Пропускаем чат {chat_id}: нет сообщений за сегодня")
                continue
                
            summary = await summary_processor.summarize(chat_id=chat_id, is_manual=False)
            if summary and not summary.startswith("Нет сообщений для суммаризации"):
                try:
                    await context.bot.send_message(chat_id, f"Дневная сводка:\n{summary}")
                    # Обновляем время последней автоматической сводки
                    summary_processor.update_last_auto_summary_time(chat_id)
                except Exception as e:
                    logger.error(f"Не удалось отправить сводку в чат {chat_id}: {e}")
        except Exception as e:
            logger.error(f"Ошибка обработки файла {log_file.name}: {e}")

# -----------------------------------------------------------------------------
# Точка входа приложения
# -----------------------------------------------------------------------------
def main():
    global llama_server_process
    application = Application.builder().token(Config.BOT_TOKEN).build()

    # Очищаем все существующие задачи при запуске
    existing_jobs = application.job_queue.get_jobs_by_name("daily_summary_task")
    for job in existing_jobs:
        job.schedule_removal()
    logger.info(f"Удалено {len(existing_jobs)} существующих задач на отправку сводки")

    # Регистрируем команды
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("summarize", summarize_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("restart", restart_command))
    # Регистрируем обработчики сообщений
    application.add_handler(MessageHandler(filters.VOICE, voice_message_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    # Планируем ежедневную задачу на 23:00 (локальное время)
    local_tz = get_localzone()

    try:
        # Время сводки в формате HH:MM (локальное время)
        daily_time_local = datetime.strptime(Config.DAILY_SUMMARY_TIME, "%H:%M").time()
    except ValueError:
        daily_time_local = time(23, 0)

    # Формируем datetime для сегодняшней даты с локальным временем сводки
    today_local = datetime.combine(datetime.now(local_tz).date(), daily_time_local)
    # Если объект не осведомлён о часовом поясе, делаем его "aware"
    if today_local.tzinfo is None:
        today_local = today_local.replace(tzinfo=local_tz)

    # Переводим в UTC
    today_utc = today_local.astimezone(timezone.utc)
    utc_time = today_utc.time()

    # Проверяем, нет ли уже такой задачи
    existing_jobs = application.job_queue.get_jobs_by_name("daily_summary_task")
    if not existing_jobs:
        application.job_queue.run_daily(
            daily_summary_task,
            time=utc_time,
            days=(0, 1, 2, 3, 4, 5, 6),
            name="daily_summary_task"
        )
        logger.info(f"Запланирована ежедневная задача на {utc_time} UTC")
    else:
        logger.info("Ежедневная задача уже существует")

    loop = asyncio.get_event_loop()
    llama_server_process = loop.run_until_complete(start_llama_server())

    try:
        application.run_polling()
    finally:
        loop.run_until_complete(stop_llama_server(llama_server_process))

if __name__ == "__main__":
    main()
