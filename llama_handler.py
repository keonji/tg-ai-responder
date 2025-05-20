import os
import re
import asyncio
import logging
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import httpx

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Класс LlamaOptimized: Обёртка для llama-server
# -----------------------------------------------------------------------------
class LlamaOptimized:
    def __init__(self):
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "Вы универсальный Чат-бот.")
        if self.system_prompt and not self.system_prompt.startswith("<system>"):
            self.system_prompt = f"<system>{self.system_prompt}</system>"
        self.contexts = defaultdict(list)
        self.lock = asyncio.Lock()
        self.threads = int(os.getenv("LLAMA_THREADS", 8))
        self.max_ctx_size = int(os.getenv("MAX_CTX_SIZE", 2048))
        self.temperature = float(os.getenv("GENERATION_TEMP", 0.7))
        self.repeat_penalty = float(os.getenv("REPEAT_PENALTY", 1.1))
        self.batch_size = int(os.getenv("BATCH_SIZE", 512))
        self.timeout = int(os.getenv("GENERATION_TIMEOUT", 300))
        self.max_response_tokens = int(os.getenv("MAX_RESPONSE_TOKENS", 768))
        self.server_host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self.server_port = os.getenv("LLAMA_SERVER_PORT", "8080")
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _parse_context(self, ctx_key, use_system_prompt=True):
        messages = []
        for entry in self.contexts[ctx_key]:
            entry = entry.strip()
            if entry.startswith("<system>"):
                role = "system"
                content = re.sub(r"</?system>", "", entry).strip()
            elif entry.startswith("<user>"):
                role = "user"
                content = re.sub(r"</?user>", "", entry).strip()
            elif entry.startswith("<assistant>"):
                role = "assistant"
                content = re.sub(r"</?assistant>", "", entry).strip()
            elif entry.startswith("<thinking>"):
                continue
            else:
                role = "user"
                content = entry
            messages.append({"role": role, "content": content})
        
        # Добавляем системный промпт только если это разрешено
        if use_system_prompt and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": re.sub(r"</?system>", "", self.system_prompt)})
        return messages

    def _update_context(self, ctx_key: str, text: str) -> None:
        """Обновляет контекст с учетом ограничения размера."""
        self.contexts[ctx_key].append(text)
        total_size = len('\n'.join(self.contexts[ctx_key]))
        while total_size > self.max_ctx_size * 0.75:
            removed = self.contexts[ctx_key].pop(0)
            total_size -= len(removed) + 1  # +1 for newline

    async def generate(self, prompt: str, ctx_key: str, max_tokens: Optional[int] = None, 
                      tag: str = "thinking", add_system_prompt: bool = True) -> Optional[str]:
        if max_tokens is None:
            max_tokens = self.max_response_tokens
        async with self.lock:
            try:
                if add_system_prompt and not self.contexts[ctx_key] and self.system_prompt:
                    self._update_context(ctx_key, self.system_prompt)
                if prompt.startswith("<user>") or prompt.startswith("<assistant>") or prompt.startswith("<system>"):
                    wrapped = prompt
                else:
                    wrapped = f"<{tag}>{prompt}</{tag}>"
                self._update_context(ctx_key, wrapped)
                conversation = self._parse_context(ctx_key, use_system_prompt=add_system_prompt)
                payload = {
                    "messages": conversation,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "top_p": 1.0
                }
                url = f"http://{self.server_host}:{self.server_port}/v1/chat/completions"
                client = await self._get_http_client()
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    logging.error(f"Llama server error: {resp.text}")
                    return None
                data = resp.json()
                completion = data["choices"][0]["message"]["content"]
                self._update_context(ctx_key, f"<assistant>{completion}</assistant>")
                return completion.strip()
            except Exception as e:
                logging.error(f"Ошибка генерации: {e}")
                return None

    def clear_context(self, ctx_key: str) -> None:
        """Очищает контекст для указанного ключа."""
        if ctx_key in self.contexts:
            del self.contexts[ctx_key]

# -----------------------------------------------------------------------------
# Класс ChatLogger: Логирование сообщений в файл за день с фильтрацией
# -----------------------------------------------------------------------------
class ChatLogger:
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_ttl = 300  # 5 минут
        self._last_cache_cleanup = datetime.now()

    def _cleanup_cache(self) -> None:
        """Очищает устаревшие записи из кэша."""
        now = datetime.now()
        if (now - self._last_cache_cleanup).total_seconds() > self._cache_ttl:
            self._cache.clear()
            self._last_cache_cleanup = now

    def _get_daily_file(self, chat_id: int) -> Path:
        """Возвращает путь к файлу лога за текущий день."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.logs_dir / f"{today}_{chat_id}.log"

    def _sanitize_text(self, text: str) -> str:
        """
        Очищает текст сообщения от лишних элементов.
        
        Args:
            text: Исходный текст сообщения
            
        Returns:
            Очищенный текст сообщения
        """
        if not text:
            return ""
            
        # Удаляем ссылки
        text = re.sub(r'https?://\S+', '', text)
        
        # Удаляем HTML/технические теги
        text = re.sub(r'<.*?>', '', text)
        
        # Удаляем упоминания медиафайлов
        # text = re.sub(r'\b(video|audio|voice|photo|document|sticker|animation)\b', '', text, flags=re.IGNORECASE)
        
        # Удаляем текст репоста
        # text = re.sub(r'Forwarded from .+', '', text)
        
        # Удаляем эмодзи (опционально)
        # text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
        
        # Удаляем множественные пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def log_message(self, chat_id: int, user_id: int, message_id: int, text: str) -> None:
        """
        Логирует сообщение в файл.
        
        Args:
            chat_id: ID чата
            user_id: ID пользователя
            message_id: ID сообщения
            text: Текст сообщения
        """
        sanitized = self._sanitize_text(text)
        if not sanitized:
            return

        message_data = {
            "timestamp": datetime.now().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "message_id": message_id,
            "text": sanitized
        }
        
        log_file = self._get_daily_file(chat_id)
        try:
            with log_file.open("a", encoding="utf-8") as f:
                json.dump(message_data, f, ensure_ascii=False)
                f.write("\n")
            # Инвалидируем кэш для этого чата
            cache_key = f"{chat_id}_{datetime.now().strftime('%Y-%m-%d')}"
            if cache_key in self._cache:
                del self._cache[cache_key]
        except Exception as e:
            logging.error(f"Ошибка логирования сообщения: {e}")

    def get_messages_today(self, chat_id: int) -> List[Dict[str, Any]]:
        """Получает сообщения за текущий день с использованием кэша."""
        self._cleanup_cache()
        cache_key = f"{chat_id}_{datetime.now().strftime('%Y-%m-%d')}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]

        log_file = self._get_daily_file(chat_id)
        if not log_file.exists():
            return []

        messages = []
        try:
            with log_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    messages.append(data)
            # Сохраняем в кэш
            self._cache[cache_key] = messages
        except Exception as e:
            logging.error(f"Ошибка чтения лога для чата {chat_id}: {e}")
        return messages

# -----------------------------------------------------------------------------
# Класс SummaryStateManager: Управление состоянием суммаризации для чатов
# -----------------------------------------------------------------------------
class SummaryStateManager:
    def __init__(self, state_file: Path = Path("summary_state.json")):
        self.state_file = state_file
        self.states = self._load_states()

    def _load_states(self) -> Dict[int, bool]:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return {int(k): v for k, v in json.load(f).items()}
            except Exception as e:
                logging.error(f"Ошибка загрузки состояний суммаризации: {e}")
                return {}
        return {}

    def _save_states(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.states, f)
        except Exception as e:
            logging.error(f"Ошибка сохранения состояний суммаризации: {e}")

    def is_enabled(self, chat_id: int) -> bool:
        return self.states.get(chat_id, False)  # По умолчанию отключено

    def enable(self, chat_id: int):
        self.states[chat_id] = True
        self._save_states()

    def disable(self, chat_id: int):
        self.states[chat_id] = False
        self._save_states()

# -----------------------------------------------------------------------------
# Класс LlamaSummary: Отдельное подключение к llama-server для суммаризации
# -----------------------------------------------------------------------------
class LlamaSummary:
    def __init__(self):
        self.server_host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self.server_port = os.getenv("LLAMA_SERVER_PORT", "8080")
        self.timeout = int(os.getenv("GENERATION_TIMEOUT", 300))
        self.max_response_tokens = int(os.getenv("MAX_RESPONSE_TOKENS", 768))
        self.temperature = float(os.getenv("SUMMARY_TEMP", "0.3"))  # Более низкая температура для более точных сводок
        self.lock = asyncio.Lock()

    async def generate_summary(self, prompt: str) -> str:
        """Генерирует сводку для одного топика."""
        async with self.lock:
            # Проверяем, что промпт не обрезан
            if len(prompt) > 8000:  # Увеличенный лимит для llama-server
                logging.warning("Промпт слишком длинный, обрезаем до 8000 символов")
                prompt = prompt[:8000]
            
            # Разделяем системный промпт и пользовательский ввод
            system_prompt = prompt.split("\n\nТекст обсуждения:")[0]
            user_input = prompt.split("\n\nТекст обсуждения:")[1] if "\n\nТекст обсуждения:" in prompt else ""
            
            # Формируем сообщения для API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            payload = {
                "messages": messages,
                "max_tokens": self.max_response_tokens,
                "temperature": self.temperature,
                "top_p": 1.0,
                "stop": ["</s>", "<|end_of_turn|>"]  # Добавляем стоп-токены
            }
            
            url = f"http://{self.server_host}:{self.server_port}/v1/chat/completions"
            
            # Логируем запрос
            logging.info(f"Отправка запроса к llama-server:\nURL: {url}\nPayload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    # Логируем ответ
                    logging.info(f"Ответ от llama-server:\nStatus: {resp.status_code}\nHeaders: {dict(resp.headers)}\nBody: {resp.text}")
                    
                    if resp.status_code != 200:
                        logging.error(f"Llama server error during summarization: {resp.text}")
                        return None
                    data = resp.json()
                    response = data["choices"][0]["message"]["content"].strip()
                    
                    # Проверяем, что ответ не пустой и не содержит только пробелы
                    if not response or response.isspace():
                        logging.warning("Получен пустой ответ от llama-server")
                        return "Ошибка: получен пустой ответ"
                    
                    return response
            except Exception as e:
                logging.error(f"Ошибка генерации сводки: {e}")
                return None

# -----------------------------------------------------------------------------
# Класс SummaryProcessor: Сводит сообщения из логов с помощью Llama
# -----------------------------------------------------------------------------
class SummaryProcessor:
    def __init__(self, chunk_size: int = 4000):
        self.llama = LlamaSummary()
        self.chunk_size = chunk_size
        self._processing_locks: Dict[str, asyncio.Lock] = {}
        self._last_auto_summary_time: Dict[int, datetime] = {}  # Оставляем только время последней автоматической сводки
        self._cache_file = Path("summary_cache.json")
        self._load_cache()
        self.summary_system_prompt = (
            "<s>Ты - ассистент по созданию кратких и точных сводок обсуждений.\n"
            "Твоя задача - создавать структурированные сводки на русском языке.\n\n"
            "КРИТИЧЕСКИ ВАЖНО:\n"
            "1. ОБЯЗАТЕЛЬНО отвечай ТОЛЬКО на русском языке\n"
            "2. НИКОГДА не используй английский язык\n"
            "3. Если получаешь текст на английском - переведи его на русский\n\n"
            "ПРАВИЛА СВОДКИ:\n"
            "1. Используй только фактическую информацию из обсуждения\n"
            "2. Не добавляй свои комментарии и оценки\n"
            "3. Сохраняй деловой стиль\n"
            "4. Не используй художественные элементы\n\n"
            "ФОРМАТ ВЫВОДА:\n"
            "Тема: [основная тема обсуждения]\n"
            "Период: [время начала] - [время окончания]\n"
            "Содержание: [2-3 предложения о конкретных обсуждаемых вопросах и решениях]\n\n"
            "ПРИМЕР:\n"
            "Тема: Обсуждение проекта\n"
            "Период: 10:00 - 10:30\n"
            "Содержание: Обсуждались сроки реализации проекта. Принято решение о начале работ в следующем месяце.\n\n"
            "ЗАПРЕЩЕНО:\n"
            "- Использовать английский язык\n"
            "- Добавлять свои комментарии\n"
            "- Использовать эмоциональные оценки\n"
            "- Включать информацию не из обсуждения</s>"
        )

    def _load_cache(self) -> None:
        """Загружает только время последней автоматической сводки."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Загружаем только время последних автоматических сводок
                    self._last_auto_summary_time = {
                        int(k): datetime.fromisoformat(v)
                        for k, v in data.get('last_auto_times', {}).items()
                    }
                logger.info(f"Загружены данные о последних автоматических сводках")
            except Exception as e:
                logger.error(f"Ошибка загрузки кэша: {e}")

    def _save_cache(self) -> None:
        """Сохраняет только время последней автоматической сводки."""
        try:
            data = {
                'last_auto_times': {
                    str(k): v.isoformat()
                    for k, v in self._last_auto_summary_time.items()
                }
            }
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Сохранены данные о последних автоматических сводках")
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def _group_messages_by_topic(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Улучшенная группировка сообщений по темам с учетом:
        - Размера контекста
        - Семантической связности
        - Временных промежутков
        """
        if not messages:
            return []
        
        MAX_TOPIC_SIZE = self.chunk_size
        MIN_MESSAGES = 3  # минимальное количество сообщений для топика
        
        topics = []
        current_topic = {
            'messages': [],
            'start_time': None,
            'end_time': None,
            'total_chars': 0
        }
        
        for msg in messages:
            try:
                timestamp = datetime.fromisoformat(msg['timestamp'])
                msg_text = msg['text']
                msg_size = len(msg_text)
                
                # Начинаем новый топик если:
                should_start_new = (
                    # Текущий топик пустой
                    not current_topic['messages'] or
                    # Прошло больше 30 минут с последнего сообщения
                    (timestamp - current_topic['end_time']).total_seconds() > 1800 or
                    # Превышен максимальный размер топика И в текущем топике достаточно сообщений
                    (current_topic['total_chars'] + msg_size > MAX_TOPIC_SIZE and 
                     len(current_topic['messages']) >= MIN_MESSAGES)
                )
                
                if should_start_new:
                    if current_topic['messages']:
                        topics.append(current_topic)
                    current_topic = {
                        'messages': [msg],
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'total_chars': msg_size
                    }
                else:
                    current_topic['messages'].append(msg)
                    current_topic['end_time'] = timestamp
                    current_topic['total_chars'] += msg_size
                    
            except (ValueError, KeyError) as e:
                logging.error(f"Ошибка обработки сообщения: {e}")
                continue
        
        # Добавляем последний топик
        if current_topic['messages']:
            topics.append(current_topic)
        
        # Объединяем маленькие последовательные топики
        merged_topics = []
        current_merged = None
        
        for topic in topics:
            if not current_merged:
                current_merged = topic
                continue
                
            # Объединяем топики если они небольшие и между ними менее 10 минут
            time_diff = (topic['start_time'] - current_merged['end_time']).total_seconds()
            total_size = current_merged['total_chars'] + topic['total_chars']
            
            if time_diff <= 600 and total_size <= MAX_TOPIC_SIZE:
                current_merged['messages'].extend(topic['messages'])
                current_merged['end_time'] = topic['end_time']
                current_merged['total_chars'] = total_size
            else:
                merged_topics.append(current_merged)
                current_merged = topic
        
        if current_merged:
            merged_topics.append(current_merged)
        
        logging.info(f"Сгруппировано {len(merged_topics)} топиков из {len(messages)} сообщений")
        return merged_topics

    def _split_large_topic(self, text: str) -> List[str]:
        """Разбивает большой топик на подтопики с учетом смысловых границ."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Разбиваем текст на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _create_summary_prompt(self, text: str, start_time: str, end_time: str) -> str:
        """Создает промпт для суммаризации текста."""
        prompt = (
            f"{self.summary_system_prompt}\n\n"
            f"Текст обсуждения:\n{text}\n\n"
            f"Время обсуждения: {start_time} - {end_time}"
        )
        logging.info(f"Создан промпт для суммаризации:\n{prompt}")
        return prompt

    def _create_combining_prompt(self, summaries: List[str]) -> str:
        """Создает промпт для объединения саммари подтопиков."""
        prompt = (
            f"{self.summary_system_prompt}\n\n"
            "Объедини следующие части обсуждения в одну связную сводку:\n\n"
            f"{'\n---\n'.join(summaries)}"
        )
        logging.info(f"Создан промпт для объединения сводок:\n{prompt}")
        return prompt

    def update_last_auto_summary_time(self, chat_id: int) -> None:
        """Обновляет время последней автоматической сводки для чата."""
        self._last_auto_summary_time[chat_id] = datetime.now()
        self._save_cache()

    async def summarize(self, chat_id: int, is_manual: bool = False) -> str:
        """Создает сводку сообщений с улучшенной обработкой топиков."""
        cache_key = f"{chat_id}_{datetime.now().strftime('%Y-%m-%d')}"
        
        # Получаем или создаем блокировку для этого чата
        if cache_key not in self._processing_locks:
            self._processing_locks[cache_key] = asyncio.Lock()
        
        # Используем блокировку для предотвращения параллельной обработки
        async with self._processing_locks[cache_key]:
            # Получаем сообщения
            messages = ChatLogger(Path(os.getenv("LOGS_DIR", "./chat_logs"))).get_messages_today(chat_id)
            
            # Для ручной сводки фильтруем сообщения по времени последней автоматической сводки
            if is_manual and chat_id in self._last_auto_summary_time:
                last_auto_time = self._last_auto_summary_time[chat_id]
                messages = [
                    msg for msg in messages 
                    if datetime.fromisoformat(msg['timestamp']) > last_auto_time
                ]
                if not messages:
                    return "Нет новых сообщений с момента последней автоматической сводки."
            
            if not messages:
                return "Нет сообщений для суммаризации."
            
            topics = self._group_messages_by_topic(messages)
            logging.info(f"Сгруппировано топиков для чата {chat_id}: {len(topics)}")
            
            current_time = datetime.now()
            date_str = current_time.strftime("%d.%m.%Y")
            time_str = current_time.strftime("%H:%M")
            
            if not topics:
                return f"Дата: {date_str}\nВремя создания сводки: {time_str}\n\n(Нет активных обсуждений)"
            
            final_summary = f"Дата: {date_str}\nВремя создания сводки: {time_str}\n\n"
            
            for i, topic in enumerate(topics, start=1):
                start_str = topic['start_time'].strftime('%H:%M')
                end_str = topic['end_time'].strftime('%H:%M')
                
                # Анализируем размер топика
                topic_text = "\n".join(m['text'] for m in topic['messages'])
                if topic['total_chars'] > self.chunk_size:
                    logging.info(f"Большой топик {i}, размер: {topic['total_chars']} символов")
                    # Разбиваем большой топик на подтопики для лучшей суммаризации
                    subtopics = self._split_large_topic(topic_text)
                    subtopic_summaries = []
                    
                    for j, subtopic in enumerate(subtopics, start=1):
                        logging.info(f"Обработка подтопика {j} топика {i}")
                        prompt = self._create_summary_prompt(subtopic, start_str, end_str)
                        summary = await self.llama.generate_summary(prompt)
                        if summary:
                            subtopic_summaries.append(summary)
                    
                    # Объединяем саммари подтопиков
                    if subtopic_summaries:
                        combined_prompt = self._create_combining_prompt(subtopic_summaries)
                        topic_summary = await self.llama.generate_summary(combined_prompt)
                    else:
                        topic_summary = "(Не удалось получить сводку для большого топика)"
                else:
                    logging.info(f"Обработка топика {i}, размер: {topic['total_chars']} символов")
                    prompt = self._create_summary_prompt(topic_text, start_str, end_str)
                    topic_summary = await self.llama.generate_summary(prompt)
                
                if not topic_summary:
                    topic_summary = "(Не удалось получить сводку)"
                
                final_summary += f"Обсуждение {i}:\n{topic_summary}\n\n"
            
            return final_summary
