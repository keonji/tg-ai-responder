
## Telegram бот
- Расшифровка голосовых сообщений
- GPT чат в ЛС и в группах через llama server
- Суммаризация сообщений в чате, выдача сводки
> Требуется минимум 24gb RAM

#### Установите нужное окружение:
```
sudo apt update && sudo apt upgrade -y
sudo apt install -y git cmake make g++ python3-pip python3-venv screen htop libopenblas-dev libopenblas64-dev pkg-config ffmpeg
pip install python-telegram-bot vosk transformers torch sentencepiece nest-asyncio python-dotenv tzlocal --break-system-packages
wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip
unzip vosk-model-ru-0.42.zip
mv vosk-model-ru-0.42 vosk-model
```

#### Тюнинг при необходимости:
```
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
echo "vm.overcommit_memory=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
echo "* soft nofile 65535" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65535" | sudo tee -a /etc/security/limits.conf
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

> При наличии AVX512 инструкций, укажите -DGGML_AVX512=ON тэг. В make -j8 вместо 8 укажите количество потоков ЦПУ.
#### Сборка llama:
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_NATIVE=ON -DGGML_AVX2=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
make -j8
```

Скачайте нужную gguf модель, положите ее в директорию models/, укажите путь в файле .env
Замените 'YOUR_TELEGRAM_BOT_TOKEN' в .env файле на токен вашего бота, полученный от @BotFather.

#### Запуск:
```
python3 bot.py
```

#### Автозапуск:
Файл сервиса /etc/systemd/system/telegram_voice_bot.service:
```
[Unit]
Description=Telegram Voice Bot
After=network.target

[Service]
User=yourusername
WorkingDirectory=/path/to/your/bot
ExecStart=/usr/bin/python3 /path/to/your/bot/bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

User: замените на ваше имя пользователя.
WorkingDirectory и ExecStart: замените на путь к вашему скрипту.

##### Активируйте и запустите сервис:
```
sudo systemctl daemon-reload
sudo systemctl enable telegram_voice_bot.service
sudo systemctl start telegram_voice_bot.service
```
