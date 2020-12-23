# Telegram Bot

## Project setup
```
cd myproject

python3 -m venv venv

. venv/bin/activate
```

### Train data
```
python3 train.py
```

### Setting ngrok
```
./ngrok http 8080
copy ngrok URL from terminal and paste it to the input parameter of updater.bot.setWebhook() in bot.py
```

### Run bot.py
```
python3 bot.py
```