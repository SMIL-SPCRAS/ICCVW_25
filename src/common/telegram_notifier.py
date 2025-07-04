import requests


class TelegramNotifier:
    def __init__(self, cfg: dict[str, any]) -> None:
        self.token = cfg['telegram_token']
        self.chat_id = cfg['telegram_chat_id']
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send(self, message: str) -> None:
        data = {"chat_id": self.chat_id, "text": message}
        try:
            response = requests.post(self.api_url, data=data)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to send Telegram message: {e}")