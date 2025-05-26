import requests
import uuid

class GigaChatClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.token = None
        self.get_token()

    def get_token(self):
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        rq_uid = str(uuid.uuid4())
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': rq_uid,
            'Authorization': f'Basic {self.api_key}'
        }
        payload = {
            'grant_type': 'client_credentials',
            'scope': 'GIGACHAT_API_PERS'
        }

        response = requests.post(url, headers=headers, data=payload, verify=False)

        print("=== Ответ от сервера при получении токена ===")
        print("Ответ:", response.text)
        print("=============================================")

        response.raise_for_status()
        self.token = response.json()['access_token']

    def ask(self, messages: list[dict]) -> str:
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        rq_uid = str(uuid.uuid4())
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
            'RqUID': rq_uid,
            'Accept': 'application/json'
        }
        data = {
            "model": "GigaChat:latest",  # Это работает точно
            "messages": messages
        }
        response = requests.post(url, headers=headers, json=data, verify=False)
        response.raise_for_status()
        res_json = response.json()
        return res_json['choices'][0]['message']['content']
