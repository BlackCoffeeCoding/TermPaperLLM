from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TransformersLLMClient:
    def __init__(self):
        model_id = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Загрузка модели {model_id} на устройство {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        print("Модель загружена.")

        self.system_prompt = (
            "Ты — ассистент для помощи веб-разработчикам на Spring Boot. Отвечай кратко и по делу."
        )
        self.chat_history = []

    def reset_history(self):
        self.chat_history = []

    def build_prompt(self, last_user_input: str) -> str:
        conversation = ""
        for turn in self.chat_history[-6:]:  # количество сохраняемых в истории последних сообщений
            conversation += f"<Пользователь>: {turn['user']}\n<Ассистент>: {turn['bot']}\n"
        conversation += f"<Пользователь>: {last_user_input}\n<Ассистент>:"
        return f"{self.system_prompt}\n{conversation}"

    def ask(self, user_input: str) -> str:
        self.chat_history.append({"user": user_input, "bot": ""})

        prompt = self.build_prompt(user_input)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=1024,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Выделяем ответ ассистента
        if "<Ассистент>:" in generated_text:
            answer = generated_text.split("<Ассистент>:")[-1].strip()
        else:
            answer = generated_text.strip()

        # Убираем возможные вставки следующего запроса
        answer = answer.split("<Пользователь>:")[0].strip()

        self.chat_history[-1]["bot"] = answer
        return answer