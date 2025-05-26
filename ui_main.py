import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QMessageBox, QDialog
)
from gigachat_api import GigaChatClient
from local_llm import TransformersLLMClient

# Импорт matplotlib для графиков
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class StatsWindow(QDialog):
    def __init__(self, ratings_giga, ratings_local):
        super().__init__()
        self.setWindowTitle("Статистика оценок")
        self.resize(900, 400)

        layout = QVBoxLayout(self)

        # Подсчёт среднего балла
        avg_giga = np.mean(ratings_giga) if ratings_giga else 0
        avg_local = np.mean(ratings_local) if ratings_local else 0

        label_giga = QLabel(f"Средний балл GigaChat: {avg_giga:.2f} сформирован по {len(ratings_giga)} оценкам")
        label_local = QLabel(f"Средний балл Локальной LLM: {avg_local:.2f} сформирован по {len(ratings_local)} оценкам")

        layout.addWidget(label_giga)
        layout.addWidget(label_local)

        # Создаём фигуру для графиков
        fig = Figure(figsize=(5, 3))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Гистограмма для GigaChat
        if ratings_giga:
            bins = np.arange(1, 7) - 0.5  # для чисел 1-5 по центру столбца
            ax1.hist(ratings_giga, bins=bins, rwidth=0.8, color='blue')
            ax1.set_xticks(range(1, 6))
            ax1.set_title("Распределение оценок GigaChat")
            ax1.set_xlabel("Оценка")
            ax1.set_ylabel("Количество ответов")
        else:
            ax1.text(0.5, 0.5, "Нет оценок GigaChat", ha='center', va='center')

        # Гистограмма для Local LLM
        if ratings_local:
            bins = np.arange(1, 7) - 0.5
            ax2.hist(ratings_local, bins=bins, rwidth=0.8, color='green')
            ax2.set_xticks(range(1, 6))
            ax2.set_title("Распределение оценок Локальной LLM")
            ax2.set_xlabel("Оценка")
            ax2.set_ylabel("Количество ответов")
        else:
            ax2.text(0.5, 0.5, "Нет оценок Локальной LLM", ha='center', va='center')

        fig.tight_layout()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Диалоговое приложение")
        self.resize(700, 600)

        # Истории диалогов
        self.history_gigachat = []
        self.history_local = []
        self.history_compare = []
        self.history_compare_gigachat = []
        self.history_compare_local = []

        # Для хранения оценок (списки)
        self.ratings_giga = []
        self.ratings_local = []

        self.current_mode = None

        self.gigachat_client = GigaChatClient("YTBiYjgwNmYtMmUxZi00ODVhLTg0YjQtYjAzN2U5OWI5Njc4OmZkMjUzODUzLTlkNGItNDYzNy05NDU3LTJiNmExYmFkNDZiNQ==")

        self.local_llm_client = TransformersLLMClient()

        self.layout = QVBoxLayout(self)

        self.label_mode = QLabel("Выберите режим и начните диалог")
        self.layout.addWidget(self.label_mode)

        self.btn_gigachat = QPushButton("Диалог с GigaChat (сервер)")
        self.btn_local = QPushButton("Диалог с локальной LLM")
        self.btn_compare = QPushButton("Диалог с двумя моделями")

        self.layout.addWidget(self.btn_gigachat)
        self.layout.addWidget(self.btn_local)
        self.layout.addWidget(self.btn_compare)

        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        self.layout.addWidget(self.chat_output)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите ваш вопрос здесь...")
        self.layout.addWidget(self.input_field)

        self.btn_send = QPushButton("Отправить")
        self.layout.addWidget(self.btn_send)

        self.btn_end = QPushButton("Завершить диалог")
        self.layout.addWidget(self.btn_end)

        # Кнопка показа статистики
        self.btn_show_stats = QPushButton("Показать статистику оценок")
        self.layout.addWidget(self.btn_show_stats)
        self.btn_show_stats.setEnabled(False)  # включим когда появятся оценки

        # Виджеты оценки для GigaChat и локальной LLM
        self.rating_label_giga = QLabel("Оцените ответ GigaChat (1–5):")
        self.rating_label_local = QLabel("Оцените ответ Локальной LLM (1–5):")

        self.rating_widget_giga = QWidget()
        self.rating_layout_giga = QHBoxLayout(self.rating_widget_giga)
        self.rating_buttons_giga = []
        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda _, rating=i: self.handle_rating("gigachat", rating))
            self.rating_buttons_giga.append(btn)
            self.rating_layout_giga.addWidget(btn)
        self.rating_widget_giga.hide()

        self.rating_widget_local = QWidget()
        self.rating_layout_local = QHBoxLayout(self.rating_widget_local)
        self.rating_buttons_local = []
        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda _, rating=i: self.handle_rating("local", rating))
            self.rating_buttons_local.append(btn)
            self.rating_layout_local.addWidget(btn)
        self.rating_widget_local.hide()

        self.layout.addWidget(self.rating_label_giga)
        self.layout.addWidget(self.rating_widget_giga)
        self.layout.addWidget(self.rating_label_local)
        self.layout.addWidget(self.rating_widget_local)

        self.rating_label_giga.hide()
        self.rating_label_local.hide()

        # Сигналы
        self.btn_gigachat.clicked.connect(self.start_gigachat)
        self.btn_local.clicked.connect(self.start_local)
        self.btn_compare.clicked.connect(self.start_compare)
        self.btn_send.clicked.connect(self.handle_input)
        self.btn_end.clicked.connect(self.end_dialog)
        self.btn_show_stats.clicked.connect(self.show_statistics_window)

        # Флаги оценки
        self.rating_giga_rated = False
        self.rating_local_rated = False

    def start_gigachat(self):
        self.current_mode = "gigachat"
        self.label_mode.setText("Режим: Диалог с GigaChat (сервер)")
        self.load_history()
        self.append_to_history("Начинаем диалог с GigaChat.\n", mode="gigachat")
        self.hide_rating_widgets()
        self.btn_show_stats.setEnabled(False)

    def start_local(self):
        self.current_mode = "local"
        self.label_mode.setText("Режим: Диалог с локальной LLM")
        self.load_history()
        self.append_to_history("Начинаем диалог с локальной моделью.\n", mode="local")
        self.local_llm_client.reset_history()
        self.hide_rating_widgets()
        self.btn_show_stats.setEnabled(False)

    def start_compare(self):
        self.current_mode = "compare"
        self.label_mode.setText("Режим: Сравнение ответов двух моделей")
        self.load_history()
        self.append_to_history("Начинаем диалог с двумя моделями.\n", mode="compare")
        self.local_llm_client.reset_history()
        self.hide_rating_widgets()
        self.rating_giga_rated = False
        self.rating_local_rated = False
        self.btn_show_stats.setEnabled(bool(self.ratings_giga or self.ratings_local))

    def load_history(self):
        self.chat_output.clear()
        if self.current_mode == "gigachat":
            for line in self.history_gigachat:
                self.chat_output.append(line)
        elif self.current_mode == "local":
            for line in self.history_local:
                self.chat_output.append(line)
        elif self.current_mode == "compare":
            for line in self.history_compare:
                self.chat_output.append(line)

    def append_to_history(self, text: str, mode: str):
        self.chat_output.append(text)
        if mode == "gigachat":
            self.history_gigachat.append(text)
        elif mode == "local":
            self.history_local.append(text)
        elif mode == "compare":
            self.history_compare.append(text)

    def handle_input(self):
        if self.current_mode is None:
            QMessageBox.warning(self, "Ошибка", "Выберите режим работы!")
            return

        user_text = self.input_field.text().strip()
        if not user_text:
            return

        self.input_field.clear()

        if self.current_mode == "gigachat":
            self.append_to_history(f"🧑‍💻 Вы: {user_text}", mode="gigachat")
            messages = [{"role": "system", "content": "Ты — ассистент для помощи веб-разработчикам на Spring Boot. Отвечай кратко и по делу."}]
            for entry in self.history_gigachat:
                if entry.startswith("🧑‍💻 Вы: "):
                    messages.append({"role": "user", "content": entry[8:]})
                elif entry.startswith("🤖 GigaChat: "):
                    messages.append({"role": "assistant", "content": entry[13:]})
            try:
                response = self.gigachat_client.ask(messages)
                self.append_to_history(f"🤖 GigaChat: {response}", mode="gigachat")
            except Exception as e:
                self.append_to_history(f"❌ Ошибка: {e}", mode="gigachat")
            self.hide_rating_widgets()

        elif self.current_mode == "local":
            self.append_to_history(f"🧑‍💻 Вы: {user_text}", mode="local")
            try:
                response = self.local_llm_client.ask(user_text)
                self.append_to_history(f"🤖 Локальная LLM: {response}", mode="local")
            except Exception as e:
                self.append_to_history(f"❌ Ошибка: {e}", mode="local")
            self.hide_rating_widgets()

        elif self.current_mode == "compare":
            self.append_to_history(f"🧑‍💻 Вы: {user_text}", mode="compare")
            self.history_compare_gigachat.append({"role": "user", "content": user_text})
            messages_giga = [{"role": "system", "content": "Ты — ассистент для помощи веб-разработчикам на Spring Boot. Отвечай кратко и по делу."}] + self.history_compare_gigachat
            try:
                response_giga = self.gigachat_client.ask(messages_giga)
                self.history_compare_gigachat.append({"role": "assistant", "content": response_giga})
            except Exception as e:
                response_giga = f"❌ Ошибка GigaChat: {e}"

            # self.history_compare_local.append({"role": "user", "content": user_text})
            try:
                # context_local = ""
                # for msg in self.history_compare_local:
                #     prefix = "<Пользователь>: " if msg["role"] == "user" else "<Ассистент>:"
                #     context_local += f"{prefix} {msg['content']}\n"
                response_local = self.local_llm_client.ask(user_text)
                # self.history_compare_local.append({"role": "assistant", "content": response_local})
            except Exception as e:
                response_local = f"❌ Ошибка Локальной LLM: {e}"

            self.append_to_history(f"🤖 GigaChat: {response_giga}", mode="compare")
            self.append_to_history(f"🤖 Локальная LLM: {response_local}", mode="compare")

            self.show_rating_widgets()
            self.rating_giga_rated = False
            self.rating_local_rated = False

    def show_rating_widgets(self):
        self.rating_label_giga.show()
        self.rating_widget_giga.show()
        self.rating_label_local.show()
        self.rating_widget_local.show()

    def hide_rating_widgets(self):
        self.rating_label_giga.hide()
        self.rating_widget_giga.hide()
        self.rating_label_local.hide()
        self.rating_widget_local.hide()

    def handle_rating(self, model_name: str, rating: int):
        self.append_to_history(f"⭐ Оценка ответа {model_name}: {rating}", mode="compare")

        # Запоминаем оценки
        if model_name == "gigachat":
            self.rating_giga_rated = True
            self.ratings_giga.append(rating)
            self.rating_widget_giga.hide()
            self.rating_label_giga.hide()
        elif model_name == "local":
            self.rating_local_rated = True
            self.ratings_local.append(rating)
            self.rating_widget_local.hide()
            self.rating_label_local.hide()

        # Если обе оценки поставлены — скрываем виджеты и активируем кнопку статистики
        if self.rating_giga_rated and self.rating_local_rated:
            self.hide_rating_widgets()
            self.btn_show_stats.setEnabled(True)

    def show_statistics_window(self):
        if not (self.ratings_giga or self.ratings_local):
            QMessageBox.information(self, "Статистика", "Нет оценок для отображения.")
            return

        stats_win = StatsWindow(self.ratings_giga, self.ratings_local)
        stats_win.exec()

    def end_dialog(self):
        self.history_gigachat.clear()
        self.history_local.clear()
        self.history_compare.clear()
        self.history_compare_gigachat.clear()
        self.history_compare_local.clear()
        self.chat_output.clear()
        self.current_mode = None
        self.label_mode.setText("Диалог завершён. Выберите режим.")
        self.hide_rating_widgets()
        self.btn_show_stats.setEnabled(False)
        # Можно очистить оценки или не очищать — по желанию
        # self.ratings_giga.clear()
        # self.ratings_local.clear()