"""
AI-анализатор отзывов о дронах
Объединенный файл для анализа отзывов с помощью LLM моделей
"""
import json
import os
import re
import warnings
from typing import Dict, Optional
from dotenv import load_dotenv
import pandas as pd

# Подавляем предупреждения SSL
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Конфигурация
load_dotenv(override=True, encoding='utf-8-sig')

def get_env_key(key_name: str) -> str:
    value = os.getenv(key_name, "")
    return value.strip().lstrip('\ufeff') if value else ""

GROQ_API_KEY = get_env_key("GROQ_API_KEY")
GIGACHAT_CLIENT_ID = get_env_key("GIGACHAT_CLIENT_ID")
GIGACHAT_CLIENT_SECRET = get_env_key("GIGACHAT_CLIENT_SECRET")

# Промпт для анализа
PROMPT_TEMPLATE = """Проанализируй следующий отзыв о дроне/квадрокоптере и верни результат в формате JSON:

Отзыв: "{review_text}"

Верни JSON со следующими полями (пример формата):
{{
    "sentiment": "положительный",
    "main_topic": "качество камеры",
    "issue": "короткое время работы батареи",
    "rating": 4
}}

Описание полей:
- "sentiment": тональность отзыва - "положительный", "нейтральный" или "отрицательный"
- "main_topic": основная тема или преимущество (например, "качество камеры", "время работы батареи", "управление")
- "issue": основная проблема, если есть (если проблемы нет, верни пустую строку "")
- "rating": оценка от 1 до 5 (где 1 - очень плохо, 5 - отлично)

Верни ТОЛЬКО JSON без дополнительных комментариев в точно таком же формате."""


def parse_json_response(response_text: str) -> Optional[Dict]:
    """Парсинг JSON из ответа модели"""
    response_text = response_text.strip()
    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


class GigaChatAnalyzer:
    """Анализатор на основе GigaChat"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
    
    def analyze(self, review_text: str) -> Optional[Dict]:
        if not self.client_id or not self.client_secret:
            return None
        
        try:
            import requests
            import base64
            
            prompt = PROMPT_TEMPLATE.format(review_text=review_text)
            
            # Получаем токен
            token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
            credentials = f"{self.client_id}:{self.client_secret}"
            credentials_b64 = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {credentials_b64}",
                "RqUID": self.client_id,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            token_response = requests.post(
                token_url,
                headers=headers,
                data={"scope": "GIGACHAT_API_PERS"},
                verify=False,
                timeout=30
            )
            
            if token_response.status_code != 200:
                return None
            
            access_token = token_response.json().get("access_token")
            if not access_token:
                return None
            
            # Запрос к API
            api_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
            chat_headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "GigaChat-Pro",
                "messages": [
                    {"role": "system", "content": "Ты - эксперт по анализу отзывов. Всегда отвечай в формате JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            api_response = requests.post(api_url, headers=chat_headers, json=payload, verify=False, timeout=60)
            
            if api_response.status_code != 200:
                return None
            
            response_data = api_response.json()
            if "choices" not in response_data or not response_data["choices"]:
                return None
            
            result_text = response_data["choices"][0].get("message", {}).get("content", "")
            if not result_text:
                return None
            
            parsed_result = parse_json_response(result_text)
            if parsed_result:
                return {
                    "sentiment": parsed_result.get("sentiment", "нейтральный"),
                    "main_topic": parsed_result.get("main_topic", ""),
                    "issue": parsed_result.get("issue", ""),
                    "rating": parsed_result.get("rating", 3)
                }
            
            return None
        except:
            return None


class GroqAnalyzer:
    """Анализатор на основе Groq"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        except:
            self.client = None
    
    def analyze(self, review_text: str) -> Optional[Dict]:
        if not self.client:
            return None
        
        try:
            prompt = PROMPT_TEMPLATE.format(review_text=review_text)
            models = ["llama-3.1-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"]
            
            for model in models:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Ты - эксперт по анализу отзывов. Всегда отвечай в формате JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        response_format={"type": "json_object"}
                    )
                    
                    result_text = response.choices[0].message.content
                    parsed_result = parse_json_response(result_text)
                    
                    if parsed_result:
                        return {
                            "sentiment": parsed_result.get("sentiment", "нейтральный"),
                            "main_topic": parsed_result.get("main_topic", ""),
                            "issue": parsed_result.get("issue", ""),
                            "rating": parsed_result.get("rating", 3)
                        }
                except:
                    continue
            
            return None
        except:
            return None


class ModelComparison:
    """Сравнение результатов моделей"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, review_id: int, review_text: str, model_name: str, analysis: Dict):
        self.results.append({
            'review_id': review_id,
            'review_text': review_text,
            'model': model_name,
            'sentiment': analysis.get('sentiment', 'нейтральный'),
            'main_topic': analysis.get('main_topic', ''),
            'issue': analysis.get('issue', ''),
            'rating': analysis.get('rating', 3)
        })
    
    def get_comparison_dataframe(self):
        return pd.DataFrame(self.results)


def create_analyzers():
    """Создание анализаторов"""
    analyzers = {}
    
    if GROQ_API_KEY:
        try:
            analyzers['groq'] = GroqAnalyzer(GROQ_API_KEY)
        except:
            pass
    
    if GIGACHAT_CLIENT_ID and GIGACHAT_CLIENT_SECRET:
        try:
            analyzers['gigachat'] = GigaChatAnalyzer(GIGACHAT_CLIENT_ID, GIGACHAT_CLIENT_SECRET)
        except:
            pass
    
    return analyzers


def load_reviews(filename: str = "reviews.json"):
    """Загрузка отзывов из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        return [r for r in reviews if isinstance(r, dict) and r.get('text')]
    except:
        return []


def save_results(reviews, comparison, output_dir="results"):
    """Сохранение результатов"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "reviews.json"), 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    
    df = comparison.get_comparison_dataframe()
    df.to_json(os.path.join(output_dir, "detailed_results.json"), orient='records', force_ascii=False, indent=2)


def main():
    """Основная функция"""
    reviews = load_reviews("reviews.json")
    if not reviews:
        return
    
    analyzers = create_analyzers()
    if not analyzers:
        return
    
    comparison = ModelComparison()
    reviews_with_analysis = []
    
    for review_id, review_data in enumerate(reviews, 1):
        review_text = review_data.get('text', '')
        if not review_text:
            continue
        
        review_analysis = {
            'review_id': review_id,
            'review_text': review_text,
            'original_rating': review_data.get('rating'),
            'original_author': review_data.get('author'),
            'source_url': review_data.get('source_url'),
            'model_results': {}
        }
        
        for model_name, analyzer in analyzers.items():
            try:
                analysis = analyzer.analyze(review_text)
                if analysis:
                    review_analysis['model_results'][model_name] = analysis
                    comparison.add_result(review_id, review_text, model_name, analysis)
            except:
                continue
        
        reviews_with_analysis.append(review_analysis)
    
    save_results(reviews_with_analysis, comparison)


if __name__ == "__main__":
    main()
