import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import re
from torch.optim import AdamW

from utils import *

class CoffeeOrderDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_len=128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

        # BIO tags를 정의 {BIO tags : priority}
        self.tag2id = {
            'O': 0,
            'B-MENU': 1, 'I-MENU': 2,
            'B-QUANTITY': 3, 'I-QUANTITY': 4,
            'B-TEMPERATURE': 5, 'I-TEMPERATURE': 6,
            'B-OPTION': 7, 'I-OPTION': 8,
            'PAD': 9
        }
        self.id2tag = {v: k for k, v in self.tag2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tags = self.tags[idx]

        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = []
        word_ids = encoding.word_ids()

        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                labels.append(self.tag2id.get(tags[word_idx], self.tag2id['O']))

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels)
        }

class CoffeeOrderSystem:
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path or "klue/bert-base"

        self.menu_items = {
            "아메리카노": {"type": "커피", "base_price": 4000},
            "카페라떼": {"type": "커피", "base_price": 4500},
            "카푸치노": {"type": "커피", "base_price": 4500},
            "에스프레소": {"type": "커피", "base_price": 3500}
        }

        self.required_fields = {
            "menu_item": "메뉴",
            "temperature": "온도",
            "quantity": "수량"
        }

        self.field_mapping = {
            "메뉴": "menu_item",
            "온도": "temperature",
            "수량": "quantity"
        }

        self.valid_options = {
            "temperature": ["뜨거운", "차가운"],
            "size": ["레귤러", "톨", "그란데"],
            "sugar": ["없음", "추가"]
        }

        self.label_list = [
            'O',
            'B-MENU', 'I-MENU',
            'B-QUANTITY', 'I-QUANTITY',
            'B-TEMPERATURE', 'I-TEMPERATURE',
            'B-OPTION', 'I-OPTION',
            'PAD'
        ]
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)

        self.nlp = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1,
            aggregation_strategy="simple"
        )

    def train(self, train_texts, train_tags, epochs=5, batch_size=16):
        dataset = CoffeeOrderDataset(train_texts, train_tags, self.tokenizer)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            progress_bar = tqdm(train_dataloader, desc='Training')

            total_loss = 0
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_dataloader)
            print(f'\nAverage loss: {avg_loss:.4f}')

    def evaluate(self, test_texts, test_tags):
        self.model.eval()
        predictions = []
        real_labels = []

        with torch.no_grad():
            for text, tags in zip(test_texts, test_tags):
                encoding = self.tokenizer(
                    text.split(),
                    is_split_into_words=True,
                    return_offsets_mapping=True,
                    return_tensors='pt',
                    truncation=True,
                    max_length=128
                )
                tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
                word_ids = encoding.word_ids()

                labels = []
                for word_idx in word_ids:
                    if word_idx is None:
                        continue
                    else:
                        labels.append(tags[word_idx])

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                predictions_ids = torch.argmax(logits, dim=2)[0].cpu().numpy()
                pred_tags = [self.id2label[pred_id] for pred_id in predictions_ids]

                pred_labels = []
                prev_word_idx = None
                for idx, word_idx in enumerate(word_ids):
                    if word_idx != prev_word_idx and word_idx is not None:
                        pred_labels.append(pred_tags[idx])
                        prev_word_idx = word_idx

                # 실제 태그와 예측 태그의 길이가 일치하는지 확인
                if len(pred_labels) != len(tags):
                    min_length = min(len(pred_labels), len(tags))
                    pred_labels = pred_labels[:min_length]
                    tags = tags[:min_length]

                predictions.append(pred_labels)
                real_labels.append(tags)

                print(f"\n텍스트: {text}")
                print(f"예측 태그: {pred_labels}")
                print(f"실제 태그: {tags}")

        print("\n=== 평가 결과 ===")
        report = classification_report(real_labels, predictions, digits=4)
        print(report)
        return report

    def parse_order(self, text: str) -> Dict:
        outputs = self.nlp(text)
        print("Debug - NLP 출력:", outputs)

        order = {
            "menu_item": None,
            "quantity": None,
            "temperature": None,
            "options": {}
        }

        for entity in outputs:
            entity_type = entity['entity_group']
            entity_type = entity_type.replace("B-", "").replace("I-", "")
            word = entity['word'].replace('##', '').strip()

            print(f"Debug - 처리중: {word} ({entity_type})")

            if entity_type == 'MENU':
                for menu_item in self.menu_items.keys():
                    if menu_item in word:
                        order["menu_item"] = menu_item
                        break
            elif entity_type == 'QUANTITY':
                try:
                    nums = ''.join(filter(str.isdigit, word))
                    if nums:
                        order["quantity"] = int(nums)
                    else:
                        numbers = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}
                        for k, v in numbers.items():
                            if k in word:
                                order["quantity"] = v
                                break
                except Exception as e:
                    print(f"Error parsing quantity: {e}")
            elif entity_type == 'TEMPERATURE':
                if any(temp in word for temp in ["아이스", "차가운", "시원"]):
                    order["temperature"] = "차가운"
                elif any(temp in word for temp in ["따뜻", "뜨거운", "핫"]):
                    order["temperature"] = "뜨거운"

        print("Debug - 최종 주문:", order)
        return order

    def get_missing_fields(self, order: Dict) -> List[str]:
        return [name for field, name in self.required_fields.items()
                if order.get(field) is None]

    def generate_question(self, missing_field: str) -> str:
        questions = {
            "메뉴": "어떤 메뉴를 원하시나요?",
            "온도": "뜨거운 음료로 드릴까요, 차가운 음료로 드릴까요?",
            "수량": "몇 잔 원하시나요?",
        }

        field_key = self.field_mapping.get(missing_field, missing_field.lower())

        if field_key in self.valid_options:
            options = self.valid_options[field_key]
            return f"{questions[missing_field]} ({'/'.join(options)})"
        return questions.get(missing_field, f"{missing_field}을(를) 선택해주세요.")

    def format_order(self, order: Dict) -> str:
        if not order["menu_item"]:
            return "주문 내역이 없습니다."

        result = f"\n현재 주문 내역:\n- 메뉴: {order['menu_item']}"

        if order.get("quantity"):
            result += f"\n- 수량: {order['quantity']}잔"
        if order.get("temperature"):
            result += f"\n- 온도: {order['temperature']}"

        if order.get("quantity"):
            base_price = self.menu_items[order["menu_item"]]["base_price"]
            total_price = base_price * order["quantity"]
            result += f"\n\n총 금액: {total_price:,}원"

        return result

def interactive_order_session(system: CoffeeOrderSystem):
    print("=== 커피 주문 시스템 ===")
    order_text = input("주문하실 메뉴를 말씀해주세요: ")

    order_text = order_text.strip()
    if not order_text.endswith(('요', '다')):
        order_text += " 주세요"

    order = system.parse_order(order_text)
    print(system.format_order(order))

    missing_fields = system.get_missing_fields(order)
    while missing_fields:
        field = missing_fields[0]
        question = system.generate_question(field)
        print(f"\n{question}")

        response = input("답변: ").strip()
        if not response.endswith(('요', '다')):
            response += "요"

        new_order = system.parse_order(response)

        if field == "메뉴" and new_order["menu_item"]:
            order["menu_item"] = new_order["menu_item"]
        elif field == "온도" and new_order["temperature"]:
            order["temperature"] = new_order["temperature"]
        elif field == "수량" and new_order["quantity"]:
            order["quantity"] = new_order["quantity"]
        else:
            if field == "메뉴":
                for menu in system.menu_items:
                    if menu in response:
                        order["menu_item"] = menu
                        break
            elif field == "온도":
                if any(word in response for word in ["차가운", "아이스", "시원"]):
                    order["temperature"] = "차가운"
                elif any(word in response for word in ["따뜻", "뜨거운", "핫"]):
                    order["temperature"] = "뜨거운"
            elif field == "수량":
                try:
                    nums = ''.join(filter(str.isdigit, response))
                    if nums:
                        order["quantity"] = int(nums)
                    else:
                        numbers = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}
                        for k, v in numbers.items():
                            if k in response:
                                order["quantity"] = v
                                break
                except ValueError:
                    pass

        print(system.format_order(order))
        missing_fields = system.get_missing_fields(order)

    print("\n주문이 완료되었습니다!")

if __name__ == "__main__":
    system = CoffeeOrderSystem()
    
    train_texts, train_tags = get_training_data()
    print("모델 학습 시작...")
    system.train(train_texts, train_tags, epochs=5)

    print("\n모델 평가:")
    eval_results = system.evaluate(train_texts, train_tags)
    print(eval_results)

    interactive_order_session(system)