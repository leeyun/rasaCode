from typing import List, Tuple, Dict
import json
from pathlib import Path

class CoffeeOrderDataManager:
    def __init__(self):
        # 기본 태그 정의
        self.tag_types = {
            'MENU': '메뉴',
            'TEMPERATURE': '온도',
            'QUANTITY': '수량',
            'OPTION': '옵션'
        }
        
        # 데이터 검증을 위한 유효한 값들
        self.valid_values = {
            'MENU': ['아메리카노', '카페라떼', '카푸치노', '에스프레소'],
            'TEMPERATURE': ['아이스', '차가운', '따뜻한', '뜨거운', '핫'],
            'QUANTITY': ['한잔', '두잔', '세잔', '네잔', '다섯잔', '1잔', '2잔', '3잔', '4잔', '5잔'],
            'OPTION': ['설탕', '시럽']
        }

    def prepare_training_data(self) -> Tuple[List[str], List[List[str]]]:
        """기본 학습 데이터 준비"""
        texts = [
            "아이스 아메리카노 한잔 주세요",
            "따뜻한 카페라떼 2잔이요",
            "에스프레소 3잔 설탕 추가해주세요",
            "아메리카노 차가운걸로 주세요",
            "카푸치노 두잔 뜨겁게 해주세요",
            "차가운 아메리카노 3잔이요",
            "라떼 따뜻하게 해주세요",
            "에스프레소 5잔 시럽 추가요",
            "아이스 카페라떼 한잔만 주세요",
            "뜨거운 아메리카노 4잔 부탁드려요"
        ]
        
        tags = [
            ['B-TEMPERATURE', 'B-MENU', 'B-QUANTITY', 'O'],
            ['B-TEMPERATURE', 'B-MENU', 'B-QUANTITY'],
            ['B-MENU', 'B-QUANTITY', 'B-OPTION', 'O'],
            ['B-MENU', 'B-TEMPERATURE', 'O'],
            ['B-MENU', 'B-QUANTITY', 'B-TEMPERATURE', 'O'],
            ['B-TEMPERATURE', 'B-MENU', 'B-QUANTITY'],
            ['B-MENU', 'B-TEMPERATURE', 'O'],
            ['B-MENU', 'B-QUANTITY', 'B-OPTION', 'O'],
            ['B-TEMPERATURE', 'B-MENU', 'B-QUANTITY', 'O'],
            ['B-TEMPERATURE', 'B-MENU', 'B-QUANTITY', 'O']
        ]
        
        return texts, tags

    def save_data(self, texts: List[str], tags: List[List[str]], file_path: str):
        """학습 데이터를 JSON 형식으로 저장"""
        data = {
            'texts': texts,
            'tags': tags
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_data(self, file_path: str) -> Tuple[List[str], List[List[str]]]:
        """저장된 학습 데이터 불러오기"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['texts'], data['tags']

    def validate_data(self, texts: List[str], tags: List[List[str]]) -> bool:
        """데이터 유효성 검사"""
        if len(texts) != len(tags):
            print("Error: texts와 tags의 길이가 일치하지 않습니다.")
            return False
            
        for i, (text, tag_sequence) in enumerate(zip(texts, tags)):
            # 텍스트와 태그 개수 확인
            text_tokens = text.split()
            if len(text_tokens) < len(tag_sequence):
                print(f"Error: 인덱스 {i}의 태그 수가 텍스트 토큰 수보다 많습니다.")
                return False
            
            # 태그 형식 확인
            for tag in tag_sequence:
                if tag != 'O' and not (tag.startswith('B-') or tag.startswith('I-')):
                    print(f"Error: 인덱스 {i}의 잘못된 태그 형식: {tag}")
                    return False
                    
                if tag != 'O':
                    tag_type = tag[2:]  # B- 또는 I- 제거
                    if tag_type not in self.tag_types:
                        print(f"Error: 인덱스 {i}의 알 수 없는 태그 타입: {tag_type}")
                        return False
        
        return True

    def add_training_data(self, text: str, tags: List[str], file_path: str):
        """새로운 학습 데이터 추가"""
        if Path(file_path).exists():
            texts, existing_tags = self.load_data(file_path)
        else:
            texts, existing_tags = self.prepare_training_data()
        
        texts.append(text)
        existing_tags.append(tags)
        
        if self.validate_data(texts, existing_tags):
            self.save_data(texts, existing_tags, file_path)
            print("새로운 데이터가 추가되었습니다.")
        else:
            print("데이터 추가 실패: 유효성 검사 오류")

def get_training_data() -> Tuple[List[str], List[List[str]]]:
    """학습 데이터 가져오기 편의 함수"""
    manager = CoffeeOrderDataManager()
    return manager.prepare_training_data()

# 사용 예시
if __name__ == "__main__":
    manager = CoffeeOrderDataManager()
    
    # 기본 데이터 준비
    texts, tags = manager.prepare_training_data()
    
    # 데이터 저장
    manager.save_data(texts, tags, 'coffee_order_data.json')
    
    # 새로운 데이터 추가
    new_text = "아이스 카푸치노 2잔 시럽 추가해주세요"
    new_tags = ['B-TEMPERATURE', 'B-MENU', 'B-QUANTITY', 'B-OPTION', 'O', 'O']
    manager.add_training_data(new_text, new_tags, 'coffee_order_data.json')
    
    # 데이터 불러오기 및 검증
    loaded_texts, loaded_tags = manager.load_data('coffee_order_data.json')
    is_valid = manager.validate_data(loaded_texts, loaded_tags)
    print(f"데이터 유효성: {is_valid}")
