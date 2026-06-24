# text to json 
# 뉴스 데이터를 통으로 api 에 붙여 넣으면 json 에러가 나서 만든 헬퍼 스크립트입니다 ( 추후 수정 예정 )

import json

def make_swagger_json(text: str) -> str:
    return json.dumps(
        {"text": text},
        ensure_ascii=False,
        indent=2
    )

# 사용 예시
with open("article.txt", "r", encoding="utf-8") as f:
    article = f.read()

print(make_swagger_json(article))