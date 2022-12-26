# Talking Agent-위로봇(데모)
> 검색기반과 koGPT2를 기반으로 아이들을 위로해주는 위로봇의 답변을 보여주는 웹 애플리케이션입니다.
> 검색기반과 koGPT2를 이용한 답변과 **cosine similarity**를 기준으로 한 최종 답변을 보여줍니다.

## Framework
`Flask`, `Ajax`
## Getting Started
### Clone Repository
```
$ git clone https://github.com/peter0107/comfort_chatbot.git
```
### How to Run
**Installation:**
```
$ python -m pip install -U pip
$ pip install -r requirements.txt
```
만약 위의 내용대로 했는데 안되신다면 Python 버전의 문제일 수 있습니다. (제가 쓴 버전은 python 3.8입니다)

**To run Flask:**
```
$ python app.py
```
## 파일 구조
```
.
├── static/
│    ├── usuful.js
├── template/
│    ├── index.html
├── app.py
├── require.py
├── wellness.csv
├── wellness_data.csv
├── wellness_data_embed.csv
```
* 모든 api는 app.py에 정의되어 있음.
* useful.js는 Ajax를 사용해서 결과값을 api를 통해서 가져온뒤 html에 보내는 역할을 함
* require.py는 cosine similarity를 계산하는 역할을 함
* wellness.csv는 원초 데이터셋
  
  wellness_data.csv는 누락된 부분을 없앤 데이터셋, 
  
  wellness_data_embed.csv는 embedding값을 추가한 데이터셋
  
## API 설명
### 1) / [GET]
template/index.html을 렌더링함
### 2) /answer [POST]
사용자의 질문에 대해 KoGPT2을 이용해 응답함

검색기반과 문장생성 결과, cosine similarity, 최종응답(S=검색기반, G=문장생성)을 html에 보냄

**Request 예시**
(json)
```
{
    method: "post",
    url: "http://[서버주소]:[포트번호]/answer",
    params: {
        question: "시험 망쳤어..",
    },
    headers: {
        "Content-Type": "application/json",
    }
}
```

**Response 예시**
(text)
```
"Cosine similarity: 0.6687833
 검색 기반: 운이 나빴던 거라고 생각해요.
 kogpt로 문장생성: 정말 당황스러우셨겠어요. 하지만 미리 걱정하는 건 자신을 더 잘 이해할 수 있는 방법이에요.*(S) 운이 나빴던 거라고 생각해요."
```


