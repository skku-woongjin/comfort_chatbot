import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model=SentenceTransformer('jhgan/ko-sroberta-multitask')


df=pd.read_csv("wellness_data.csv")
df.head()
df=df.drop(columns=['Unnamed: 3'])
df=df[~df['챗봇'].isna()]
df['embedding']=pd.Series([[]]*len(df))

df['embedding']=df['유저'].map(lambda x:list(model.encode(x)))

#데이터셋과 질문 사이 cosine similarity 구하기(검색 기반으로 할지 문장 생성으로 할지 판단하는 기준)
def check_similarity(question):
    check={}
    embedding=model.encode(question)
    df['distance']=df['embedding'].map(lambda x:cosine_similarity([embedding],[x]).squeeze())
    df.head()
    answer=df.loc[df['distance'].idxmax()]
    print("similarity: {}".format(answer['distance']))
    print("검색 기반 문장:",answer['챗봇'])
    check['answer']=answer['챗봇']
    if answer['distance']>0.65:
        
        print("Chatbot > {}".format(answer['챗봇']))
        check['kogpt']=False
        check['similarity']=answer['distance']
    else:
        
        
        print("KoGPT2로 문장생성")  
        check['kogpt']=True
        check['similarity']=answer['distance']
    return check

       
#문장생성 기반 답 구하기
def get_string(question):
    embedding=model.encode(question)
    df['distance']=df['embedding'].map(lambda x:cosine_similarity([embedding],[x]).squeeze())
    df.head()
    answer=df.loc[df['distance'].idxmax()]
    return answer['챗봇']