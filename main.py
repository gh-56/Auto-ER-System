# FastAPI 관련 import
from fastapi import FastAPI, HTTPException, File, UploadFile
from typing import Optional, List
from pydantic import BaseModel
from pydub import AudioSegment
import emergency
import os

BASE_PATH = './'

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 입력 데이터 모델 정의
class EmergencyInput(BaseModel):
    text: str
    latitude: float
    longitude: float

# 출력 데이터 모델 정의 (필요한 정보만 포함하도록 수정)
class HospitalRecommendation(BaseModel):
    hospital_name: str
    address: str
    distance_km: float
    duration: str
    arrival_time: str
    main_number: str
    emergency_number: str

class ResponseData(BaseModel):
    summary: Optional[dict] = None
    emergency_class: Optional[int] = None
    nearest_hospitals: Optional[List[HospitalRecommendation]] = None
    message: Optional[str] = None # 응급상황이 아닐 경우를 대비


# /recommend_hospital GET 요청 처리
@app.get("/recommend_hospital/", response_model=ResponseData)
async def get_recommend_hospital(text: str, latitude: float, longitude: float):
    # 파일 업로드 대신 텍스트, 위도, 경도를 직접 받음
    client = emergency.init_openai()
    
    # 텍스트 요약 및 키워드 추출
    summary_result = emergency.summarize_text(client, text)
    
    if not summary_result:
        raise HTTPException(status_code=500, detail="텍스트 요약 실패")
    
    # 응급 모델 로드
    tokenizer, model, device = emergency.load_emergency_model()
    
    if not all([tokenizer, model, device]):
        raise HTTPException(status_code=500, detail="응급 모델 로드 실패")
    
    # 응급도 예측
    predicted_class, probabilities = emergency.predict_emergency(summary_result["keywords"], tokenizer, model, device)
    
    if predicted_class is None:
        raise HTTPException(status_code=500, detail="응급도 예측 실패")
    
    if(predicted_class+1 <= 3):
        # 병원 추천
        hospital_df = emergency.recommend_nearest_hospitals(latitude, longitude, emergency.get_hospital_data())
    
        if hospital_df.empty:
            raise HTTPException(status_code=404, detail="추천할 병원을 찾을 수 없습니다.")
    
        # 필요한 정보만 추출하여 딕셔너리 리스트로 변환
        result = {
            "summary": summary_result,
            "emergency_class": predicted_class + 1,
            "nearest_hospitals": hospital_df.to_dict(orient='records'),
            "message": "응급 상황입니다."
        }
    
        return result
    else:
        result = {
            "summary": {},  # 또는 None
            "emergency_class": predicted_class + 1, # 또는 None, 상황에 맞게
            "nearest_hospitals": [],  # 빈 리스트
            "message": "응급 상황이 아닙니다."
        }
        return result