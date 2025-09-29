import os
import re
import json
from typing import Optional, List, Dict, Any
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings  # BGE 사용
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
# from huggingface_hub import login  # BGE 사용시 주석 해제
from pypdf import PdfReader


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# torch 보안 경고 우회
os.environ.setdefault("PYTORCH_DISABLE_WARN_LOAD", "1")

# API 키 설정 (환경 변수에서 읽기)
openai_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
# if hf_token:
#     login(hf_token)  # BGE 사용시 주석 해제

# SQLite 데이터베이스 설정
DATABASE_URL = "sqlite:///./insurance_system.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 데이터베이스 모델들
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    age = Column(Integer)
    gender = Column(String)
    medical_history = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Insurance(Base):
    __tablename__ = "insurances"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)  # PDF 파일명
    type = Column(String, index=True)  # 생명/손해/자동차
    file_path = Column(String)         # 저장된 파일 경로
    user_id = Column(Integer, ForeignKey("users.id"), index=True)  # 사용자별 보험 분리
    embedding_model = Column(String, default="openai", index=True)  # 사용된 임베딩 모델
    created_at = Column(DateTime, default=datetime.utcnow)

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# 데이터베이스 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="보험 청구 심사 자동화 시스템 API",
    description="보험 약관 PDF 기반 유사도 검색 및 심사 지원",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# BGE-M3 임베딩 클래스 (safetensors 사용, torch 보안 우회)
class LazyBgeEmbeddings(Embeddings):
    """BAAI/bge-m3 임베딩 모델 사용을 위한 지연 로딩 클래스 (safetensors)"""
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", normalize: bool = True):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._tokenizer = None
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        self.passage_instruction = "Represent this passage for retrieval: "

    def _ensure_model(self):
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModel
                
                # safetensors 사용으로 torch 보안 문제 우회
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    trust_remote_code=True
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model.eval()
                
                print(f"✅ BGE-M3 모델이 safetensors로 성공적으로 로드되었습니다.")
                
            except Exception as e:
                print(f"⚠️ BGE-M3 로드 실패: {e}")
                raise RuntimeError(f"BGE-M3 모델 로드 실패: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        inst_texts = [f"{self.passage_instruction}{t}" for t in texts]
        
        import torch
        embeddings = []
        
        for text in inst_texts:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # 평균 풀링으로 임베딩 생성
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                
                if self.normalize:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                
                embeddings.append(embedding.tolist())
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        self._ensure_model()
        q = f"{self.query_instruction}{text}"
        
        import torch
        inputs = self._tokenizer(q, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            if self.normalize:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            
            return embedding.tolist()


# Qwen 로컬 LLM 클래스
class LazyQwenLLM:
    """Qwen 로컬 LLM 사용을 위한 지연 로딩 클래스"""
    def __init__(self, model_name: str = "Qwen/Qwen1.5-0.5B-Chat"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True, 
                    torch_dtype=torch.float32
                ).to("cpu")
                
                # 생성 설정 - 참고 파일과 동일하게
                gc = self._model.generation_config
                gc.do_sample = False
                gc.temperature = None
                gc.top_p = None
                gc.top_k = None
                gc.typical_p = None
                gc.num_beams = 1
                gc.use_cache = True
                self._model.generation_config = gc
                
                print(f"✅ Qwen 모델이 성공적으로 로드되었습니다.")
                
            except Exception as e:
                print(f"⚠️ Qwen 로드 실패: {e}")
                raise RuntimeError(f"Qwen 모델 로드 실패: {e}")

    def generate_answer(self, prompt: str, max_input_tokens: int = 1536, max_new_tokens: int = 128) -> str:
        self._ensure_model()
        
        import torch
        enc = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
        input_ids = enc.input_ids.to(self._model.device)
        attention_mask = enc.attention_mask.to(self._model.device)
        
        input_length = input_ids.shape[1]
        
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,  # greedy decoding
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 전체 응답을 디코딩
        full_response = self._tokenizer.decode(out[0], skip_special_tokens=True)
        
        # 입력 프롬프트 부분을 제거하여 생성된 답변만 추출
        original_prompt = self._tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        if full_response.startswith(original_prompt):
            response = full_response[len(original_prompt):].strip()
        else:
            response = full_response.strip()
        
        # 응답이 없거나 의미없는 경우 기본값 반환
        if not response or len(response) < 2:
            response = "보류"
        
        return response


_BGE_LAZY: Optional[LazyBgeEmbeddings] = None
_QWEN_LAZY: Optional[LazyQwenLLM] = None

def get_bge_embeddings() -> LazyBgeEmbeddings:
    global _BGE_LAZY
    if _BGE_LAZY is None:
        _BGE_LAZY = LazyBgeEmbeddings(model_name="BAAI/bge-m3", device="cpu", normalize=True)
    return _BGE_LAZY

def get_qwen_llm() -> LazyQwenLLM:
    global _QWEN_LAZY
    if _QWEN_LAZY is None:
        _QWEN_LAZY = LazyQwenLLM(model_name="Qwen/Qwen1.5-0.5B-Chat")
    return _QWEN_LAZY

def get_embeddings(model_type: str = "openai"):
    """임베딩 모델 선택"""
    if model_type == "openai":
        if openai_key:
            return OpenAIEmbeddings()
        else:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API 키가 필요합니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
            )
    elif model_type == "bge-m3":
        return get_bge_embeddings()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 임베딩 모델: {model_type}. 지원 모델: openai, bge-m3"
        )


# 보험 종류별 벡터 스토어 관리 (사용자별)
INSURANCE_TYPES = ["생명보험", "손해보험", "자동차보험"]
# VECTORSTORES[user_id][insurance_type] 구조
VECTORSTORES = {}


def _normalize_pdf_text(txt: str) -> str:
    if not txt:
        return ""
    # 리가처 처리
    lig_map = {
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl"
    }
    for k, v in lig_map.items():
        txt = txt.replace(k, v)
    
    # 하이픈으로 분리된 단어 결합
    txt = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", txt)
    # 개행을 공백으로 변환
    txt = re.sub(r"\s*\n\s*", " ", txt)
    # 다중 공백 축소
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    raw = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    return _normalize_pdf_text(raw)


def get_user_vectorstore(user_id: int, insurance_type: str, embedding_model: str = "openai"):
    """사용자별 보험 종류별 벡터스토어 가져오기"""
    cache_key = f"{user_id}_{insurance_type}_{embedding_model}"
    
    if user_id not in VECTORSTORES:
        VECTORSTORES[user_id] = {}
    
    if cache_key not in VECTORSTORES[user_id]:
        # 저장된 인덱스 로드 시도
        faiss_path = f"faiss_db/user_{user_id}_{insurance_type}_{embedding_model}"
        if Path(f"{faiss_path}/index.faiss").exists() and Path(f"{faiss_path}/index.pkl").exists():
            try:
                VECTORSTORES[user_id][cache_key] = FAISS.load_local(
                    faiss_path,
                    embeddings=get_embeddings(embedding_model),
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"❌ 사용자 {user_id} {insurance_type} ({embedding_model}) 벡터스토어 로드 실패: {e}")
                VECTORSTORES[user_id][cache_key] = None
        else:
            VECTORSTORES[user_id][cache_key] = None
    
    return VECTORSTORES[user_id][cache_key]


def save_user_vectorstore(user_id: int, insurance_type: str, vectorstore, embedding_model: str = "openai"):
    """사용자별 보험 종류별 벡터스토어 저장"""
    cache_key = f"{user_id}_{insurance_type}_{embedding_model}"
    save_path = f"faiss_db/user_{user_id}_{insurance_type}_{embedding_model}"
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    
    # 메모리에도 저장
    if user_id not in VECTORSTORES:
        VECTORSTORES[user_id] = {}
    VECTORSTORES[user_id][cache_key] = vectorstore


# 더미 사용자 초기화 함수
def init_dummy_users():
    db = SessionLocal()
    try:
        # 더미 사용자가 이미 존재하는지 확인
        existing_users = db.query(User).count()
        if existing_users == 0:
            # 더미 사용자 생성
            dummy_users = [
                User(name="하동우", age=28, gender="남성", medical_history="고혈압, 당뇨병 가족력"),
                User(name="최우진", age=32, gender="남성", medical_history="알레르기 체질, 천식"),
                User(name="안현욱", age=26, gender="남성", medical_history="특이사항 없음")
            ]
            
            for user in dummy_users:
                db.add(user)
            db.commit()
            print("✅ 더미 사용자 3명 생성 완료")
        else:
            print("ℹ️ 기존 사용자 데이터 확인됨")
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    # 더미 사용자 초기화
    init_dummy_users()
    # 사용자별 벡터 스토어는 필요시 동적 로드
    print("✅ 보험 청구 심사 시스템 시작 - 사용자별 벡터스토어 준비 완료")


class UserInfo(BaseModel):
    name: str
    age: int
    gender: str
    insurance_type: str
    claim_amount: int
    claim_type: str
    medical_history: Optional[str] = ""
    current_condition: Optional[str] = ""
    additional_info: Optional[str] = ""


class RAGRequest(BaseModel):
    prompt: str = Field(..., description="검색 쿼리")
    insurance_type: str = Field(..., description="보험 종류: 생명보험/손해보험/자동차보험")
    user_id: int = Field(..., description="사용자 ID")
    embedding_model: str = Field("openai", description="임베딩 모델: openai/bge-m3")
    rag_model: str = Field("openai", description="RAG 모델: openai/qwen")
    top_k: int = Field(3, ge=1, le=10, description="검색할 결과 수")


class SearchResponse(BaseModel):
    prompt: str
    response: str
    summary: Optional[str] = None  # 한줄 요약 추가
    pdf_titles: Optional[List[str]] = None

class UserResponse(BaseModel):
    id: int
    name: str
    age: int
    gender: str
    medical_history: str

class InsuranceUploadRequest(BaseModel):
    insurance_type: str = Field(..., description="보험 종류: 생명보험/손해보험/자동차보험")


# 사용자 관련 API
@app.get("/users", response_model=List[UserResponse], summary="더미 사용자 목록 조회")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@app.get("/users/{user_name}", response_model=UserResponse, summary="특정 사용자 정보 조회")
async def get_user(user_name: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.name == user_name).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return user

@app.get("/insurance-types", summary="지원하는 보험 종류 목록")
async def get_insurance_types():
    return {"insurance_types": INSURANCE_TYPES}

@app.get("/insurances/{insurance_type}", summary="특정 보험 종류의 업로드된 약관 목록")
async def get_insurances_by_type(insurance_type: str, user_id: int = None, db: Session = Depends(get_db)):
    if insurance_type not in INSURANCE_TYPES:
        raise HTTPException(status_code=400, detail="지원하지 않는 보험 종류입니다.")
    
    if user_id:
        # 특정 사용자의 보험만 조회
        insurances = db.query(Insurance).filter(
            Insurance.type == insurance_type,
            Insurance.user_id == user_id
        ).all()
    else:
        # 전체 조회 (기존 호환성)
        insurances = db.query(Insurance).filter(Insurance.type == insurance_type).all()
    
    return {
        "insurance_type": insurance_type,
        "count": len(insurances),
        "insurances": [{"id": ins.id, "name": ins.name, "embedding_model": ins.embedding_model, "created_at": ins.created_at} for ins in insurances]
    }


@app.post("/upload-insurance", summary="보험 종류별 약관 PDF 업로드")
async def upload_insurance(
    insurance_type: str = Body(...),
    user_id: int = Body(...),
    embedding_model: str = Body("openai"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 보험 종류 유효성 검사
    if insurance_type not in INSURANCE_TYPES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 보험 종류입니다. 가능한 종류: {INSURANCE_TYPES}")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 허용됩니다.")
    
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # PDF 텍스트 추출
        paper_text = load_pdf(tmp_path)
        if not paper_text or len(paper_text) < 50:
            raise HTTPException(
                status_code=400, 
                detail="PDF에서 텍스트를 추출하지 못했습니다. 스캔본이면 OCR 후 재업로드하세요."
            )

        # 텍스트 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = splitter.create_documents([paper_text])
        
        # 메타데이터에 보험 종류와 파일명 추가
        for doc in docs:
            doc.metadata["source"] = file.filename
            doc.metadata["insurance_type"] = insurance_type

        # 임베딩 모델 유효성 검사
        valid_models = ["openai", "bge-m3"]
        if embedding_model not in valid_models:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 임베딩 모델입니다. 가능한 모델: {valid_models}")

        # 사용자별 보험 종류별 벡터 스토어 생성/업데이트
        user_vectorstore = get_user_vectorstore(user_id, insurance_type, embedding_model)
        
        if user_vectorstore is None:
            # 새 벡터스토어 생성
            new_vectorstore = FAISS.from_documents(
                docs, 
                embedding=get_embeddings(embedding_model), 
                distance_strategy=DistanceStrategy.COSINE
            )
        else:
            # 기존 벡터스토어에 문서 추가
            new_docs_vectorstore = FAISS.from_documents(
                docs, 
                embedding=get_embeddings(embedding_model), 
                distance_strategy=DistanceStrategy.COSINE
            )
            user_vectorstore.merge_from(new_docs_vectorstore)
            new_vectorstore = user_vectorstore

        # 사용자별 보험 종류별 디렉토리에 저장
        save_user_vectorstore(user_id, insurance_type, new_vectorstore, embedding_model)

        # 데이터베이스에 보험 정보 저장
        save_path = f"faiss_db/user_{user_id}_{insurance_type}_{embedding_model}"
        insurance_record = Insurance(
            name=file.filename,
            type=insurance_type,
            file_path=save_path,
            user_id=user_id,
            embedding_model=embedding_model
        )
        db.add(insurance_record)
        db.commit()

        # 임시 파일 삭제
        os.unlink(tmp_path)

        return JSONResponse(content={
            "message": f"{insurance_type} - {file.filename}: {len(docs)}개의 청크로 벡터화 및 저장 완료 ({embedding_model} 임베딩)",
            "filename": file.filename,
            "insurance_type": insurance_type,
            "embedding_model": embedding_model,
            "chunks": len(docs)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"처리 오류: {str(e)}")


# 기존 업로드 API는 호환성을 위해 유지 (deprecated)
@app.post("/upload-paper", summary="PDF 업로드 (호환성용, deprecated)")
async def upload_pdf_deprecated(file: UploadFile = File(...)):
    # 기본적으로 생명보험으로 분류
    return await upload_insurance("생명보험", file, next(get_db()))


@app.post("/search-similar", response_model=SearchResponse, summary="보험 종류별 유사도 기반 약관 검색")
async def search_similar_terms(rag_request: RAGRequest = Body(...), db: Session = Depends(get_db)):
    # 보험 종류 유효성 검사
    if rag_request.insurance_type not in INSURANCE_TYPES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 보험 종류입니다. 가능한 종류: {INSURANCE_TYPES}")
    
    # 사용자별 해당 보험 종류의 벡터스토어 확인 (임베딩 모델별)
    vectorstore = get_user_vectorstore(rag_request.user_id, rag_request.insurance_type, rag_request.embedding_model)
    if vectorstore is None:
        # 업로드된 약관이 있는지 DB에서 확인 (사용자별, 임베딩 모델별)
        insurance_count = db.query(Insurance).filter(
            Insurance.type == rag_request.insurance_type,
            Insurance.user_id == rag_request.user_id,
            Insurance.embedding_model == rag_request.embedding_model
        ).count()
        if insurance_count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"사용자 {rag_request.user_id}의 {rag_request.insurance_type} ({rag_request.embedding_model} 임베딩) 종류 등록된 약관이 없습니다. 보험 약관을 먼저 업로드해주세요."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"사용자 {rag_request.user_id}의 {rag_request.insurance_type} ({rag_request.embedding_model}) 벡터스토어 로드에 실패했습니다."
            )
    
    try:
        # 유사도 검색 수행
        fetch_k = max(rag_request.top_k * 3, 10)
        
        # 기본 검색
        docs_with_scores = vectorstore.similarity_search_with_score(
            rag_request.prompt, 
            k=fetch_k
        )
        docs = [doc for doc, score in docs_with_scores]
        
        # MMR로 다양성 확보
        try:
            docs = vectorstore.max_marginal_relevance_search(
                rag_request.prompt, 
                k=min(len(docs), rag_request.top_k), 
                fetch_k=fetch_k
            )
        except Exception:
            docs = docs[:rag_request.top_k]
        
        if not docs:
            return SearchResponse(
                prompt=rag_request.prompt,
                response="관련 약관을 찾을 수 없습니다.",
                summary="심사 통과 확률 보통 - 관련 약관 검색 결과 없음"
            )
        
        # 컨텍스트 구성
        context_parts = []
        pdf_titles = set()
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get('source', 'Unknown')
            pdf_titles.add(source)
            
            context_parts.append(f"[문서 {i}: {source}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # RAG 모델에 따른 답변 생성
        if rag_request.rag_model == "qwen":
            # Qwen 로컬 모델 사용
            try:
                qwen_llm = get_qwen_llm()
                
                # Qwen에게 매우 간단한 심사 결과만 요청
                simple_prompt = f"""보험 청구: {rag_request.prompt[:100]}

약관 정보: {context[:300]}

위 정보로 심사 결과를 다음 중 하나로 답하세요:
승인, 보류, 거절

답변:"""
                
                # 간단한 심사 결과 생성
                result_response = qwen_llm.generate_answer(simple_prompt, max_input_tokens=800, max_new_tokens=50)
                
                print(f"🔍 Qwen 심사결과 응답: '{result_response}'")
                
                # 심사 결과 추출
                if "승인" in result_response:
                    decision = "승인"
                    reason = "관련 약관에서 보장 조건을 충족하는 것으로 판단됩니다."
                elif "거절" in result_response:
                    decision = "거절"
                    reason = "약관상 보장 제외 조건에 해당하거나 증빙이 부족합니다."
                else:
                    decision = "보류"
                    reason = "추가 서류 검토 및 전문가 심사가 필요합니다."
                
                # 간단한 형식으로 응답 구성
                answer = f"""🔍 **심사 결과**: {decision}

📝 **심사 의견**: {reason}

📋 **참고 약관**:
{context[:400]}...

※ 이는 Qwen 로컬 모델의 1차 심사 결과이며, 정확한 심사를 위해서는 전문가 검토가 필요합니다."""
                
                # 심사 결과에 따른 요약 생성
                if decision == "승인":
                    summary = "심사 통과 확률 높음 - 보장 조건 충족으로 승인 가능"
                elif decision == "거절":
                    summary = "심사 통과 확률 낮음 - 약관상 보장 제외 조건 해당"
                else:
                    summary = "심사 통과 확률 보통 - 추가 서류 및 전문가 검토 필요"
                
            except Exception as e:
                answer = f"""
관련 약관 내용을 찾았습니다:

{context[:1000]}...

※ Qwen 모델 호출 실패로 기본 응답을 제공합니다. ({str(e)})
상세한 심사를 위해 전문가의 검토가 필요합니다.
"""
                summary = "심사 통과 확률 보통 - Qwen 분석 실패로 전문가 검토 필요"
                
        elif rag_request.rag_model == "openai" and openai_key:
            # OpenAI 모델 사용
            llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
            
            full_prompt = f"""
다음은 보험 약관에서 추출한 관련 내용들입니다. 
이 내용을 바탕으로 사용자의 보험 청구 상황에 대한 심사 의견을 제공해주세요.

관련 약관 내용:
{context}

사용자 질의: {rag_request.prompt}

답변 형식:
1. 관련 약관 요약
2. 청구 승인 가능성 
3. 주의사항이나 추가 필요 서류
4. 참고한 문서명

한국어로 명확하고 구체적으로 답변해주세요.
"""
            
            # 한줄 요약을 위한 별도 프롬프트
            summary_prompt = f"""
다음 보험 청구 상황을 분석하여 한 문장으로 승인 가능성을 요약해주세요.

청구 상황: {rag_request.prompt}
관련 약관: {context[:500]}

다음 중 하나의 형태로만 답변하세요:
- "심사 통과 확률 높음 - [주요 이유]"
- "심사 통과 확률 보통 - [주요 이유]" 
- "심사 통과 확률 낮음 - [주요 이유]"

한 문장으로 간결하게 답변하세요.
"""
            
            try:
                # 상세 답변 생성
                response = llm.invoke([HumanMessage(content=full_prompt)])
                answer = response.content.strip()
                
                # 한줄 요약 생성
                summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
                summary = summary_response.content.strip()
                
            except Exception as e:
                # OpenAI 호출 실패 시 기본 응답
                answer = f"""
관련 약관 내용을 찾았습니다:

{context[:1000]}...

※ AI 모델 호출 실패로 기본 응답을 제공합니다. ({str(e)})
상세한 심사를 위해 전문가의 검토가 필요합니다.
"""
                summary = "심사 통과 확률 보통 - AI 분석 실패로 전문가 검토 필요"
        else:
            # OpenAI 키가 없는 경우 기본 응답
            answer = f"""
관련 약관 내용:

{context[:1500]}

※ 상세한 AI 분석을 위해서는 OpenAI API 키 설정이나 Qwen 모델 선택이 필요합니다.
위 약관 내용을 참고하여 전문가의 검토를 받으시기 바랍니다.
"""
            summary = "심사 통과 확률 보통 - 전문가 검토 필요 (AI 분석 불가)"
        
        return SearchResponse(
            prompt=rag_request.prompt,
            response=answer,
            summary=summary,
            pdf_titles=list(pdf_titles)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")


@app.post("/rag-query", response_model=SearchResponse, summary="RAG 질의 (호환성)")
async def rag_query(rag_request: RAGRequest = Body(...)):
    """기존 RAG API와의 호환성을 위한 엔드포인트"""
    return await search_similar_terms(rag_request)


@app.get("/debug-stats", summary="사용자별 보험 종류별 상태 확인")
async def debug_stats(user_id: int = None, db: Session = Depends(get_db)):
    result = {
        "server_status": "running",
        "embeddings_model": "OpenAI",
        "insurance_types": {}
    }
    
    if user_id:
        # 특정 사용자의 보험 통계
        for insurance_type in INSURANCE_TYPES:
            # DB에서 파일 수 조회
            file_count = db.query(Insurance).filter(
                Insurance.type == insurance_type,
                Insurance.user_id == user_id
            ).count()
            
            # 벡터스토어 로드 상태 확인
            user_vectorstore = get_user_vectorstore(user_id, insurance_type)
            loaded = user_vectorstore is not None
            
            result["insurance_types"][insurance_type] = {
                "loaded": loaded,
                "docs": file_count  # 파일 수로 변경
            }
    else:
        # 전체 사용자의 보험 통계
        for insurance_type in INSURANCE_TYPES:
            total_count = db.query(Insurance).filter(Insurance.type == insurance_type).count()
            result["insurance_types"][insurance_type] = {
                "loaded": total_count > 0,
                "docs": total_count
            }
    
    return result


@app.get("/health", summary="서버 상태 확인")
async def health_check():
    return {"status": "healthy", "message": "보험 청구 심사 시스템 정상 작동 중"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)