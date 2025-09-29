import streamlit as st
import requests
import json
import os
from typing import Dict, Any, Optional
import tempfile


FASTAPI_BASE_URL = "http://localhost:8005"


def initialize_session_state():
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {}
    if 'temp_claim_info' not in st.session_state:
        st.session_state.temp_claim_info = {}
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'available_users' not in st.session_state:
        st.session_state.available_users = []
    if 'insurance_types' not in st.session_state:
        st.session_state.insurance_types = ["생명보험", "손해보험", "자동차보험"]
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "openai"


def load_users():
    """백엔드에서 사용자 목록을 가져옴"""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/users")
        if response.status_code == 200:
            st.session_state.available_users = response.json()
        else:
            st.error("사용자 목록을 불러오는데 실패했습니다.")
    except requests.RequestException:
        st.error("서버에 연결할 수 없습니다.")


def render_user_selection():
    """더미 사용자 선택 화면"""
    st.header("🔐 사용자 선택")
    
    if not st.session_state.available_users:
        load_users()
    
    if st.session_state.available_users:
        user_names = [user['name'] for user in st.session_state.available_users]
        
        selected_name = st.selectbox(
            "사용자를 선택하세요",
            options=user_names,
            index=0 if not st.session_state.selected_user else user_names.index(st.session_state.selected_user['name'])
        )
        
        if st.button("사용자 확인", type="primary"):
            selected_user = next(user for user in st.session_state.available_users if user['name'] == selected_name)
            st.session_state.selected_user = selected_user
            
            # 기본 사용자 정보 설정
            st.session_state.user_info = {
                'name': selected_user['name'],
                'age': selected_user['age'],
                'gender': selected_user['gender'],
                'medical_history': selected_user['medical_history']
            }
            
            st.success(f"{selected_name}님으로 로그인되었습니다.")
            st.rerun()
    else:
        st.warning("사용 가능한 사용자가 없습니다.")


def get_insurance_list(insurance_type, user_id=None):
    """특정 보험 종류의 업로드된 약관 목록 조회"""
    try:
        params = {"user_id": user_id} if user_id else {}
        response = requests.get(f"{FASTAPI_BASE_URL}/insurances/{insurance_type}", params=params)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def render_user_input_form():
    if not st.session_state.selected_user:
        st.warning("먼저 사용자를 선택해주세요.")
        return
        
    st.header("📋 보험 청구 심사 정보 입력")
    
    # 현재 로그인된 사용자 정보 표시
    st.info(f"현재 사용자: {st.session_state.selected_user['name']}")
    
    with st.form("claim_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # 기본 정보 (읽기 전용)
            st.text_input("성명", value=st.session_state.user_info.get('name', ''), disabled=True)
            st.number_input("나이", value=st.session_state.user_info.get('age', 0), disabled=True)
            st.text_input("성별", value=st.session_state.user_info.get('gender', ''), disabled=True)
            
        with col2:
            # 청구 정보 (임시 저장)
            insurance_type = st.selectbox(
                "보험 종류", 
                st.session_state.insurance_types,
                index=st.session_state.insurance_types.index(st.session_state.temp_claim_info.get('insurance_type', '생명보험'))
            )
            
            claim_amount = st.number_input(
                "청구 금액 (원)", 
                min_value=0, 
                value=st.session_state.temp_claim_info.get('claim_amount', 0)
            )
            
            claim_type = st.selectbox(
                "청구 유형", 
                ["질병", "상해", "사고", "입원", "수술", "기타"], 
                index=["질병", "상해", "사고", "입원", "수술", "기타"].index(st.session_state.temp_claim_info.get('claim_type', '질병'))
            )
            
        
        # 기존 병력 (읽기 전용)
        st.text_area("기존 병력", value=st.session_state.user_info.get('medical_history', ''), disabled=True)
        
        # 현재 상황 (예시글 포함)
        current_condition = st.text_area(
            "현재 상황 설명", 
            value=st.session_state.temp_claim_info.get('current_condition', ''),
            placeholder="예시: 2024년 1월 15일 교통사고로 인한 골절상 발생, 3일간 입원 치료 후 현재 통원 치료 중입니다. 의료비는 총 150만원이 소요되었으며, 향후 물리치료가 필요한 상황입니다."
        )
        
        additional_info = st.text_area(
            "추가 정보", 
            value=st.session_state.temp_claim_info.get('additional_info', ''),
            placeholder="추가로 전달하고 싶은 정보가 있다면 입력해주세요."
        )
        
        submitted = st.form_submit_button("청구 정보 저장")
        
        if submitted:
            # 사용자별 보험 종류에 등록된 약관이 있는지 확인
            user_id = st.session_state.selected_user['id']
            insurance_data = get_insurance_list(insurance_type, user_id)
            if not insurance_data or insurance_data['count'] == 0:
                st.error(f"⚠️ 현재 사용자({st.session_state.selected_user['name']})의 {insurance_type}에 등록된 약관이 없습니다! 먼저 보험 약관을 업로드해주세요.")
                return
            
            # 임시 청구 정보 저장 (캐시)
            st.session_state.temp_claim_info = {
                'insurance_type': insurance_type,
                'claim_amount': claim_amount,
                'claim_type': claim_type,
                'current_condition': current_condition,
                'additional_info': additional_info
            }
            st.success("청구 정보가 저장되었습니다.")
            
            # 등록된 약관 표시
            st.info(f"✅ {insurance_type}에 등록된 약관: {insurance_data['count']}개")
            for insurance in insurance_data['insurances']:
                st.write(f"• {insurance['name']}")


def create_search_query(user_info: Dict[str, Any]) -> str:
    query_parts = []
    
    if user_info.get('insurance_type'):
        query_parts.append(f"{user_info['insurance_type']}")
    
    if user_info.get('claim_type'):
        query_parts.append(f"{user_info['claim_type']}")
    
    if user_info.get('age'):
        if user_info['age'] < 20:
            query_parts.append("미성년자")
        elif user_info['age'] >= 65:
            query_parts.append("고령자")
    
    if user_info.get('medical_history'):
        query_parts.append(f"기존병력 {user_info['medical_history']}")
    
    if user_info.get('current_condition'):
        query_parts.append(user_info['current_condition'])
    
    return " ".join(query_parts)


def perform_similarity_search(rag_model="openai"):
    if not st.session_state.selected_user or not st.session_state.temp_claim_info:
        st.error("먼저 사용자를 선택하고 청구 정보를 입력해주세요.")
        return
    
    insurance_type = st.session_state.temp_claim_info.get('insurance_type')
    if not insurance_type:
        st.error("보험 종류가 선택되지 않았습니다.")
        return
    
    # 사용자별 해당 보험 종류의 약관이 있는지 확인
    user_id = st.session_state.selected_user['id']
    insurance_data = get_insurance_list(insurance_type, user_id)
    if not insurance_data or insurance_data['count'] == 0:
        st.error(f"⚠️ {st.session_state.selected_user['name']}님의 {insurance_type}에 등록된 약관이 없습니다! 먼저 보험 약관을 업로드해주세요.")
        return
    
    # 업로드된 문서의 임베딩 모델 자동 감지
    embedding_model = "openai"  # 기본값
    if insurance_data.get('insurances'):
        # 첫 번째 업로드된 문서의 임베딩 모델 사용
        first_insurance = insurance_data['insurances'][0]
        if 'embedding_model' in first_insurance and first_insurance['embedding_model']:
            embedding_model = first_insurance['embedding_model']
        
        # 여러 문서가 있고 임베딩 모델이 다른 경우 경고
        embedding_models = set()
        for ins in insurance_data['insurances']:
            if ins.get('embedding_model'):
                embedding_models.add(ins['embedding_model'])
        
        if len(embedding_models) > 1:
            st.warning(f"⚠️ 여러 임베딩 모델이 혼재되어 있습니다: {', '.join(embedding_models)}. 첫 번째 문서의 모델({embedding_model})을 사용합니다.")
    
    # 검색 쿼리 생성 (사용자 정보 + 청구 정보 결합)
    combined_info = {**st.session_state.user_info, **st.session_state.temp_claim_info}
    search_query = create_search_query(combined_info)
    
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/search-similar",
            json={
                "prompt": search_query,
                "insurance_type": insurance_type,
                "user_id": user_id,
                "embedding_model": embedding_model,
                "rag_model": rag_model,
                "top_k": 3
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.search_results = result
            st.session_state.search_results['insurance_type'] = insurance_type
            st.session_state.search_results['insurance_data'] = insurance_data
            st.session_state.search_results['rag_model'] = rag_model
            st.success(f"유사도 검색이 완료되었습니다. (AI 모델: {rag_model.upper()})")
        else:
            error_detail = response.json().get('detail', response.text) if response.headers.get('content-type', '').startswith('application/json') else response.text
            st.error(f"검색 실패: {error_detail}")
            
    except requests.RequestException as e:
        st.error(f"API 연결 오류: {str(e)}")


def display_search_results():
    if st.session_state.search_results:
        st.header("🔍 보험 청구 심사 결과")
        
        # 한줄 요약 표시
        if st.session_state.search_results.get('summary'):
            summary = st.session_state.search_results['summary']
            if "확률 높음" in summary:
                st.success(f"✅ {summary}")
            elif "확률 낮음" in summary:
                st.error(f"❌ {summary}")
            else:
                st.warning(f"⚠️ {summary}")
        
        # 검색된 보험 종류 정보
        insurance_type = st.session_state.search_results.get('insurance_type', 'N/A')
        insurance_data = st.session_state.search_results.get('insurance_data', {})
        rag_model = st.session_state.search_results.get('rag_model', 'openai')
        
        # 실제 사용된 임베딩 모델 표시
        embedding_model = "자동감지"
        if insurance_data.get('insurances'):
            first_insurance = insurance_data['insurances'][0]
            if 'embedding_model' in first_insurance and first_insurance['embedding_model']:
                embedding_model = first_insurance['embedding_model'].upper()
        
        st.info(f"📋 검색 대상: {insurance_type} ({embedding_model} 임베딩, 등록된 약관 {insurance_data.get('count', 0)}개)")
        st.info(f"🤖 사용된 AI 모델: {rag_model.upper()} {'(로컬)' if rag_model == 'qwen' else '(클라우드)'}")
        
        # 등록된 약관 목록 표시
        if insurance_data.get('insurances'):
            with st.expander(f"{insurance_type} 등록 약관 목록"):
                for insurance in insurance_data['insurances']:
                    st.write(f"• {insurance['name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔎 검색 쿼리")
            combined_info = {**st.session_state.user_info, **st.session_state.temp_claim_info}
            search_query = create_search_query(combined_info)
            st.text(search_query)
            
            st.subheader("📄 관련 약관 및 AI 심사 의견")
            st.text_area(
                "심사 결과", 
                value=st.session_state.search_results['response'], 
                height=400, 
                disabled=True
            )
            
            # PDF 제목 표시 (있는 경우)
            if st.session_state.search_results.get('pdf_titles'):
                st.subheader("📚 참고 문서")
                for title in st.session_state.search_results['pdf_titles']:
                    st.write(f"• {title}")
        
        with col2:
            st.subheader("👤 청구자 정보")
            
            # 기본 정보 카드
            with st.container():
                st.markdown("**📋 기본 정보**")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("성명", st.session_state.user_info.get('name', 'N/A'))
                    st.metric("나이", f"{st.session_state.user_info.get('age', 0)}세")
                with col_info2:
                    st.metric("성별", st.session_state.user_info.get('gender', 'N/A'))
                    
            # 청구 정보 카드
            st.markdown("---")
            st.markdown("**💰 청구 정보**")
            col_claim1, col_claim2 = st.columns(2)
            with col_claim1:
                st.metric("보험 종류", insurance_type)
                st.metric("청구 유형", st.session_state.temp_claim_info.get('claim_type', 'N/A'))
            with col_claim2:
                claim_amount = st.session_state.temp_claim_info.get('claim_amount', 0)
                st.metric("청구 금액", f"{claim_amount:,}원")
                
            # 의료 정보 카드
            st.markdown("---")
            st.markdown("**🏥 의료 정보**")
            medical_history = st.session_state.user_info.get('medical_history', 'N/A')
            if len(medical_history) > 30:
                st.text_area("기존 병력", value=medical_history, height=60, disabled=True, key="medical_sidebar")
            else:
                st.info(f"기존 병력: {medical_history}")
            
            # 현재 상황
            current_condition = st.session_state.temp_claim_info.get('current_condition', '')
            if current_condition:
                st.markdown("---")
                st.markdown("**📝 현재 상황**")
                st.text_area("", value=current_condition, height=80, disabled=True, key="condition_sidebar")


def upload_insurance_documents():
    st.header("📄 보험 업로드")
    
    # 사용자 선택 확인
    if not st.session_state.selected_user:
        st.warning("먼저 사용자를 선택해주세요.")
        return
    
    # 현재 로그인된 사용자 표시
    st.info(f"현재 사용자: {st.session_state.selected_user['name']}")
    
    # 보험 종류 선택
    insurance_type = st.selectbox(
        "보험 종류를 선택하세요",
        st.session_state.insurance_types,
        help="업로드할 PDF가 속하는 보험 종류를 선택해주세요."
    )
    
    # 임베딩 모델 선택
    embedding_model = st.selectbox(
        "임베딩 모델을 선택하세요",
        ["openai", "bge-m3"],
        index=0,
        help="벡터화에 사용할 임베딩 모델을 선택해주세요."
    )
    
    # 임베딩 모델 설명
    if embedding_model == "openai":
        st.info("🤖 **OpenAI Embeddings**: 높은 정확도, API 키 필요, 유료")
    else:
        st.info("🚀 **BGE-M3**: 무료 오픈소스, 로컬 실행, torch 설치 필요")
    
    uploaded_files = st.file_uploader(
        "보험 약관 PDF 파일을 업로드하세요",
        type=['pdf'],
        accept_multiple_files=True,
        help=f"선택한 {insurance_type} 카테고리에 저장됩니다."
    )
    
    if uploaded_files:
        st.info(f"업로드 대상: {insurance_type} ({embedding_model} 임베딩)")
        
        if st.button("보험 약관 업로드 시작", type="primary"):
            progress_bar = st.progress(0)
            user_id = st.session_state.selected_user['id']
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # 멀티파트 폼 데이터로 전송
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {
                        "insurance_type": insurance_type, 
                        "user_id": user_id,
                        "embedding_model": embedding_model
                    }
                    
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/upload-insurance",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {uploaded_file.name} ({insurance_type}) - {result['chunks']}개 청크로 처리 완료")
                    else:
                        error_detail = response.json().get('detail', response.text)
                        st.error(f"❌ {uploaded_file.name} 업로드 실패: {error_detail}")
                        
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name} 처리 중 오류: {str(e)}")
                
                # 진행률 업데이트
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success("모든 파일 업로드가 완료되었습니다!")
    
    # 현재 사용자의 업로드된 약관 목록 표시
    st.subheader(f"📊 {st.session_state.selected_user['name']}님의 업로드된 약관 현황")
    user_id = st.session_state.selected_user['id']
    for ins_type in st.session_state.insurance_types:
        insurance_data = get_insurance_list(ins_type, user_id)
        if insurance_data and insurance_data['count'] > 0:
            with st.expander(f"{ins_type} ({insurance_data['count']}개)"):
                for insurance in insurance_data['insurances']:
                    st.write(f"• {insurance['name']} (등록일: {insurance['created_at'][:10]})")
        else:
            st.write(f"• {ins_type}: 등록된 약관 없음")


def check_server_status(user_id=None):
    try:
        params = {"user_id": user_id} if user_id else {}
        response = requests.get(f"{FASTAPI_BASE_URL}/debug-stats", params=params)
        if response.status_code == 200:
            stats = response.json()
            return True, stats
        return False, None
    except:
        return False, None


def main():
    st.set_page_config(
        page_title="보험 청구 심사 자동화 시스템",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 보험 청구 심사 자동화 시스템")
    
    initialize_session_state()
    
    # 서버 상태 확인
    user_id = st.session_state.selected_user['id'] if st.session_state.selected_user else None
    server_status, stats = check_server_status(user_id)
    if not server_status:
        st.error("⚠️ FastAPI 서버가 실행되지 않았습니다. 먼저 서버를 시작해주세요.")
        st.code("cd insurance_claim_system && python main.py")
        return
    
    # 사이드바
    with st.sidebar:
        st.header("시스템 상태")
        if stats and st.session_state.selected_user:
            st.write(f"**{st.session_state.selected_user['name']}님의 보험 현황**")
            # 사용자별 보험 종류별 통계 표시
            if 'insurance_types' in stats:
                for ins_type, data in stats['insurance_types'].items():
                    st.write(f"{ins_type}: {data.get('docs', 0)}개")
        elif stats:
            st.write("**전체 시스템 현황**")
            if 'insurance_types' in stats:
                for ins_type, data in stats['insurance_types'].items():
                    st.write(f"{ins_type}: {data.get('docs', 0)}개")
        
        # 현재 로그인 사용자 표시
        if st.session_state.selected_user:
            st.success(f"로그인: {st.session_state.selected_user['name']}")
            if st.button("사용자 변경"):
                st.session_state.selected_user = None
                st.session_state.user_info = {}
                st.session_state.temp_claim_info = {}
                st.session_state.search_results = None
                st.rerun()
        
        st.header("메뉴")
        menu_options = ["사용자 선택", "청구 정보 입력", "보험 업로드", "심사 실행"]
        menu = st.radio("선택하세요", menu_options)
    
    # 메인 콘텐츠
    if menu == "사용자 선택":
        render_user_selection()
        
    elif menu == "청구 정보 입력":
        render_user_input_form()
        
    elif menu == "보험 업로드":
        upload_insurance_documents()
        
    elif menu == "심사 실행":
        if not st.session_state.selected_user:
            st.warning("먼저 사용자를 선택해주세요.")
        elif not st.session_state.temp_claim_info:
            st.warning("먼저 청구 정보를 입력해주세요.")
        else:
            # 선택된 보험 종류의 약관 확인 (사용자별)
            insurance_type = st.session_state.temp_claim_info.get('insurance_type')
            user_id = st.session_state.selected_user['id']
            if insurance_type:
                insurance_data = get_insurance_list(insurance_type, user_id)
                if insurance_data and insurance_data['count'] > 0:
                    st.info(f"📋 {st.session_state.selected_user['name']}님의 {insurance_type} 약관 {insurance_data['count']}개가 등록되어 있습니다.")
                    
                    # RAG 모델 선택
                    rag_model = st.selectbox(
                        "🤖 심사에 사용할 AI 모델",
                        ["openai", "qwen"],
                        index=0,
                        help="OpenAI: 고성능 클라우드 모델 (API 비용), Qwen: 무료 로컬 모델"
                    )
                    
                    if rag_model == "openai":
                        st.info("🌐 **OpenAI GPT-4o**: 높은 품질의 심사 분석, API 비용 발생")
                    else:
                        st.info("🏠 **Qwen 로컬**: 완전 무료, 로컬 실행, 기본적인 심사 분석")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button("청구 심사 실행", type="primary"):
                            perform_similarity_search(rag_model)
                    
                    with col2:
                        if st.button("결과 초기화"):
                            st.session_state.search_results = None
                            st.rerun()
                    
                    display_search_results()
                else:
                    st.error(f"⚠️ {st.session_state.selected_user['name']}님의 {insurance_type}에 등록된 약관이 없습니다!")
                    st.info("'보험 업로드' 메뉴에서 먼저 약관을 업로드해주세요.")
            else:
                st.warning("보험 종류가 선택되지 않았습니다.")


if __name__ == "__main__":
    main()