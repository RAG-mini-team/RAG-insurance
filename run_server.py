#!/usr/bin/env python3

import subprocess
import sys
import os
import time
from pathlib import Path


def check_requirements():
    """필수 패키지 설치 여부 확인"""
    try:
        import streamlit
        import fastapi
        import langchain
        print("✅ 필수 패키지 확인 완료")
        return True
    except ImportError as e:
        print(f"❌ 필수 패키지 누락: {e}")
        print("다음 명령어로 설치하세요: pip install -r requirements.txt")
        return False


def check_env_file():
    """환경변수 파일 확인"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env 파일이 없습니다.")
        print("📋 .env.example을 참고하여 .env 파일을 생성하고 API 키를 설정하세요.")
        return False
    
    print("✅ .env 파일 확인 완료")
    return True


def start_fastapi_server():
    """FastAPI 서버 시작"""
    print("🚀 FastAPI 서버 시작 중...")
    try:
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 서버 시작 대기
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ FastAPI 서버가 포트 8004에서 실행 중입니다.")
            print("📍 API 문서: http://localhost:8004/docs")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ FastAPI 서버 시작 실패:")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ FastAPI 서버 시작 오류: {e}")
        return None


def start_streamlit_app():
    """Streamlit 앱 시작"""
    print("🎨 Streamlit 앱 시작 중...")
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py", 
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        time.sleep(2)
        
        if process.poll() is None:
            print("✅ Streamlit 앱이 실행되었습니다.")
            print("🌐 웹 인터페이스: http://localhost:8501")
            return process
        else:
            print("❌ Streamlit 앱 시작 실패")
            return None
            
    except Exception as e:
        print(f"❌ Streamlit 앱 시작 오류: {e}")
        return None


def main():
    print("=" * 60)
    print("🏥 보험 청구 심사 자동화 시스템")
    print("=" * 60)
    
    # 현재 디렉토리 확인
    if not Path("main.py").exists() or not Path("app.py").exists():
        print("❌ 잘못된 디렉토리입니다. insurance_claim_system 폴더에서 실행하세요.")
        return
    
    # 필수 패키지 확인
    if not check_requirements():
        return
    
    # 환경 변수 파일 확인 (선택사항)
    check_env_file()
    
    try:
        # FastAPI 서버 시작
        fastapi_process = start_fastapi_server()
        if not fastapi_process:
            print("FastAPI 서버 시작에 실패했습니다.")
            return
        
        # Streamlit 앱 시작
        streamlit_process = start_streamlit_app()
        if not streamlit_process:
            print("Streamlit 앱 시작에 실패했습니다.")
            if fastapi_process:
                fastapi_process.terminate()
            return
        
        print("\n" + "=" * 60)
        print("🎉 시스템이 성공적으로 시작되었습니다!")
        print("=" * 60)
        print("📍 FastAPI 서버: http://localhost:8005")
        print("📍 API 문서: http://localhost:8005/docs")
        print("🌐 웹 인터페이스: http://localhost:8501")
        print("=" * 60)
        print("종료하려면 Ctrl+C를 누르세요.")
        
        # 프로세스 대기
        try:
            fastapi_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 시스템 종료 중...")
            if streamlit_process and streamlit_process.poll() is None:
                streamlit_process.terminate()
            if fastapi_process and fastapi_process.poll() is None:
                fastapi_process.terminate()
            print("✅ 종료 완료")
            
    except Exception as e:
        print(f"❌ 실행 오류: {e}")


if __name__ == "__main__":
    main()