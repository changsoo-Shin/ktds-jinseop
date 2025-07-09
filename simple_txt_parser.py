#!/usr/bin/env python3
"""
간단한 TXT 파일 파싱 스크립트
"""

import re
from pathlib import Path

def parse_linux_master_txt():
    """리눅스마스터 txt 파일 파싱"""
    txt_file = Path("extracted_questions/리눅스마스터1급202303_questions.txt")
    
    if not txt_file.exists():
        print("❌ TXT 파일을 찾을 수 없습니다.")
        return []
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📄 파일 크기: {len(content)} 문자")
    
    # 문제 번호 패턴들
    patterns = [
        r'(\d+)\.\s*[가-힣]',  # 38. 다음
        r'(\d+)\.\s*[A-Za-z]',  # 38. A
        r'(\d+)\.\s*[①②③④⑤⑥⑦⑧⑨⑩]',  # 38. ①
    ]
    
    questions = []
    lines = content.split('\n')
    
    current_question = None
    current_lines = []
    current_number = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # 문제 시작 감지
        is_question_start = False
        detected_number = None
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                detected_number = match.group(1)
                if detected_number.isdigit() and 1 <= int(detected_number) <= 200:
                    is_question_start = True
                    break
        
        if is_question_start:
            # 이전 문제 저장
            if current_question and current_lines:
                question_text = '\n'.join(current_lines).strip()
                if len(question_text) > 20:
                    questions.append({
                        "number": current_number,
                        "text": question_text
                    })
                    print(f"✅ 문제 {current_number} 저장: {len(question_text)} 문자")
            
            # 새 문제 시작
            current_question = True
            current_lines = [line]
            current_number = detected_number
        else:
            # 현재 문제에 라인 추가
            if current_question is not None:
                current_lines.append(line)
    
    # 마지막 문제 저장
    if current_question and current_lines:
        question_text = '\n'.join(current_lines).strip()
        if len(question_text) > 20:
            questions.append({
                "number": current_number,
                "text": question_text
            })
            print(f"✅ 문제 {current_number} 저장: {len(question_text)} 문자")
    
    print(f"\n📊 총 {len(questions)}개 문제 파싱 완료")
    
    # 문제 번호 목록 출력
    numbers = [q['number'] for q in questions]
    print(f"📋 문제 번호: {numbers}")
    
    return questions

if __name__ == "__main__":
    questions = parse_linux_master_txt() 