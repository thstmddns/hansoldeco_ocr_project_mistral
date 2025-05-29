import os
import shutil
import pandas as pd

def get_keyword_categories():
    return {
        '단차불량': ['단차불량', '단차'],
        '훼손': ['훼손', '찢김', '긁힘', '파손', '깨짐', '갈라짐', '찍힘',
               '스크래치', '손상', '뜯김', '찢어짐', '칼자국', '터짐', '까짐', '흠집',
               '웨손', '긁험', '찍험', '찍임', '직힘', '긁림', '찢심', '횟손', '찢검', '찢감'],
        '오염': ['오염', '더러움', '얼룩', '변색', '낙서', '볼펜자국'],
        '누수 및 곰팡이': ['누수 및 곰팡이', '곰팡이', '누수', '곧광이', '곰광이'],
        '면불량': ['면불량', '면 불량', '퍼티', '돌출', '이물질', '돌기', '벽면불', '면불랑'],
        '들뜸': ['들뜸', '들뜰', '들픔', '들듬', '들음', '들등', '둘뜸', '들뜯', '돌뜸'],
        '꼬임': ['꼬임'],
        '주름': ['주름'],
        '울음': ['울음'],
        '석고수정': ['석고', '석고수정', '석고보드', '석고작업', '석고면불량'],
        '몰딩수정': ['몰딩', '몰딩수정', '몰딩교체', '몰딩작업', '돌딩', '올딩'],
        '걸레받이 수정': ['걸레받이 수정', '걸레받이', '걸래받이', '걸레받지', '걸레받이수정', '걸레받이 교체', '걸레받이 작업'],
        '문틀수정': ['문틀수정', '문틀'],
        '가구수정': ['가구', '가구수정'],
        '틈새': ['틈새', '틈새수정', '틈새과다', '벌어짐'],
        '합판': ['합판길이부족', '합판'],
        '결로': ['결로'],
        '이음새': ['이음새', '이음'],
        '오타공': ['오타공', '오타콩', '타공과다', '피스타공', '과타공', '타공'],
        '탈락': ['탈락'],
        "후속" : ['후속', '내장후속', '내장 후속'],
        "마감불량" : ["마감불량"],
        "폼시공" : ["폼시공"],
    }

def classify_and_move_files(excel_path, source_folder):
    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    keyword_mapping = get_keyword_categories()

    for _, row in df.iterrows():
        file_name = str(row["파일명"]).strip()
        keywords = str(row["키워드"]).strip()
        keyword_list = [k.strip() for k in keywords.replace("/", ",").split(",") if k.strip()]

        target_folder = None
        for keyword in keyword_list:
            for category, synonyms in keyword_mapping.items():
                if keyword in synonyms:
                    target_folder = category
                    break
            if target_folder:
                break

        if not target_folder:
            target_folder = "unidentified"

        source_path = os.path.join(source_folder, file_name)
        dest_folder = os.path.join(source_folder, target_folder)
        dest_path = os.path.join(dest_folder, file_name)

        if os.path.exists(source_path):
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(source_path, dest_path)
            print(f"Moved: {file_name} -> {target_folder}")
        else:
            print(f"File not found: {file_name}")

    print("✅ 파일 분류 및 이동 완료.")
