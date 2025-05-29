import os
import re
import time
import base64
import logging
import difflib
import tempfile
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from mistralai import Mistral
from httpx import HTTPStatusError
import streamlit as st
from classify import classify_and_move_files
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
api_key = st.secrets["MISTRAL_API_KEY"]
if not api_key:
    raise EnvironmentError("MISTRAL_API_KEY 환경 변수가 설정되지 않았습니다.")
client = Mistral(api_key=api_key)

def parse_data_from_template(text):
    LABELS = ["현장명", "공종", "동호수", "위치", "하자유형", "일자", "치수", "비고", "현황"]
    data = {}
    for line in text.splitlines():
        if ':' not in line:
            continue
        key, val = map(str.strip, line.split(':', 1))
        if key in LABELS:
            data[key] = val
    return data

def get_keyword_categories():
    return {
        '단차불량': ['단차불량', '단차'],
        "훼손": ["훼손", "찢김", "긁힘", "파손", "깨짐", "갈라짐", "찍힘", "스크래치", "손상", "뜯김", "찢어짐", "칼자국", "터짐", "까짐", "흠집",
                 "찍김", "웨손", "긁험", "찍험", "찍임", "직힘", "긁림", "긁임", "찢심", "횟손", "찢검", "찢감"],
        "오염": ["오염", "더러움", "얼룩", "변색", "낙서"],
        "누수 및 곰팡이": ["곰팡이", "누수", "곧광이", "곰광이"],
        "면불량": ["면불량", "면 불량", "퍼티", "돌출", "이물질", "돌기", "벽면불", "면불랑"],
        '들뜸': ["들뜸", "들뜰", "들픔", "들듬", "들음", "들등", "둘뜸", "들뜯", "돌뜸"],
        '꼬임': ["꼬임"], '주름': ["주름"], '울음': ["울음"],
        "석고수정": ["석고", "석고수정", "석고보드", "석고작업", "석고면불량"],
        "몰딩수정": ["몰딩", "몰딩수정", "몰딩교체", "몰딩작업", "돌딩", "올딩"],
        '걸레받이 수정': ["걸레받이", "걸래받이", "걸레받지", "걸레받이수정", "걸레받이 교체", "걸레받이 작업"],
        '문틀수정': ["문틀수정", "문틀"],
        '가구수정': ["가구", "가구수정"],
        '틈새': ["틈새", "틈새수정", "틈새과다", "벌어짐"],
        '합판': ["합판길이부족", "합판"],
        '결로': ["결로"],
        '이음새': ["이음새", "이음"],
        "오타공": ["오타공", "오타콩", "타공과다", "피스타공", "과타공", "타공"],
        "내장후속": ['내장후속', '내장 후속'],
        "탈락": ["탈락"],
        "마감불량": ["마감불량"],
    }

def classify_defect(text):
    categories = get_keyword_categories()
    matched = set()
    words = re.split(r'\s+', text.lower())
    for cat, keywords in categories.items():
        for kw in keywords:
            for word in words:
                if kw in word or difflib.SequenceMatcher(None, kw, word).ratio() > 0.8:
                    matched.add(cat)
                    break
    return ", ".join(sorted(matched)) if matched else ""

def extract_dong_ho(location_text):
    dash_match = re.search(r'\b(\d+)-(\d+)\b', location_text)
    if dash_match:
        return dash_match.group(1), dash_match.group(2)
    dong_match = re.search(r'(\d+)\s*동', location_text)
    ho_match = re.search(r'(\d+)\s*호', location_text)
    dong = dong_match.group(1) if dong_match else ""
    ho = ho_match.group(1) if ho_match else ""
    if not dong or not ho:
        nums = re.findall(r'\d+', location_text)
        if len(nums) == 2:
            dong = dong or nums[0]
            ho = ho or nums[1]
    return dong, ho

def extract_text_from_image(image_path, max_retries=3, backoff=2):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    for attempt in range(max_retries):
        try:
            response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_data}"
                },
                include_image_base64=False
            )
            time.sleep(1.2)

            data = response.model_dump()

            if data.get("text"):
                return data["text"]

            if "pages" in data and "markdown" in data["pages"][0]:
                markdown = data["pages"][0]["markdown"]
                lines = []
                for line in markdown.splitlines():
                    if line.startswith("|") and line.count("|") >= 3:
                        parts = [part.strip() for part in line.strip().split("|")[1:-1]]
                        if len(parts) == 2:
                            key, value = parts
                            lines.append(f"{key}: {value}")
                return "\n".join(lines).strip()

            return ""

        except HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = backoff * (attempt + 1)
                logging.warning(f"429 Too Many Requests - {wait_time}초 후 재시도")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            logging.error(f"OCR 실패: {e}")
            raise
    raise RuntimeError("최대 재시도 초과")

def process_one_image(file_path):
    filename = os.path.basename(file_path)
    try:
        raw_text = extract_text_from_image(file_path)
    except Exception as e:
        logging.warning(f"[오류] {filename} OCR 실패: {e}")
        return None

    parsed = parse_data_from_template(raw_text)
    for fld in ["현장명", "공종", "동호수", "위치", "하자유형", "일자", "치수", "비고", "현황"]:
        parsed.setdefault(fld, "")

    dong, ho = extract_dong_ho(parsed["동호수"])
    keyword = classify_defect(parsed["하자유형"])

    return {
        "파일명": filename,
        "현장명": parsed["현장명"],
        "공종": parsed["공종"],
        "동호수": parsed["동호수"],
        "동": dong,
        "호": ho,
        "위치": parsed["위치"],
        "하자유형": parsed["하자유형"],
        "키워드": keyword,
        "일자": parsed["일자"],
        "치수": parsed["치수"],
        "비고": parsed["비고"],
        "현황": parsed["현황"]
    }

def process_images_to_excel(folder_path, max_workers=1):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_exts)
    ]

    records = []
    progress_bar = st.progress(0, text="이미지 OCR 진행 중...")
    total = len(file_list)
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_image, fp): fp for fp in file_list}
        for future in as_completed(futures):
            rec = future.result()
            if rec is not None:
                records.append(rec)
            completed += 1
            progress_bar.progress(completed / total, text=f"{completed}/{total} 처리 완료")

    progress_bar.empty()
    
    if records:
        columns = [
            "파일명", "현장명", "공종", "동호수", "동", "호",
            "위치", "하자유형", "키워드", "일자", "치수", "비고", "현황"
        ]
        df = pd.DataFrame(records, columns=columns)
        df["동"] = pd.to_numeric(df["동"], errors="coerce").astype("Int64")
        df["호"] = pd.to_numeric(df["호"], errors="coerce").astype("Int64")
        df["파일링크"] = df.apply(
            lambda row: f'=HYPERLINK("{row["키워드"].split(",")[0]}/{row["파일명"]}", "{row["파일명"]}")',
            axis=1
        )
        df = df[["파일명", "파일링크"] + columns[1:]]
        excel_path = os.path.join(folder_path, "results_mistral_ocr.xlsx")
        df.to_excel(excel_path, index=False)
        logging.info(f"[완료] 엑셀 저장: {excel_path}")
    else:
        logging.warning(f"[주의] 유효한 이미지가 없습니다: {folder_path}")

def run_all_steps():
    excel_path = os.path.join(st.session_state.uploaded_dir, "results_mistral_ocr.xlsx")
    classified_dir = os.path.join(tempfile.mkdtemp(), "classified")
    os.makedirs(classified_dir, exist_ok=True)

    # 파일 분류 수행
    classify_and_move_files(excel_path, st.session_state.uploaded_dir)

    # 엑셀 복사
    shutil.copy(excel_path, os.path.join(classified_dir, "results_mistral_ocr.xlsx"))

    # 하위 폴더(분류된 하자유형 폴더들) 복사
    for subdir in os.listdir(st.session_state.uploaded_dir):
        full_path = os.path.join(st.session_state.uploaded_dir, subdir)
        if os.path.isdir(full_path):
            shutil.copytree(full_path, os.path.join(classified_dir, subdir), dirs_exist_ok=True)

    # zip으로 압축
    zip_path = shutil.make_archive(classified_dir, 'zip', classified_dir)
    return zip_path


# Streamlit UI
st.title('HOPEZIP')
st.header('Hansoldeco OCR Project Extraction ZIP')

uploaded_files = st.file_uploader("이미지 파일을 업로드해주세요.", accept_multiple_files=True)

if uploaded_files:
    if "uploaded_dir" not in st.session_state:
        st.session_state.uploaded_dir = tempfile.mkdtemp()

    uploaded_dir = st.session_state.uploaded_dir

    for file in uploaded_files:
        save_path = os.path.join(uploaded_dir, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())

    if st.button("🔍 OCR 및 엑셀 생성"):
        process_images_to_excel(uploaded_dir, max_workers=1)
        st.session_state.ocr_done = True
        st.success("OCR 완료 및 엑셀 생성 완료")

    excel_path = os.path.join(uploaded_dir, "results_mistral_ocr.xlsx")

    if os.path.exists(excel_path):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="📥 엑셀 다운로드",
                data=open(excel_path, "rb").read(),
                file_name="results_mistral_ocr.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )

        with col2:
            with st.spinner("ZIP 생성 중..."):
                zip_path = run_all_steps()
                with open(zip_path, "rb") as f:
                    zip_data = f.read()
            st.download_button(
                label="📦 ZIP 다운로드",
                data=zip_data,
                file_name="classified_files_with_excel.zip",
                mime="application/zip",
                key="zip_download"
            )


        # if st.button("📂 파일 분류 및 엑셀 ZIP 다운로드"):
        #     classified_dir = os.path.join(tempfile.mkdtemp(), "classified")
        #     os.makedirs(classified_dir, exist_ok=True)

        #     classify_and_move_files(excel_path, uploaded_dir)
        #     shutil.copy(excel_path, os.path.join(classified_dir, "results_mistral_ocr.xlsx"))

        #     for subdir in os.listdir(uploaded_dir):
        #         full_path = os.path.join(uploaded_dir, subdir)
        #         if os.path.isdir(full_path):
        #             shutil.copytree(full_path, os.path.join(classified_dir, subdir), dirs_exist_ok=True)

        #     zip_path = shutil.make_archive(classified_dir, 'zip', classified_dir)

        #     with open(zip_path, "rb") as f:
        #         st.download_button(
        #             label="📦 ZIP 다운로드",
        #             data=f.read(),
        #             file_name="classified_files_with_excel.zip",
        #             mime="application/zip"
        #         )
    else:
        st.info("아직 OCR이 수행되지 않았습니다. 먼저 'OCR 및 엑셀 생성'을 클릭하세요.")


