import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple
from  tqdm import tqdm

import requests
import psycopg2
import psycopg2.extras
from sql_connect import get_connection

# =========================
# 설정 파일
# =========================
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


_config = load_config()

COLLECT_DB_DSN = _config.get("COLLECT_DB_DSN")
COLLECT_DB_HOST = _config.get("COLLECT_DB_HOST")
COLLECT_DB_PORT = _config.get("COLLECT_DB_PORT", 5432)
COLLECT_DB_NAME = _config.get("COLLECT_DB_NAME")
COLLECT_DB_USER = _config.get("COLLECT_DB_USER")
COLLECT_DB_PASSWORD = _config.get("COLLECT_DB_PASSWORD")

CLASSIFY_DB_DSN = _config.get("CLASSIFY_DB_DSN")
CLASSIFY_DB_HOST = _config.get("CLASSIFY_DB_HOST")
CLASSIFY_DB_PORT = _config.get("CLASSIFY_DB_PORT", 5432)
CLASSIFY_DB_NAME = _config.get("CLASSIFY_DB_NAME")
CLASSIFY_DB_USER = _config.get("CLASSIFY_DB_USER")
CLASSIFY_DB_PASSWORD = _config.get("CLASSIFY_DB_PASSWORD")

EXAONE_API_URL = _config.get("EXAONE_API_URL")  # 예: https://.../v1/chat/completions (사내 endpoint)
EXAONE_API_KEY = _config.get("EXAONE_API_KEY")
MODEL_NAME = _config.get("EXAONE_MODEL_NAME", "exaone")

BATCH_SIZE = int(_config.get("BATCH_SIZE", 100))
SLEEP_SEC = float(_config.get("SLEEP_SEC", 0.15))
DAILY_AT = _config.get("DAILY_AT")  # 예: "02:00" 또는 ["02:00", "14:00"]


# =========================
# DB: 대상 데이터 조회
# =========================
FETCH_SQL_KOR = """
SELECT cd.ds_id, cd.title, cd.notes
FROM katech.tbdataset cd
WHERE NOT EXISTS (
  SELECT 1
  FROM katech.data_classification_kor_sync dc
  WHERE dc.ds_id = cd.ds_id
    AND dc.status = 'SUCCESS'
) AND cd.data_srttn = '외부'
ORDER BY cd.ds_id
LIMIT %s;
"""

FETCH_SQL_WORLD = """
SELECT cd.ds_id, cd.title, cd.notes
FROM katech.tbdataset cd
WHERE NOT EXISTS (
  SELECT 1
  FROM katech.data_classification_world_sync dc
  WHERE dc.ds_id = cd.ds_id
    AND dc.status = 'SUCCESS'
) AND cd.data_srttn = '해외'
ORDER BY cd.ds_id
LIMIT %s;
"""

UPSERT_SQL = """
INSERT INTO katech.data_classification_kor
  (ds_id, title, notes, model_name, classified_at, is_auto, category_id, reason, confidence, status, error_message, modified_at, input_tokens, output_tokens)
VALUES
  (%s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, NOW(), %s, %s)
ON CONFLICT (ds_id) DO UPDATE SET
  title = EXCLUDED.title,
  notes = EXCLUDED.notes,
  model_name = EXCLUDED.model_name,
  classified_at = EXCLUDED.classified_at,
  is_auto = EXCLUDED.is_auto,
  category_id = EXCLUDED.category_id,
  reason = EXCLUDED.reason,
  confidence = EXCLUDED.confidence,
  status = EXCLUDED.status,
  error_message = EXCLUDED.error_message,
  modified_at = NOW();
	input_tokens = EXCLUDED.input_tokens,
	output_tokens = EXCLUDED.output_tokens
"""

def normalize_text(s: Optional[str], max_len: int) -> str:
    if not s:
        return ""
    s = " ".join(s.split())
    return s[:max_len]


# =========================
# EXAONE: 이진 분류
# =========================
def build_binary_prompt(title: str, notes: str) -> Tuple[str, str]:
    system = f"""
			너는 분류 테스트용이다. 입력데이터의 title과 notes를 파악하여 해당 데이터가 자동차산업 데이터인지 판단한다.
			필수 규칙:
					- 반드시 JSON 형식으로만 응답한다.
					- 추가 설명, 코드블록, 마크다운 등을 절대 출력하지 마라
					- 자동차 산업 관련 데이터면 is_auto=true, 아니면 false                            
					- 출력 형식은 아래와 같다.
			출력 형식:
			{{"is_auto": true or false}}
			""".strip()							
    
    user = f"""
			입력 데이터:
			- title: {title}
			- notes: {notes}
			""".strip()
    return system, user

def parse_json_only(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"JSON 파싱 실패: {text[:200]}")
    return json.loads(text[start:end + 1])


def exaone_call(system: str, user: str) -> Tuple[bool, Optional[float]]:
    """
    반환: (is_auto, confidence)
    confidence는 EXAONE이 제공하지 않으면 None
    """
    headers = {
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
        "X-Api-Key": EXAONE_API_KEY,
        "Content-Type": "application/json"
    }

    # ⚠️ 너희 EXAONE endpoint 스펙에 맞게 payload 키만 조정하면 됨
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 500,
        "temperature": 0.7,
    }

    r = requests.post(EXAONE_API_URL, headers=headers, json=payload)
    r.raise_for_status()
    res = r.json()

    # =========================
    # ✅ 여기 1곳이 “환경별 수정 포인트”
    # 응답이 OpenAI 스타일이면 아래 그대로 OK
    # =========================
    content = res["choices"][0]["message"]["content"]
    obj = parse_json_only(content)
    is_auto = bool(obj["is_auto"])
    input_tokens = res["usage"]["prompt_tokens"]
    output_tokens = res["usage"]["completion_tokens"]
    # confidence 제공 형태가 있으면 여기에 매핑 (없으면 None)
    conf = None
    # 예: conf = res["choices"][0].get("confidence")

    return is_auto, conf, input_tokens, output_tokens


def test_exaone_connection() -> None:
    system = """
            너는 분류 테스트용이다. 입력데이터의 title과 notes를 파악하여 해당 데이터가 자동차산업 데이터인지 판단한다.
            필수 규칙:
                - 반드시 JSON 형식으로만 응답한다.
                - 추가 설명, 코드블록, 마크다운 등을 절대 출력하지 마라
                - 출력 형식은 아래와 같다.
            출력 형식:
            {{"is_auto": true or false}}
                """.strip()
    user = """
            입력 데이터:
            - title: BLM AZ Administrative Units.
            - notes: This dataset depicts Bureau of Land Management (BLM) administrative unit (ADMU) land boundaries and office locations. The land areas for higher level administrative units (district and administrative states) are derived from the lower level administrative units (field office).
            """.strip()

    try:
        is_auto, _, tokens = exaone_call(system, user)
        print(f"EXAONE API 테스트 성공 (is_auto={is_auto}, usage_tokens={tokens})")
    except Exception as e:
        raise RuntimeError(f"EXAONE API 테스트 실패: {e}") from e

def run_test(batch_size: int) -> None:
    collect_conn = get_connection(
        db_dsn=COLLECT_DB_DSN,
        host=COLLECT_DB_HOST,
        port=COLLECT_DB_PORT,
        database=COLLECT_DB_NAME,
        user=COLLECT_DB_USER,
        password=COLLECT_DB_PASSWORD,
    )
    collect_conn.autocommit = False

    classify_conn = get_connection(
        db_dsn=CLASSIFY_DB_DSN,
        host=CLASSIFY_DB_HOST,
        port=CLASSIFY_DB_PORT,
        database=CLASSIFY_DB_NAME,
        user=CLASSIFY_DB_USER,
        password=CLASSIFY_DB_PASSWORD,
    )
    classify_conn.autocommit = False

    # 연결 DB 확인
    with collect_conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print(cur.fetchone())

    with classify_conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print(cur.fetchone())

    with classify_conn.cursor() as cur:
        ds_is = 99999999
        title = "테스트 제목"
        notes = "테스트 노트"
        is_auto = True
        category_id = 0
        reason = None
        conf = 0.95

        cur.execute(
            # UPSERT_SQL,
            # (ds_is, title, notes, MODEL_NAME, is_auto, category_id, reason, conf, "SUCCESS", None)
            "SELECT current_user, session_user;"
        )
        print(cur.fetchone())
    classify_conn.commit()
        
# =========================
# 실행
# =========================
def _parse_daily_times(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]
        return [value.strip()]
    return []


def _next_run_datetime(daily_times: Iterable[str]) -> datetime:
    now = datetime.now()
    candidates = []
    for t in daily_times:
        hour, minute = map(int, t.split(":"))
        dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if dt <= now:
            dt = dt + timedelta(days=1)
        candidates.append(dt)
    return min(candidates)


def _sleep_until_next_run(daily_times: Iterable[str]) -> None:
    next_run = _next_run_datetime(daily_times)
    time.sleep((next_run - datetime.now()).total_seconds())


def run_once(batch_size: int) -> None:
    # api 호출 테스트
    # test_exaone_connection()

    if not all([EXAONE_API_URL, EXAONE_API_KEY]):
        raise RuntimeError("EXAONE_API_URL / EXAONE_API_KEY 설정이 필요합니다")

    if not (COLLECT_DB_DSN or all([COLLECT_DB_HOST, COLLECT_DB_NAME, COLLECT_DB_USER, COLLECT_DB_PASSWORD])):
        raise RuntimeError("수집 DB 설정(COLLECT_DB_*)이 필요합니다")

    if not (CLASSIFY_DB_DSN or all([CLASSIFY_DB_HOST, CLASSIFY_DB_NAME, CLASSIFY_DB_USER, CLASSIFY_DB_PASSWORD])):
        raise RuntimeError("분류 DB 설정(CLASSIFY_DB_*)이 필요합니다")

    collect_conn = get_connection(
        db_dsn=COLLECT_DB_DSN,
        host=COLLECT_DB_HOST,
        port=COLLECT_DB_PORT,
        database=COLLECT_DB_NAME,
        user=COLLECT_DB_USER,
        password=COLLECT_DB_PASSWORD,
    )
    collect_conn.autocommit = False

    classify_conn = get_connection(
        db_dsn=CLASSIFY_DB_DSN,
        host=CLASSIFY_DB_HOST,
        port=CLASSIFY_DB_PORT,
        database=CLASSIFY_DB_NAME,
        user=CLASSIFY_DB_USER,
        password=CLASSIFY_DB_PASSWORD,
    )
    classify_conn.autocommit = False

    # 연결 DB 확인
    with collect_conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print(cur.fetchone())

    with classify_conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print(cur.fetchone())

    # 분류 대상 데이터 조회(국내)
    with collect_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(FETCH_SQL_KOR, (batch_size,))
        rows = cur.fetchall()

    processed = 0
    success = 0
    failed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for row in tqdm(rows, desc="Processing"):
        ds_is = row["ds_id"]
        title = normalize_text(row["title"], 1000)
        notes = normalize_text(row["notes"], 2000)

        try:
            if not title or not notes:
                # 정보 부족이면 분류 대상 제외, category=99
                with classify_conn.cursor() as cur:
                    cur.execute(
                        UPSERT_SQL,
                        (ds_is, title, notes, MODEL_NAME, False, 99, "title/notes 정보 부족", None, "SUCCESS", None)
                    )
                classify_conn.commit()
                processed += 1
                success += 1
                continue

            system, user = build_binary_prompt(title, notes)
            is_auto, conf, input_tokens, output_tokens = exaone_call(system, user)

            # 이진 분류 단계에서는 category/reason은 비움(또는 is_auto=False면 7/사유 저장)
            category_id = 0 if not is_auto else None
            reason = "자동차 산업과 직접 관련성이 낮음" if not is_auto else None
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            with classify_conn.cursor() as cur:
                cur.execute(
                    UPSERT_SQL,
                    (ds_is, title, notes, MODEL_NAME, is_auto, category_id, reason, conf, "SUCCESS", None, input_tokens, output_tokens)
                )
            classify_conn.commit()

            processed += 1
            success += 1
            time.sleep(SLEEP_SEC)

        except Exception as e:
            classify_conn.rollback()
            with classify_conn.cursor() as cur:
                cur.execute(
                    UPSERT_SQL,
                    (ds_is, title, notes, MODEL_NAME, False, None, None, None, "FAILED", str(e))
                )
            classify_conn.commit()

            processed += 1
            failed += 1

    collect_conn.close()
    classify_conn.close()
    print(f"done. processed={processed}, success={success}, failed={failed}, usage_total_tokens={total_input_tokens + total_output_tokens}")

def main() -> None:
    parser = argparse.ArgumentParser(description="데이터 분류 배치")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--daily-at", type=str, default=DAILY_AT)
    args = parser.parse_args()

    daily_times = _parse_daily_times(args.daily_at)

    if daily_times:
        while True:
            _sleep_until_next_run(daily_times)
            # run_once(args.batch_size)
    else:
        run_once(args.batch_size)

    # run_test(args.batch_size)

if __name__ == "__main__":
    main()
