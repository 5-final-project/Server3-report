# uvicorn action:app --reload --host 0.0.0.0 --port 8010

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import aiohttp
import asyncio
import re
import time

app = FastAPI(
    title="Async Map-Reduce 병렬 요약 API",
    description="Qwen LLM을 이용해 map 단계는 병렬, reduce(combine) 단계는 통합 LLM 요청으로 처리",
    version="1.1"
)

QWEN_API_URL = "https://qwen3.ap.loclx.io/api/generate"

# ----- 데이터 모델 -----
class RelatedDoc(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class ChunkItem(BaseModel):
    chunk_en: str
    related_docs: List[RelatedDoc]

class MeetingMeta(BaseModel):
    title: str
    datetime: str
    author: str
    participants: List[str]

class MeetingInput(BaseModel):
    meeting_meta: MeetingMeta
    meeting_purpose: str
    insights: List[str]
    text_stt: str
    chunks: List[ChunkItem]
    elapsed_time: Optional[float] = None
    error: Optional[str] = None

# ----- LLM 처리 함수 -----
def clean_llm_output(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

async def async_llm_map_summary(chunk_text: str, session: aiohttp.ClientSession) -> str:
    map_prompt = f"""
아래 회의내용을 핵심 중심으로 2줄 또는 3줄로 요약해주세요. **동일하거나 유사한 내용은 한 번만** 언급해주세요. 반드시 요약문을 출력해주세요. 

회의내용:
{chunk_text}
"""
    payload = {
        "prompt": [{"role": "user", "content": map_prompt}],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9
    }
    async with session.post(QWEN_API_URL, json=payload, timeout=120) as response:
        result = await response.json()
        if "response" not in result:
            raise ValueError("LLM 응답에 'response' 키가 없습니다.")
        return clean_llm_output(result["response"].strip())

async def async_llm_combine_summary(map_summaries: list, session: aiohttp.ClientSession) -> str:
    combined_text = "\n".join(map_summaries)
    combine_prompt = f"""
여러 회의 요약문이 아래에 나열되어 있습니다.  
이 요약들을 중복 없이 통합하여, 전체 회의의 맥락이 자연스럽게 이어지도록, 논리적이고 일관된 한 문단으로 재구성하세요. 
재구성한 문단을 절대로 누락하면 안됩니다. 반드시 출력해주세요

[chunk별 요약 목록]
{combined_text}

최종 통합 요약:
"""
    payload = {
        "prompt": [{"role": "user", "content": combine_prompt}],
        "max_tokens": 1024,
        "temperature": 0.5,
        "top_p": 0.9
    }
    async with session.post(QWEN_API_URL, json=payload, timeout=120) as response:
        result = await response.json()
        if "response" not in result:
            raise ValueError("LLM 응답에 'response' 키가 없습니다.")
        return clean_llm_output(result["response"].strip())

async def async_llm_action_items(full_summary: str, session: aiohttp.ClientSession) -> str:
    action_item_prompt = f"""
아래 회의 요약문을 기반으로 action item(실행해야 할 항목/후속조치/업무 지시사항)을 최대한 구체적으로, 명확하게 항목별로 작성해주세요.
실행 주체가 명확하다면 함께 명시하세요.

[회의 요약문]
{full_summary}

아래와 같은 형식으로 출력하세요:

1. ~~~
2. ~~~
3. ~~~
"""
    payload = {
        "prompt": [{"role": "user", "content": action_item_prompt}],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9
    }
    async with session.post(QWEN_API_URL, json=payload, timeout=120) as response:
        result = await response.json()
        if "response" not in result:
            raise ValueError("LLM 응답에 'response' 키가 없습니다.")
        return clean_llm_output(result["response"].strip())

async def async_llm_final_5line_summary(full_summary: str, session: aiohttp.ClientSession) -> str:
    final_summary_prompt = f"""
다음 회의 내용을 기반으로 내용을 5줄로 정리해주세요.

[회의내용]
{full_summary}

아래와 같은 형식으로 출력해주세요:

1. ~~~
2. ~~~
3. ~~~
4. ~~~
5. ~~~
"""
    payload = {
        "prompt": [{"role": "user", "content": final_summary_prompt}],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9
    }
    async with session.post(QWEN_API_URL, json=payload, timeout=120) as response:
        result = await response.json()
        if "response" not in result:
            raise ValueError("LLM 응답에 'response' 키가 없습니다.")
        return clean_llm_output(result["response"].strip())

# ----- 엔드포인트 -----
@app.post("/report-json")
async def report_json(request: MeetingInput):
    try:
        total_start = time.perf_counter()  # 전체 처리 시작

        # 1. chunk 분할
        stt_text = request.text_stt
        t0 = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunks = splitter.split_text(stt_text)
        t1 = time.perf_counter()
        print(f"[TIME] chunk split: {t1-t0:.3f} sec, chunks={len(chunks)}")

        if not chunks:
            raise HTTPException(status_code=400, detail="요약할 텍스트가 없습니다.")

        async with aiohttp.ClientSession() as session:
            # 2. map 단계
            t0 = time.perf_counter()
            map_tasks = [
                async_llm_map_summary(chunk, session) for chunk in chunks
            ]
            map_summaries = await asyncio.gather(*map_tasks)
            t1 = time.perf_counter()
            print(f"[TIME] map summaries (LLM 요청): {t1-t0:.3f} sec")

            # 3. reduce 단계
            t0 = time.perf_counter()
            full_summary = await async_llm_combine_summary(map_summaries, session)
            t1 = time.perf_counter()
            print(f"[TIME] reduce/combine summary (LLM 요청): {t1-t0:.3f} sec")

            # 4. Action Item
            t0 = time.perf_counter()
            action_items = await async_llm_action_items(full_summary, session)
            t1 = time.perf_counter()
            print(f"[TIME] action items (LLM 요청): {t1-t0:.3f} sec")

            # 5. 5줄 요약
            t0 = time.perf_counter()
            final_5line_summary = await async_llm_final_5line_summary(full_summary, session)
            t1 = time.perf_counter()
            print(f"[TIME] 5줄 요약 (LLM 요청): {t1-t0:.3f} sec")

        total_end = time.perf_counter()
        print(f"[TIME] 전체 처리 시간: {total_end-total_start:.3f} sec")

        meeting_info = {
            "title": request.meeting_meta.title,
            "datetime": request.meeting_meta.datetime,
            "author": request.meeting_meta.author,
            "participants": request.meeting_meta.participants,
            "meeting_purpose": request.meeting_purpose
        }

        # 모든 chunk의 related_docs에서 doc_id만 모으기 (중복제거)
        all_doc_ids = set()
        for chunk in request.chunks:
            for doc in chunk.related_docs:
                doc_id = doc.metadata.get("doc_id") if isinstance(doc.metadata, dict) else None
                if doc_id:
                    all_doc_ids.add(doc_id)
        all_related_doc_ids = list(all_doc_ids)

        return {
            "meeting_info": meeting_info,
            #"chunk_summaries": map_summaries,
            #"intermediate_summary": full_summary,
            "final_5line_summary": final_5line_summary,
            "action_items": action_items,
            "all_related_doc_ids": all_related_doc_ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 중 오류 발생: {e}")

@app.get("/")
def root():
    return {"message": "Async Map-Reduce 병렬 요약 API 작동 중"}
