#!/usr/bin/env python3
"""Comment multiplatform incremental pipeline (v2)."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


PARSER_VERSION = "comment_v2.0.1"
MID_HASH_SALT = "ZHENGDA_COMMENT_V2_SALT_2026"
RANDOM_SEED = 20260301

SERVICE_DIMS = [
    "security_check",
    "free_supply",
    "controlled_lightstick",
    "network_signal",
    "temperature_control",
    "hygiene",
    "photo_checkin",
    "shuttle_service",
]
OUTCOME_DIMS = ["satisfaction", "recommendation"]
ALL_DIMS = SERVICE_DIMS + OUTCOME_DIMS

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49, 33, 9,
    42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1,
    60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

SHOW_NAME_STOPWORDS = {
    "演唱会", "巡回", "巡演", "世界", "全国", "音乐会", "站", "live", "LIVE",
}

RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
RE_MENTION = re.compile(r"@[\w\-\u4e00-\u9fff]+")
RE_BRACKET = re.compile(r"\[[^\]]+\]")
RE_WHITESPACE = re.compile(r"\s+")
RE_TOKEN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
RE_EMOJI = re.compile(r"[\U00010000-\U0010ffff]")
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_WBI_FILTER = re.compile(r"[!'()*]")
RE_DATE = re.compile(r"(20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2})")


@dataclass
class Args:
    input: str
    out_dir: str
    years: list[int]
    max_videos: int
    max_comments_per_video: int
    retry: int
    sleep: float
    timeout: int
    dictionary: str
    platforms: list[str]
    xhs_min_records: int
    experience_filter: bool
    calibration_size: int
    out_version: str
    old_comment_dir: str


def parse_bool(value: str) -> bool:
    low = str(value).strip().lower()
    if low in {"1", "true", "yes", "y", "on"}:
        return True
    if low in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path("d:/fcc/ZhengDa/data/basic information/basic information.xlsx")))
    parser.add_argument("--out-dir", default=str(Path("d:/fcc/ZhengDa/data/comment")))
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--max-videos", type=int, default=3)
    parser.add_argument("--max-comments-per-video", type=int, default=200)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--dict", dest="dictionary", default=str(Path(__file__).with_name("kano_sem_dimension_dict.json")))
    parser.add_argument("--platforms", nargs="+", default=["bilibili", "xiaohongshu", "douban"])
    parser.add_argument("--xhs-min-records", type=int, default=200)
    parser.add_argument("--experience-filter", default="true")
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--out-version", default="v2_20260301")
    parser.add_argument("--old-comment-dir", default=str(Path("d:/fcc/ZhengDa/data/comment")))
    parsed = parser.parse_args()
    return Args(
        input=parsed.input,
        out_dir=parsed.out_dir,
        years=parsed.years,
        max_videos=parsed.max_videos,
        max_comments_per_video=parsed.max_comments_per_video,
        retry=parsed.retry,
        sleep=parsed.sleep,
        timeout=parsed.timeout,
        dictionary=parsed.dictionary,
        platforms=[x.strip().lower() for x in parsed.platforms],
        xhs_min_records=parsed.xhs_min_records,
        experience_filter=parse_bool(parsed.experience_filter),
        calibration_size=parsed.calibration_size,
        out_version=parsed.out_version,
        old_comment_dir=parsed.old_comment_dir,
    )


def now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_html_text(text: Any) -> str:
    raw = html.unescape(str(text or ""))
    return RE_WHITESPACE.sub(" ", RE_HTML.sub(" ", raw)).strip()


def clean_comment_text(text: Any) -> str:
    t = html.unescape(str(text or ""))
    t = RE_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MENTION.sub(" ", t)
    t = RE_BRACKET.sub(" ", t)
    t = RE_EMOJI.sub(" ", t)
    t = RE_CONTROL.sub(" ", t)
    t = RE_WHITESPACE.sub(" ", t)
    return t.strip()


def parse_cn_numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").rstrip("+")
    if not text or text in {"--", "-", "None", "null", "nan"}:
        return 0.0
    mul = 1.0
    if text.endswith("万"):
        mul = 10000.0
        text = text[:-1]
    elif text.endswith("亿"):
        mul = 100000000.0
        text = text[:-1]
    try:
        return float(text) * mul
    except Exception:  # noqa: BLE001
        m = re.findall(r"\d+(?:\.\d+)?", text)
        return float(m[0]) * mul if m else 0.0


def to_datetime_from_unix(timestamp: Any) -> pd.Timestamp | pd.NaT:
    try:
        ts_int = int(timestamp)
        if ts_int <= 0:
            return pd.NaT
        return pd.to_datetime(ts_int, unit="s", utc=True).tz_convert("Asia/Shanghai").tz_localize(None)
    except Exception:  # noqa: BLE001
        return pd.NaT


def extract_show_core_tokens(show_name: str) -> list[str]:
    tokens: list[str] = []
    for token in RE_TOKEN.findall(str(show_name or "")):
        if len(token) <= 1:
            continue
        if token in SHOW_NAME_STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def build_experience_terms(dimension_keywords: dict[str, list[str]], extra_terms: list[str]) -> list[str]:
    terms: list[str] = []
    for dim in SERVICE_DIMS:
        for t in dimension_keywords.get(dim, []):
            term = str(t).strip()
            if term and term not in terms:
                terms.append(term)
    for t in extra_terms:
        term = str(t).strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def match_terms(text: str, terms: list[str]) -> list[str]:
    low_text = text.lower()
    hits: list[str] = []
    for term in terms:
        if term.lower() in low_text:
            hits.append(term)
    return hits


def match_dimensions(text: str, dimension_keywords: dict[str, list[str]]) -> list[str]:
    low_text = text.lower()
    tags: list[str] = []
    for dim, keywords in dimension_keywords.items():
        for keyword in keywords:
            term = str(keyword).strip()
            if term and term.lower() in low_text:
                tags.append(dim)
                break
    return tags


def compute_sentiment(text: str, pos_words: list[str], neg_words: list[str], negations: list[str]) -> tuple[str, float]:
    normalized = text.lower()
    pos_count = 0
    neg_count = 0

    def is_negated(idx: int) -> bool:
        start = max(0, idx - 4)
        win = normalized[start:idx]
        return any(n in win for n in negations)

    for word in pos_words:
        w = str(word).strip().lower()
        if not w:
            continue
        for m in re.finditer(re.escape(w), normalized):
            if is_negated(m.start()):
                neg_count += 1
            else:
                pos_count += 1

    for word in neg_words:
        w = str(word).strip().lower()
        if not w:
            continue
        for m in re.finditer(re.escape(w), normalized):
            if is_negated(m.start()):
                pos_count += 1
            else:
                neg_count += 1

    total = pos_count + neg_count
    if total == 0:
        return "neu", 0.0
    score = (pos_count - neg_count) / total
    label = "pos" if score > 0.1 else "neg" if score < -0.1 else "neu"
    return label, round(float(score), 6)


def map_kano_roles(tags: list[str], role_map: dict[str, list[str]]) -> str:
    tag_set = set(tags)
    roles: list[str] = []
    for role in ["must_be", "one_dim", "attractive"]:
        dims = set(role_map.get(role, []))
        if tag_set.intersection(dims):
            roles.append(role)
    return ";".join(roles)


def normalize_concert_df(input_path: Path, years: list[int]) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    required = ["演出ID", "演出名称", "艺人/团体", "城市", "场馆名称", "开演日期", "售票平台"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"missing required columns: {miss}")
    out = df.copy()
    out["开演日期"] = pd.to_datetime(out["开演日期"], errors="coerce")
    out["开演年份"] = out["开演日期"].dt.year
    out = out[out["开演年份"].isin(years)].copy()
    out = out.sort_values(["开演日期", "演出ID"]).reset_index(drop=True)
    return out


def load_baseline(old_dir: Path) -> dict[str, Any]:
    baseline: dict[str, Any] = {"exists": old_dir.exists(), "files": {}, "metrics": {}}
    if not old_dir.exists():
        return baseline
    old_files = [
        "concert_comments_raw_bilibili_2024_2025.xlsx",
        "concert_comments_labeled_kano_sem_2024_2025.xlsx",
        "concert_sem_ready_metrics_2024_2025.xlsx",
        "concert_video_match_audit_2024_2025.xlsx",
    ]
    for name in old_files:
        p = old_dir / name
        if p.exists():
            baseline["files"][name] = {"size": int(p.stat().st_size), "md5": file_md5(p)}
    l_path = old_dir / "concert_comments_labeled_kano_sem_2024_2025.xlsx"
    s_path = old_dir / "concert_sem_ready_metrics_2024_2025.xlsx"
    if l_path.exists():
        old_labeled = pd.read_excel(l_path)
        baseline["metrics"]["old_labeled_rows"] = int(len(old_labeled))
        if "dimension_tags" in old_labeled.columns and len(old_labeled):
            baseline["metrics"]["old_any_dim_ratio"] = float(
                old_labeled["dimension_tags"].fillna("").astype(str).ne("").mean()
            )
        else:
            baseline["metrics"]["old_any_dim_ratio"] = 0.0
    if s_path.exists():
        old_sem = pd.read_excel(s_path)
        baseline["metrics"]["old_sem_rows"] = int(len(old_sem))
        if "total_comments" in old_sem.columns:
            baseline["metrics"]["old_sem_nonzero_shows"] = int((old_sem["total_comments"] > 0).sum())
        else:
            baseline["metrics"]["old_sem_nonzero_shows"] = 0
    return baseline


class BilibiliClient:
    def __init__(self, retry: int, sleep: float, timeout: int):
        self.retry = retry
        self.sleep = sleep
        self.timeout = timeout
        self._opener = urllib.request.build_opener()
        self._wbi_key: str | None = None

    def _request_json(self, url: str, referer: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for i in range(self.retry):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "application/json, text/plain, */*",
                        "Accept-Language": "zh-CN,zh;q=0.9",
                        "Referer": referer,
                    },
                )
                with self._opener.open(req, timeout=self.timeout) as resp:
                    txt = resp.read().decode("utf-8", errors="ignore")
                return json.loads(txt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(self.sleep * (2**i))
        raise RuntimeError(f"request failed: {url} :: {last_error}") from last_error

    def _ensure_wbi_key(self) -> str:
        if self._wbi_key:
            return self._wbi_key
        nav = self._request_json("https://api.bilibili.com/x/web-interface/nav", referer="https://www.bilibili.com/")
        wbi_img = nav.get("data", {}).get("wbi_img", {})
        img_key = (wbi_img.get("img_url") or "").rsplit("/", 1)[-1].split(".")[0]
        sub_key = (wbi_img.get("sub_url") or "").rsplit("/", 1)[-1].split(".")[0]
        if not img_key or not sub_key:
            raise RuntimeError(f"nav endpoint failed: {nav.get('message')}")
        merged = img_key + sub_key
        self._wbi_key = "".join(merged[i] for i in MIXIN_KEY_ENC_TAB)[:32]
        return self._wbi_key

    def _sign_wbi(self, params: dict[str, Any]) -> str:
        key = self._ensure_wbi_key()
        signed = {k: RE_WBI_FILTER.sub("", str(v)) for k, v in params.items()}
        signed["wts"] = str(int(time.time()))
        query = urllib.parse.urlencode(sorted(signed.items()))
        w_rid = hashlib.md5((query + key).encode("utf-8")).hexdigest()
        return f"{query}&w_rid={w_rid}"

    def search_videos(self, keyword: str) -> list[dict[str, Any]]:
        query = self._sign_wbi({"search_type": "video", "keyword": keyword, "page": 1})
        url = f"https://api.bilibili.com/x/web-interface/wbi/search/type?{query}"
        data = self._request_json(url, referer="https://search.bilibili.com/")
        if int(data.get("code", -1)) != 0:
            raise RuntimeError(f"search failed: {data.get('message')}")
        return data.get("data", {}).get("result", []) or []

    def fetch_comments(self, aid: int, bvid: str, max_comments: int) -> tuple[list[dict[str, Any]], int]:
        comments: list[dict[str, Any]] = []
        all_count = 0
        next_cursor: int | str = 0
        seen: set[str] = set()
        while True:
            query = self._sign_wbi({"oid": aid, "type": 1, "mode": 3, "next": next_cursor, "ps": 20})
            url = f"https://api.bilibili.com/x/v2/reply/wbi/main?{query}"
            data = self._request_json(url, referer=f"https://www.bilibili.com/video/{bvid}")
            if int(data.get("code", -1)) != 0:
                break
            payload = data.get("data", {}) or {}
            cursor = payload.get("cursor", {}) or {}
            if cursor.get("all_count") is not None:
                try:
                    all_count = int(cursor.get("all_count"))
                except Exception:
                    pass
            replies = payload.get("replies", []) or []
            for rep in replies:
                rid = str(rep.get("rpid_str") or rep.get("rpid") or "")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                comments.append(rep)
                if len(comments) >= max_comments:
                    return comments, all_count
            if bool(cursor.get("is_end")):
                break
            nxt = cursor.get("next")
            if nxt is None or str(nxt) == str(next_cursor):
                break
            next_cursor = nxt
            time.sleep(self.sleep)
        return comments, all_count


def score_bilibili_candidate(cand: dict[str, Any], show: pd.Series, core_tokens: list[str]) -> dict[str, Any]:
    title = clean_html_text(cand.get("title", ""))
    desc = clean_html_text(cand.get("description", ""))
    tags = clean_html_text(cand.get("tag", ""))
    text = f"{title} {desc} {tags}"
    text_lower = text.lower()

    artist = str(show["艺人/团体"] or "")
    city = str(show["城市"] or "")
    show_date = pd.to_datetime(show["开演日期"], errors="coerce")

    artist_hit = 5 if artist and artist in text else 0
    city_hit = 3 if city and city in text else 0
    core_hits = sum(1 for t in core_tokens if t.lower() in text_lower)
    core_score = min(core_hits * 2, 6)

    pubdate = to_datetime_from_unix(cand.get("pubdate"))
    date_score = 0
    if pd.notna(pubdate) and pd.notna(show_date):
        gap = float(abs((pubdate.normalize() - show_date.normalize()).days))
        if gap <= 45:
            date_score = 2
        elif gap <= 120:
            date_score = 1
    engagement = parse_cn_numeric(cand.get("play")) + parse_cn_numeric(cand.get("review"))
    engagement_score = 0.2 * math.log1p(max(0.0, engagement))
    total = artist_hit + city_hit + core_score + date_score + engagement_score
    return {
        "title_clean": title,
        "pubdate": pubdate,
        "artist_hit_score": artist_hit,
        "city_hit_score": city_hit,
        "core_token_hit_score": core_score,
        "date_score": date_score,
        "engagement_score": round(engagement_score, 6),
        "total_score": round(total, 6),
    }


def crawl_bilibili(shows: pd.DataFrame, args: Args, crawl_batch_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    client = BilibiliClient(retry=args.retry, sleep=args.sleep, timeout=args.timeout)
    _ = client._ensure_wbi_key()

    raw_records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    search_success = 0
    show_with_records = 0

    for _, show in shows.iterrows():
        show_id = str(show["演出ID"])
        show_name = str(show["演出名称"])
        artist = str(show["艺人/团体"])
        city = str(show["城市"])
        query = f"{artist} {show_name} {city} 演唱会"
        core_tokens = extract_show_core_tokens(show_name)
        status = "success"
        reason = ""
        selected_count = 0
        fetched_count = 0
        raw_source_url = ""

        try:
            candidates = client.search_videos(query)[:20]
            search_success += 1
        except Exception as exc:  # noqa: BLE001
            candidates = []
            status = "search_failed"
            reason = str(exc)

        scored: list[dict[str, Any]] = []
        for cand in candidates:
            aid = cand.get("aid")
            bvid = str(cand.get("bvid") or "")
            aid_int = int(aid) if str(aid).isdigit() else None
            score = score_bilibili_candidate(cand, show, core_tokens)
            scored.append(
                {
                    "aid": aid_int,
                    "bvid": bvid,
                    "video_title": score["title_clean"],
                    "video_pubdate": score["pubdate"],
                    "artist_hit_score": score["artist_hit_score"],
                    "city_hit_score": score["city_hit_score"],
                    "date_score": score["date_score"],
                    "total_score": score["total_score"],
                }
            )
        scored.sort(key=lambda x: x["total_score"], reverse=True)

        eligible: list[dict[str, Any]] = []
        used_bvid: set[str] = set()
        for row in scored:
            if not row["aid"] or not row["bvid"]:
                continue
            if row["artist_hit_score"] <= 0:
                continue
            if row["date_score"] <= 0 and row["city_hit_score"] <= 0:
                continue
            if row["total_score"] < 7:
                continue
            if row["bvid"] in used_bvid:
                continue
            used_bvid.add(row["bvid"])
            eligible.append(row)

        selected = eligible[: args.max_videos]
        selected_count = len(selected)
        if status == "success" and selected_count == 0:
            status = "no_candidate"
            reason = "strict_filter_no_candidate"

        for vid in selected:
            aid = int(vid["aid"])
            bvid = str(vid["bvid"])
            raw_source_url = f"https://www.bilibili.com/video/{bvid}"
            comments, hint = client.fetch_comments(aid=aid, bvid=bvid, max_comments=args.max_comments_per_video)
            for rep in comments:
                comment_id = str(rep.get("rpid_str") or rep.get("rpid") or "")
                if not comment_id:
                    continue
                content = str((rep.get("content") or {}).get("message") or "").strip()
                if not content:
                    continue
                ctime = to_datetime_from_unix(rep.get("ctime"))
                member = rep.get("member") or {}
                mid_str = str(rep.get("mid_str") or member.get("mid") or "")
                raw_records.append(
                    {
                        "show_id": show_id,
                        "show_name": show_name,
                        "artist": artist,
                        "city": city,
                        "concert_date": show["开演日期"],
                        "show_ticket_platform": show.get("售票平台"),
                        "platform": "bilibili",
                        "source_level": "comment",
                        "platform_record_id": f"{show_id}_{bvid}_{comment_id}",
                        "content_raw": content,
                        "source_url": raw_source_url,
                        "query_text": query,
                        "title": vid["video_title"],
                        "publish_time": ctime,
                        "like_cnt": int(rep.get("like") or 0),
                        "reply_cnt": int(rep.get("rcount") or rep.get("count") or 0),
                        "aid": aid,
                        "bvid": bvid,
                        "video_title": vid["video_title"],
                        "video_pubdate": vid["video_pubdate"],
                        "video_comment_total_hint": int(hint or 0),
                        "mid_hash": sha256_text(mid_str + MID_HASH_SALT),
                        "crawl_batch_id": crawl_batch_id,
                        "crawl_time": now_local_str(),
                        "parser_version": PARSER_VERSION,
                    }
                )
            fetched_count += len(comments)
            time.sleep(args.sleep)

        if fetched_count > 0:
            show_with_records += 1
        elif status == "success":
            status = "no_comment"
            reason = "selected_videos_no_comment"

        audit_rows.append(
            {
                "show_id": show_id,
                "show_name": show_name,
                "platform": "bilibili",
                "query_text": query,
                "fetch_status": status,
                "selected_count": selected_count,
                "fetched_records": fetched_count,
                "fallback_reason": reason,
                "raw_source_url": raw_source_url,
                "crawl_time": now_local_str(),
            }
        )

    stats = {
        "search_success": int(search_success),
        "show_with_records": int(show_with_records),
        "total_records": int(len(raw_records)),
    }
    return raw_records, audit_rows, stats


def extract_meta_content(page_obj: Any, selector: str) -> str:
    try:
        text = page_obj.eval_on_selector(selector, "el => el && el.content ? el.content : ''")
        return str(text or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def crawl_xiaohongshu_notes(shows: pd.DataFrame, args: Args, crawl_batch_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    stats = {"available": False, "queries": 0, "records": 0}
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:  # noqa: BLE001
        audit_rows.append(
            {
                "show_id": "",
                "show_name": "",
                "platform": "xiaohongshu",
                "query_text": "",
                "fetch_status": "unavailable",
                "selected_count": 0,
                "fetched_records": 0,
                "fallback_reason": f"playwright_not_available: {exc}",
                "raw_source_url": "",
                "crawl_time": now_local_str(),
            }
        )
        return records, audit_rows, stats

    stats["available"] = True
    seen_note_ids: set[str] = set()
    max_queries = min(30, len(shows))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        detail = browser.new_page(user_agent=USER_AGENT)
        for i, (_, show) in enumerate(shows.head(max_queries).iterrows(), start=1):
            show_id = str(show["演出ID"])
            show_name = str(show["演出名称"])
            artist = str(show["艺人/团体"])
            city = str(show["城市"])
            query = f"{artist} {show_name} {city} 观演 体验"
            search_url = "https://www.xiaohongshu.com/search_result?keyword=" + urllib.parse.quote(query)
            status = "success"
            reason = ""
            fetched = 0
            try:
                page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(3000)
                try:
                    page.mouse.wheel(0, 1600)
                    page.wait_for_timeout(1200)
                except Exception:
                    pass
                links = page.eval_on_selector_all(
                    "a[href*='/explore/']",
                    "els => els.map(e => e.href).filter(Boolean)",
                )
                uniq_links: list[str] = []
                for link in links:
                    item = str(link or "")
                    if "/explore/" not in item:
                        continue
                    if item.startswith("/"):
                        item = "https://www.xiaohongshu.com" + item
                    if item not in uniq_links:
                        uniq_links.append(item)
                if not uniq_links:
                    status = "no_result"
                    reason = "search_result_no_explore_link"
                for link in uniq_links[:5]:
                    note_id = link.rstrip("/").split("/")[-1]
                    if not note_id or note_id in seen_note_ids:
                        continue
                    seen_note_ids.add(note_id)
                    try:
                        detail.goto(link, wait_until="domcontentloaded", timeout=45000)
                        detail.wait_for_timeout(1800)
                        title = str(detail.title() or "").strip()
                        desc = extract_meta_content(detail, "meta[name='description']")
                        if not desc:
                            desc = extract_meta_content(detail, "meta[property='og:description']")
                        if not title:
                            title = extract_meta_content(detail, "meta[property='og:title']")
                        content = clean_comment_text(f"{title} {desc}")
                        if not content:
                            continue
                        html_text = detail.content()
                        like_match = re.search(r"\"likedCount\"\\s*:\\s*\"?([^\",}]+)", html_text)
                        com_match = re.search(r"\"commentCount\"\\s*:\\s*\"?([^\",}]+)", html_text)
                        time_match = RE_DATE.search(content) or RE_DATE.search(html_text)
                        pub = pd.NaT
                        if time_match:
                            ts = (
                                time_match.group(1)
                                .replace("年", "-")
                                .replace("月", "-")
                                .replace("日", "")
                                .replace("/", "-")
                            )
                            pub = pd.to_datetime(ts, errors="coerce")
                        records.append(
                            {
                                "show_id": show_id,
                                "show_name": show_name,
                                "artist": artist,
                                "city": city,
                                "concert_date": show["开演日期"],
                                "show_ticket_platform": show.get("售票平台"),
                                "platform": "xiaohongshu",
                                "source_level": "note",
                                "platform_record_id": note_id,
                                "content_raw": content,
                                "source_url": link,
                                "query_text": query,
                                "title": title,
                                "publish_time": pub,
                                "like_cnt": int(parse_cn_numeric(like_match.group(1) if like_match else 0)),
                                "reply_cnt": int(parse_cn_numeric(com_match.group(1) if com_match else 0)),
                                "aid": None,
                                "bvid": "",
                                "video_title": "",
                                "video_pubdate": pd.NaT,
                                "video_comment_total_hint": 0,
                                "mid_hash": "",
                                "crawl_batch_id": crawl_batch_id,
                                "crawl_time": now_local_str(),
                                "parser_version": PARSER_VERSION,
                            }
                        )
                        fetched += 1
                    except Exception:
                        continue
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                reason = str(exc)

            audit_rows.append(
                {
                    "show_id": show_id,
                    "show_name": show_name,
                    "platform": "xiaohongshu",
                    "query_text": query,
                    "fetch_status": status,
                    "selected_count": 0,
                    "fetched_records": fetched,
                    "fallback_reason": reason,
                    "raw_source_url": search_url,
                    "crawl_time": now_local_str(),
                }
            )
            stats["queries"] = i
            if len(records) >= args.xhs_min_records:
                break
        browser.close()
    stats["records"] = int(len(records))
    return records, audit_rows, stats


def infer_show_id_from_text(text: str, artist_to_show_ids: dict[str, list[str]]) -> str:
    for artist in sorted(artist_to_show_ids.keys(), key=len, reverse=True):
        if artist and artist in text:
            return artist_to_show_ids[artist][0]
    return ""


def crawl_douban_posts(
    shows: pd.DataFrame,
    args: Args,
    crawl_batch_id: str,
    needed_exp_records: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    exp_counter = 0
    total_queries = 0
    source_breakdown = {"douban_direct": 0, "search_engine_fallback": 0}

    artist_to_show_ids: dict[str, list[str]] = {}
    for _, show in shows.iterrows():
        artist = str(show["艺人/团体"] or "").strip()
        show_id = str(show["演出ID"])
        if artist not in artist_to_show_ids:
            artist_to_show_ids[artist] = []
        artist_to_show_ids[artist].append(show_id)

    top_artists = [a for a, _ in shows["艺人/团体"].value_counts().head(30).items()]
    top_cities = [c for c, _ in shows["城市"].value_counts().head(12).items()]
    top_show_pairs = (
        shows[["艺人/团体", "城市", "演出名称"]]
        .drop_duplicates()
        .head(60)
        .to_dict("records")
    )
    query_pool = [
        "演唱会 观演 体验",
        "演唱会 安检 排队",
        "演唱会 散场 拥堵",
        "演唱会 卫生间 厕所",
        "演唱会 闷热 空调",
        "演唱会 网络 信号",
        "演唱会 接驳 地铁",
        "演唱会 打卡 拍照",
        "音乐节 观演 体验",
    ]
    query_pool.extend([f"{a} 演唱会 体验" for a in top_artists])
    query_pool.extend([f"{c} 演唱会 体验 安检 散场 卫生间" for c in top_cities])
    query_pool.extend(
        [
            f"{str(x['艺人/团体'])} {str(x['城市'])} 演唱会 观演 体验"
            for x in top_show_pairs
        ]
    )
    query_pool = [q for q in query_pool if q.strip()]

    for query in query_pool:
        if len(records) >= needed_exp_records and exp_counter >= needed_exp_records:
            break
        for start in [0, 20, 40]:
            if len(records) >= needed_exp_records and exp_counter >= needed_exp_records:
                break
            total_queries += 1
            url = "https://www.douban.com/group/search?cat=1019&q=" + urllib.parse.quote(query) + f"&start={start}"
            status = "success"
            reason = ""
            fetched = 0
            source_type = "douban_direct"
            candidates: list[tuple[str, str, str]] = []
            gate_page = False
            direct_error = ""
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": USER_AGENT, "Referer": "https://www.douban.com/group/"},
                )
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    html_text = resp.read().decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html_text, "lxml")
                nodes = soup.select(".result")
                gate_page = ("请点击下方按钮继续浏览" in html_text) or ('id="sec"' in html_text)
                if nodes and not gate_page:
                    for node in nodes:
                        a = node.select_one("h3 a") or node.select_one("a")
                        if not a:
                            continue
                        href = str(a.get("href") or "").strip()
                        if href.startswith("/"):
                            href = "https://www.douban.com" + href
                        title = clean_comment_text(a.get_text(" ", strip=True))
                        p = node.select_one("p")
                        summary = clean_comment_text(p.get_text(" ", strip=True) if p else "")
                        if href and (title or summary):
                            candidates.append((title, summary, href))
            except Exception as exc:  # noqa: BLE001
                direct_error = str(exc)

            if not candidates:
                source_type = "search_engine_fallback"
                page_no = max(1, (start // 20) + 1)
                so_q = f"site:douban.com {query}"
                so_url = "https://www.so.com/s?q=" + urllib.parse.quote(so_q) + f"&pn={page_no}"
                try:
                    req2 = urllib.request.Request(
                        so_url,
                        headers={"User-Agent": USER_AGENT, "Referer": "https://www.so.com/"},
                    )
                    with urllib.request.urlopen(req2, timeout=args.timeout) as resp2:
                        html2 = resp2.read().decode("utf-8", errors="ignore")
                    soup2 = BeautifulSoup(html2, "lxml")
                    result_nodes = soup2.select(".res-list")
                    if not result_nodes:
                        result_nodes = soup2.select(".result")
                    for node in result_nodes:
                        a = node.select_one("h3 a") or node.select_one("a")
                        if not a:
                            continue
                        href = str(a.get("href") or "").strip()
                        if not href:
                            continue
                        if href.startswith("/"):
                            href = "https://www.so.com" + href
                        title = clean_comment_text(a.get_text(" ", strip=True))
                        s_node = node.select_one(".res-desc") or node.select_one("p") or node.select_one(".mh-detail")
                        summary = clean_comment_text(s_node.get_text(" ", strip=True) if s_node else "")
                        text_probe = f"{title} {summary} {href}".lower()
                        if ("douban" not in text_probe) and ("豆瓣" not in text_probe):
                            continue
                        if title or summary:
                            candidates.append((title, summary, href))
                    if not candidates:
                        status = "no_result"
                        reason = "no_result_after_so360_fallback"
                    else:
                        if direct_error:
                            reason = f"direct_failed_{direct_error[:64]}_use_so360_snippet"
                        else:
                            reason = "direct_blocked_use_so360_snippet" if gate_page else "so360_snippet_fallback"
                    url = so_url
                except Exception as exc:  # noqa: BLE001
                    status = "failed"
                    reason = f"so360_failed_after_direct:{exc}"

            try:
                for title, summary, href in candidates:
                    content = clean_comment_text(f"{title} {summary}")
                    if not content:
                        continue
                    record_id = hashlib.md5((href + "|" + title + "|" + summary).encode("utf-8")).hexdigest()
                    if record_id in seen:
                        continue
                    seen.add(record_id)

                    text_for_match = f"{title} {summary}"
                    show_id = infer_show_id_from_text(text_for_match, artist_to_show_ids)
                    show_name = artist = city = ""
                    concert_date = pd.NaT
                    ticket_platform = ""
                    if show_id:
                        s = shows[shows["演出ID"].astype(str) == show_id].head(1)
                        if len(s) > 0:
                            r = s.iloc[0]
                            show_name = str(r["演出名称"])
                            artist = str(r["艺人/团体"])
                            city = str(r["城市"])
                            concert_date = r["开演日期"]
                            ticket_platform = r.get("售票平台")

                    pub = pd.NaT
                    tm = RE_DATE.search(text_for_match)
                    if tm:
                        ts = tm.group(1).replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-")
                        pub = pd.to_datetime(ts, errors="coerce")

                    records.append(
                        {
                            "show_id": show_id,
                            "show_name": show_name,
                            "artist": artist,
                            "city": city,
                            "concert_date": concert_date,
                            "show_ticket_platform": ticket_platform,
                            "platform": "douban",
                            "source_level": "post",
                            "platform_record_id": record_id,
                            "content_raw": content,
                            "source_url": href,
                            "query_text": query,
                            "title": title,
                            "publish_time": pub,
                            "like_cnt": 0,
                            "reply_cnt": 0,
                            "aid": None,
                            "bvid": "",
                            "video_title": "",
                            "video_pubdate": pd.NaT,
                            "video_comment_total_hint": 0,
                            "mid_hash": "",
                            "crawl_batch_id": crawl_batch_id,
                            "crawl_time": now_local_str(),
                            "parser_version": PARSER_VERSION,
                        }
                    )
                    fetched += 1
                    if any(k in content for k in ["安检", "排队", "散场", "接驳", "卫生间", "厕所", "网络", "信号", "闷热", "空调"]):
                        exp_counter += 1
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                reason = str(exc)

            source_breakdown[source_type] = source_breakdown.get(source_type, 0) + fetched
            audit_rows.append(
                {
                    "show_id": "",
                    "show_name": "",
                    "platform": "douban",
                    "query_text": query,
                    "fetch_status": status,
                    "selected_count": 0,
                    "fetched_records": fetched,
                    "fallback_reason": reason,
                    "raw_source_url": url,
                    "crawl_time": now_local_str(),
                }
            )
            if fetched > 0:
                time.sleep(args.sleep)

    stats = {
        "records": int(len(records)),
        "estimated_experience_records": int(exp_counter),
        "queries": int(total_queries),
        "source_breakdown": source_breakdown,
    }
    return records, audit_rows, stats


def apply_labeling(
    raw_df: pd.DataFrame,
    dimension_keywords: dict[str, list[str]],
    role_map: dict[str, list[str]],
    pos_words: list[str],
    neg_words: list[str],
    negations: list[str],
    experience_terms: list[str],
) -> pd.DataFrame:
    labeled = raw_df.copy()
    labeled["content_clean"] = labeled["content_raw"].map(clean_comment_text)
    exp_hits = labeled["content_clean"].map(lambda t: match_terms(t, experience_terms))
    labeled["experience_hit_terms"] = exp_hits.map(lambda xs: ";".join(xs))
    labeled["is_experience_related"] = exp_hits.map(lambda xs: len(xs) > 0)
    labeled["content_hash"] = labeled["content_clean"].map(sha256_text)
    labeled["dimension_tags"] = labeled["content_clean"].map(lambda t: ";".join(match_dimensions(t, dimension_keywords)))
    sent = labeled["content_clean"].map(lambda t: compute_sentiment(t, pos_words, neg_words, negations))
    labeled["sentiment_label"] = sent.map(lambda x: x[0])
    labeled["sentiment_score"] = sent.map(lambda x: x[1])
    labeled["kano_role_prior"] = labeled["dimension_tags"].map(
        lambda s: map_kano_roles([x for x in str(s).split(";") if x], role_map)
    )
    return labeled


def build_sem_ready(shows: pd.DataFrame, labeled_df: pd.DataFrame, experience_filter: bool) -> pd.DataFrame:
    sem_source = labeled_df[labeled_df["is_experience_related"] == True].copy() if experience_filter else labeled_df.copy()  # noqa: E712
    rows: list[dict[str, Any]] = []
    for _, show in shows.iterrows():
        show_id = str(show["演出ID"])
        g_raw = labeled_df[labeled_df["show_id"].astype(str) == show_id]
        g = sem_source[sem_source["show_id"].astype(str) == show_id]
        total_comments = int(len(g))
        row: dict[str, Any] = {
            "show_id": show_id,
            "show_name": show["演出名称"],
            "artist": show["艺人/团体"],
            "city": show["城市"],
            "venue": show["场馆名称"],
            "concert_date": show["开演日期"],
            "show_ticket_platform": show["售票平台"],
            "total_comments_raw": int(len(g_raw)),
            "total_comments": total_comments,
            "data_gap": int(total_comments == 0),
        }
        tag_lists = g["dimension_tags"].fillna("").astype(str).str.split(";") if total_comments else pd.Series([], dtype="object")
        for dim in ALL_DIMS:
            if total_comments == 0:
                mention_cnt = 0
                mention_ratio = 0.0
                neg_ratio = 0.0
                mean_sentiment = 0.0
            else:
                mask = tag_lists.map(lambda tags, target=dim: target in tags if isinstance(tags, list) else False)
                mention_cnt = int(mask.sum())
                mention_ratio = float(mention_cnt / total_comments)
                if mention_cnt > 0:
                    focus = g.loc[mask]
                    neg_ratio = float((focus["sentiment_label"] == "neg").mean())
                    mean_sentiment = float(focus["sentiment_score"].mean())
                else:
                    neg_ratio = 0.0
                    mean_sentiment = 0.0
            row[f"{dim}_mention_cnt"] = mention_cnt
            row[f"{dim}_mention_ratio"] = round(mention_ratio, 6)
            row[f"{dim}_neg_ratio"] = round(neg_ratio, 6)
            row[f"{dim}_mean_sentiment"] = round(mean_sentiment, 6)
        sat_cnt = row["satisfaction_mention_cnt"]
        rec_cnt = row["recommendation_mention_cnt"]
        row["sat_pos_ratio"] = (
            round(float((g[g["dimension_tags"].astype(str).str.contains(r"(?:^|;)satisfaction(?:$|;)", regex=True)]["sentiment_label"] == "pos").mean()), 6)
            if sat_cnt > 0
            else 0.0
        )
        row["rec_pos_ratio"] = (
            round(float((g[g["dimension_tags"].astype(str).str.contains(r"(?:^|;)recommendation(?:$|;)", regex=True)]["sentiment_label"] == "pos").mean()), 6)
            if rec_cnt > 0
            else 0.0
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["concert_date", "show_id"]).reset_index(drop=True)


def build_calibration_sample(labeled_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    src = labeled_df[labeled_df["content_clean"].astype(str).str.len() > 0].copy()
    bilibili = src[src["platform"] == "bilibili"]
    others = src[src["platform"] != "bilibili"]
    n_other = min(len(others), sample_size // 2)
    n_bili = min(len(bilibili), sample_size - n_other)
    blocks = []
    if n_other > 0:
        blocks.append(others.sample(n=n_other, random_state=RANDOM_SEED))
    if n_bili > 0:
        blocks.append(bilibili.sample(n=n_bili, random_state=RANDOM_SEED + 1))
    cur = pd.concat(blocks, ignore_index=True) if blocks else src.head(0).copy()
    if len(cur) < sample_size and len(src) > len(cur):
        need = min(sample_size - len(cur), len(src) - len(cur))
        extra = src.sample(n=need, random_state=RANDOM_SEED + 2)
        cur = pd.concat([cur, extra], ignore_index=True)
    cur = cur.head(sample_size).copy()
    return pd.DataFrame(
        {
            "platform": cur["platform"],
            "platform_record_id": cur["platform_record_id"],
            "text": cur["content_clean"],
            "auto_is_experience": cur["is_experience_related"],
            "auto_dim_tags": cur["dimension_tags"],
            "auto_sentiment": cur["sentiment_label"],
            "human_is_experience": "",
            "human_dim_tags": "",
            "human_sentiment": "",
            "reviewer": "",
            "review_time": "",
        }
    )


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def build_calibration_report(calib_df: pd.DataFrame) -> pd.DataFrame:
    valid = calib_df[
        calib_df["human_is_experience"].astype(str).str.strip().isin(["0", "1", "true", "false", "True", "False"])
    ]
    if len(valid) == 0:
        return pd.DataFrame(
            [
                {"metric": "human_labeled_rows", "value": 0, "note": "待人工标注后自动更新精度"},
                {"metric": "experience_precision", "value": None, "note": "N/A"},
                {"metric": "experience_recall", "value": None, "note": "N/A"},
                {"metric": "experience_f1", "value": None, "note": "N/A"},
                {"metric": "dimension_precision", "value": None, "note": "N/A"},
                {"metric": "dimension_recall", "value": None, "note": "N/A"},
                {"metric": "dimension_f1", "value": None, "note": "N/A"},
            ]
        )

    y_true = valid["human_is_experience"].astype(str).str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0}).fillna(0).astype(int).tolist()
    y_pred = valid["auto_is_experience"].astype(bool).astype(int).tolist()
    exp_m = compute_binary_metrics(y_true, y_pred)
    tp = fp = fn = 0
    for _, row in valid.iterrows():
        auto_set = {x for x in str(row["auto_dim_tags"]).split(";") if x}
        human_set = {x for x in str(row["human_dim_tags"]).split(";") if x}
        tp += len(auto_set & human_set)
        fp += len(auto_set - human_set)
        fn += len(human_set - auto_set)
    dim_p = tp / (tp + fp) if (tp + fp) else 0.0
    dim_r = tp / (tp + fn) if (tp + fn) else 0.0
    dim_f1 = 2 * dim_p * dim_r / (dim_p + dim_r) if (dim_p + dim_r) else 0.0
    return pd.DataFrame(
        [
            {"metric": "human_labeled_rows", "value": int(len(valid)), "note": ""},
            {"metric": "experience_precision", "value": round(exp_m["precision"], 6), "note": ""},
            {"metric": "experience_recall", "value": round(exp_m["recall"], 6), "note": ""},
            {"metric": "experience_f1", "value": round(exp_m["f1"], 6), "note": ""},
            {"metric": "dimension_precision", "value": round(dim_p, 6), "note": ""},
            {"metric": "dimension_recall", "value": round(dim_r, 6), "note": ""},
            {"metric": "dimension_f1", "value": round(dim_f1, 6), "note": ""},
        ]
    )


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out[columns]


def load_cached_new_platform_records(
    out_root: Path,
    current_out_dir: Path,
    existing_keys: set[tuple[str, str]],
    max_records: int = 2000,
) -> list[dict[str, Any]]:
    files = sorted(
        out_root.glob("*/concert_comments_raw_multiplatform_*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    loaded: list[dict[str, Any]] = []
    for path in files:
        if current_out_dir in path.parents:
            continue
        try:
            df = pd.read_excel(path)
        except Exception:  # noqa: BLE001
            continue
        if len(df) == 0 or "platform" not in df.columns or "platform_record_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            platform = str(row.get("platform", "")).strip().lower()
            if platform not in {"douban", "xiaohongshu"}:
                continue
            pid = str(row.get("platform_record_id", "")).strip()
            if not pid:
                continue
            key = (platform, pid)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            loaded.append(
                {
                    "show_id": str(row.get("show_id", "")),
                    "show_name": str(row.get("show_name", "")),
                    "artist": str(row.get("artist", "")),
                    "city": str(row.get("city", "")),
                    "concert_date": row.get("concert_date", pd.NaT),
                    "show_ticket_platform": str(row.get("show_ticket_platform", "")),
                    "platform": platform,
                    "source_level": str(row.get("source_level", "post" if platform == "douban" else "note")),
                    "platform_record_id": pid,
                    "content_raw": str(row.get("content_raw", "")),
                    "source_url": str(row.get("source_url", "")),
                    "query_text": str(row.get("query_text", "cache_reuse")),
                    "title": str(row.get("title", "")),
                    "publish_time": row.get("publish_time", pd.NaT),
                    "like_cnt": int(parse_cn_numeric(row.get("like_cnt", 0))),
                    "reply_cnt": int(parse_cn_numeric(row.get("reply_cnt", 0))),
                    "aid": row.get("aid", None),
                    "bvid": str(row.get("bvid", "")),
                    "video_title": str(row.get("video_title", "")),
                    "video_pubdate": row.get("video_pubdate", pd.NaT),
                    "video_comment_total_hint": int(parse_cn_numeric(row.get("video_comment_total_hint", 0))),
                    "mid_hash": str(row.get("mid_hash", "")),
                    "crawl_batch_id": "cache_reuse",
                    "crawl_time": now_local_str(),
                    "parser_version": f"{PARSER_VERSION}+cache",
                }
            )
            if len(loaded) >= max_records:
                return loaded
    return loaded


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    args = parse_args()
    random.seed(RANDOM_SEED)

    input_path = Path(args.input)
    out_root = Path(args.out_dir)
    out_dir = out_root / args.out_version
    old_dir = Path(args.old_comment_dir)
    dict_path = Path(args.dictionary)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if not dict_path.exists():
        raise FileNotFoundError(f"dictionary file not found: {dict_path}")

    with dict_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    dimension_keywords: dict[str, list[str]] = cfg.get("dimensions", {})
    role_map: dict[str, list[str]] = cfg.get("kano_role_prior", {})
    sentiment_cfg: dict[str, list[str]] = cfg.get("sentiment", {})
    exp_extra_terms: list[str] = cfg.get("experience_filter_terms", [])
    pos_words = sentiment_cfg.get("positive", [])
    neg_words = sentiment_cfg.get("negative", [])
    negations = sentiment_cfg.get("negations", [])
    experience_terms = build_experience_terms(dimension_keywords, exp_extra_terms)

    baseline = load_baseline(old_dir)

    print("[1/8] load shows")
    shows = normalize_concert_df(input_path, args.years)
    print(f"      shows in scope: {len(shows)}")

    crawl_batch_id = datetime.now().strftime("comment_v2_%Y%m%d_%H%M%S")
    all_records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    platform_stats: dict[str, Any] = {}

    if "bilibili" in args.platforms:
        print("[2/8] crawl bilibili")
        rec, aud, st = crawl_bilibili(shows, args, crawl_batch_id)
        all_records.extend(rec)
        audit_rows.extend(aud)
        platform_stats["bilibili"] = st

    xhs_records: list[dict[str, Any]] = []
    xhs_stats: dict[str, Any] = {"available": False, "records": 0, "queries": 0}
    if "xiaohongshu" in args.platforms:
        print("[3/8] crawl xiaohongshu (note-level)")
        xhs_records, xhs_audit, xhs_stats = crawl_xiaohongshu_notes(shows, args, crawl_batch_id)
        all_records.extend(xhs_records)
        audit_rows.extend(xhs_audit)
        platform_stats["xiaohongshu"] = xhs_stats

    print("[4/8] decide fallback")
    temp_df = pd.DataFrame(all_records)
    if len(temp_df) > 0:
        temp_df["content_clean"] = temp_df["content_raw"].map(clean_comment_text)
        temp_df["is_experience_related"] = temp_df["content_clean"].map(lambda t: len(match_terms(t, experience_terms)) > 0)
        xhs_exp = int(((temp_df["platform"] == "xiaohongshu") & (temp_df["is_experience_related"] == True)).sum())  # noqa: E712
    else:
        xhs_exp = 0
    need_douban = (
        "douban" in args.platforms
        and (("xiaohongshu" in args.platforms and xhs_exp < args.xhs_min_records) or ("xiaohongshu" not in args.platforms))
    )
    if need_douban:
        need = max(0, args.xhs_min_records - xhs_exp)
        print(f"[5/8] fallback douban enabled, need >= {need}")
        d_rec, d_aud, d_st = crawl_douban_posts(shows, args, crawl_batch_id, needed_exp_records=max(need, 200))
        all_records.extend(d_rec)
        audit_rows.extend(d_aud)
        platform_stats["douban"] = d_st
    else:
        print("[5/8] douban fallback skipped")

    new_platform_live = sum(1 for r in all_records if str(r.get("platform", "")).strip().lower() != "bilibili")
    if new_platform_live < args.xhs_min_records:
        existing_keys: set[tuple[str, str]] = set()
        for r in all_records:
            p = str(r.get("platform", "")).strip().lower()
            pid = str(r.get("platform_record_id", "")).strip()
            if p and pid:
                existing_keys.add((p, pid))
        cache_records = load_cached_new_platform_records(
            out_root=out_root,
            current_out_dir=out_dir,
            existing_keys=existing_keys,
            max_records=2000,
        )
        if cache_records:
            all_records.extend(cache_records)
            platform_stats["cache_reuse"] = {"records": int(len(cache_records))}
            audit_rows.append(
                {
                    "show_id": "",
                    "show_name": "",
                    "platform": "douban",
                    "query_text": "cache_reuse",
                    "fetch_status": "cache_reuse",
                    "selected_count": 0,
                    "fetched_records": int(len(cache_records)),
                    "fallback_reason": "live_platform_risk_control_use_local_history",
                    "raw_source_url": str(out_root),
                    "crawl_time": now_local_str(),
                }
            )

    print("[6/8] build raw/labeled/sem")
    raw_df = pd.DataFrame(all_records)
    if raw_df.empty:
        raise RuntimeError("no records crawled from all platforms")
    raw_df["concert_date"] = pd.to_datetime(raw_df["concert_date"], errors="coerce")
    raw_df["publish_time"] = pd.to_datetime(raw_df["publish_time"], errors="coerce")
    raw_df["video_pubdate"] = pd.to_datetime(raw_df["video_pubdate"], errors="coerce")
    raw_df = raw_df.sort_values(["platform", "show_id", "platform_record_id"]).reset_index(drop=True)

    labeled_df = apply_labeling(
        raw_df=raw_df,
        dimension_keywords=dimension_keywords,
        role_map=role_map,
        pos_words=pos_words,
        neg_words=neg_words,
        negations=negations,
        experience_terms=experience_terms,
    )
    sem_df = build_sem_ready(shows, labeled_df, experience_filter=args.experience_filter)

    print("[7/8] build calibration")
    calib_sample = build_calibration_sample(labeled_df, sample_size=args.calibration_size)
    calib_report = build_calibration_report(calib_sample)

    print("[8/8] write outputs")
    suffix = args.out_version
    raw_file = out_dir / f"concert_comments_raw_multiplatform_{suffix}.xlsx"
    labeled_file = out_dir / f"concert_comments_labeled_multiplatform_{suffix}.xlsx"
    sem_file = out_dir / f"concert_sem_ready_metrics_multiplatform_{suffix}.xlsx"
    audit_file = out_dir / f"concert_platform_fetch_audit_{suffix}.xlsx"
    calib_sample_file = out_dir / f"calibration_sample_200_{suffix}.xlsx"
    calib_report_file = out_dir / f"calibration_report_{suffix}.xlsx"
    manifest_file = out_dir / f"run_manifest_{suffix}.json"

    raw_columns = [
        "show_id", "show_name", "artist", "city", "concert_date", "show_ticket_platform",
        "platform", "source_level", "platform_record_id", "content_raw", "source_url", "query_text",
        "title", "publish_time", "like_cnt", "reply_cnt", "aid", "bvid", "video_title", "video_pubdate",
        "video_comment_total_hint", "content_hash", "is_experience_related", "experience_hit_terms",
        "crawl_batch_id", "crawl_time", "parser_version",
    ]
    raw_out = ensure_columns(labeled_df, raw_columns)
    raw_out.to_excel(raw_file, index=False)
    labeled_df.to_excel(labeled_file, index=False)
    sem_df.to_excel(sem_file, index=False)
    pd.DataFrame(audit_rows).to_excel(audit_file, index=False)
    calib_sample.to_excel(calib_sample_file, index=False)
    calib_report.to_excel(calib_report_file, index=False)

    old_files_after: dict[str, Any] = {}
    for name, meta in baseline.get("files", {}).items():
        p = old_dir / name
        if p.exists():
            old_files_after[name] = {"md5": file_md5(p), "size": int(p.stat().st_size)}

    any_dim_old = float(baseline.get("metrics", {}).get("old_any_dim_ratio", 0.0))
    any_dim_new = float(labeled_df["dimension_tags"].fillna("").astype(str).ne("").mean())
    any_dim_multiplier = (any_dim_new / any_dim_old) if any_dim_old > 0 else None
    platform_unique = int(labeled_df["platform"].nunique())
    new_platform_records = int((labeled_df["platform"] != "bilibili").sum())
    dup_keys = int(labeled_df.duplicated(subset=["platform", "platform_record_id"]).sum())
    audit_df = pd.DataFrame(audit_rows)
    show_coverage = int(audit_df[audit_df["show_id"].astype(str).str.len() > 0]["show_id"].nunique()) if len(audit_df) else 0

    checks = {
        "old_md5_unchanged": all(
            old_files_after.get(name, {}).get("md5") == meta.get("md5")
            for name, meta in baseline.get("files", {}).items()
        ),
        "platform_count_ge_2": platform_unique >= 2,
        "new_platform_records_ge_200": new_platform_records >= 200,
        "any_dim_ratio_ge_5pct": any_dim_new >= 0.05,
        "any_dim_ratio_ge_3x_old": (any_dim_multiplier is not None and any_dim_multiplier >= 3.0),
        "raw_unique_platform_record_id": dup_keys == 0,
        "sem_unique_show_id": bool(sem_df["show_id"].is_unique),
        "sem_rows_eq_85": int(len(sem_df)) == 85,
        "audit_show_coverage_eq_85": show_coverage == 85,
        "calibration_rows_eq_200": int(len(calib_sample)) == args.calibration_size,
    }

    manifest = {
        "run_time": now_local_str(),
        "crawl_batch_id": crawl_batch_id,
        "parser_version": PARSER_VERSION,
        "args": asdict(args),
        "input_file": str(input_path),
        "out_dir": str(out_dir),
        "baseline": baseline,
        "old_files_after": old_files_after,
        "platform_stats": platform_stats,
        "new_metrics": {
            "raw_rows": int(len(raw_out)),
            "labeled_rows": int(len(labeled_df)),
            "sem_rows": int(len(sem_df)),
            "audit_rows": int(len(audit_df)),
            "platform_unique": platform_unique,
            "new_platform_records": new_platform_records,
            "any_dim_ratio_old": any_dim_old,
            "any_dim_ratio_new": any_dim_new,
            "any_dim_multiplier": any_dim_multiplier,
            "sem_nonzero_shows": int((sem_df["total_comments"] > 0).sum()),
        },
        "checks": checks,
        "outputs": {
            "raw": str(raw_file),
            "labeled": str(labeled_file),
            "sem_ready": str(sem_file),
            "audit": str(audit_file),
            "calibration_sample": str(calib_sample_file),
            "calibration_report": str(calib_report_file),
        },
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done.")
    print(f"raw: {raw_file}")
    print(f"labeled: {labeled_file}")
    print(f"sem: {sem_file}")
    print(f"audit: {audit_file}")
    print(f"calibration_sample: {calib_sample_file}")
    print(f"calibration_report: {calib_report_file}")
    print(f"manifest: {manifest_file}")
    print("checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
