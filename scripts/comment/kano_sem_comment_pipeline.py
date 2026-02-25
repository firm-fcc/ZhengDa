#!/usr/bin/env python3
"""Kano+SEM external evidence pipeline using public Bilibili comments."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


MIXIN_KEY_ENC_TAB = [
    46,
    47,
    18,
    2,
    53,
    8,
    23,
    32,
    15,
    50,
    10,
    31,
    58,
    3,
    45,
    35,
    27,
    43,
    5,
    49,
    33,
    9,
    42,
    19,
    29,
    28,
    14,
    39,
    12,
    38,
    41,
    13,
    37,
    48,
    7,
    16,
    24,
    55,
    40,
    61,
    26,
    17,
    0,
    1,
    60,
    51,
    30,
    4,
    22,
    25,
    54,
    21,
    56,
    59,
    6,
    63,
    57,
    62,
    11,
    36,
    20,
    34,
    44,
    52,
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
MID_HASH_SALT = "ZHENGDA_KANO_SEM_FIXED_SALT_2026"
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
SHOW_NAME_STOPWORDS = {
    "演唱会",
    "巡回",
    "世界",
    "全国",
    "live",
    "LIVE",
    "巡演",
    "音乐会",
    "武汉",
    "宜昌",
    "襄阳",
    "站",
}
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
RE_MENTION = re.compile(r"@[\w\-\u4e00-\u9fff]+")
RE_BRACKET = re.compile(r"\[[^\]]+\]")
RE_WHITESPACE = re.compile(r"\s+")
RE_TOKEN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
RE_EMOJI = re.compile(r"[\U00010000-\U0010ffff]")
RE_WBI_FILTER = re.compile(r"[!'()*]")


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


class BilibiliClient:
    """Minimal client for Bilibili WBI signed endpoints."""

    def __init__(self, retry: int, sleep: float, timeout: int):
        self.retry = retry
        self.sleep = sleep
        self.timeout = timeout
        self._opener = urllib.request.build_opener()
        self._wbi_key: str | None = None

    def _request_json(self, url: str, referer: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retry):
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
                with self._opener.open(req, timeout=self.timeout) as response:
                    payload = response.read().decode("utf-8", errors="ignore")
                data = json.loads(payload)
                return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait_s = self.sleep * (2**attempt)
                time.sleep(wait_s)
        raise RuntimeError(f"request failed: {url} :: {last_error}") from last_error

    def _ensure_wbi_key(self) -> str:
        if self._wbi_key:
            return self._wbi_key
        nav = self._request_json(
            "https://api.bilibili.com/x/web-interface/nav",
            referer="https://www.bilibili.com/",
        )
        wbi_img = nav.get("data", {}).get("wbi_img", {})
        img_key = (wbi_img.get("img_url") or "").rsplit("/", 1)[-1].split(".")[0]
        sub_key = (wbi_img.get("sub_url") or "").rsplit("/", 1)[-1].split(".")[0]
        if not img_key or not sub_key:
            raise RuntimeError(f"nav endpoint failed: {nav.get('message')}")
        merged = img_key + sub_key
        self._wbi_key = "".join(merged[i] for i in MIXIN_KEY_ENC_TAB)[:32]
        if not self._wbi_key:
            raise RuntimeError("failed to build wbi key")
        return self._wbi_key

    def _sign_wbi(self, params: dict[str, Any]) -> str:
        wbi_key = self._ensure_wbi_key()
        signed = {k: RE_WBI_FILTER.sub("", str(v)) for k, v in params.items()}
        signed["wts"] = str(int(time.time()))
        query = urllib.parse.urlencode(sorted(signed.items()))
        w_rid = hashlib.md5((query + wbi_key).encode("utf-8")).hexdigest()
        return f"{query}&w_rid={w_rid}"

    def search_videos(self, keyword: str, page: int = 1) -> list[dict[str, Any]]:
        params = {
            "search_type": "video",
            "keyword": keyword,
            "page": page,
        }
        query = self._sign_wbi(params)
        url = f"https://api.bilibili.com/x/web-interface/wbi/search/type?{query}"
        data = self._request_json(url, referer="https://search.bilibili.com/")
        if int(data.get("code", -1)) != 0:
            raise RuntimeError(f"search failed: {data.get('message')}")
        return data.get("data", {}).get("result", []) or []

    def fetch_comments(
        self,
        aid: int,
        bvid: str,
        max_comments: int,
    ) -> tuple[list[dict[str, Any]], int]:
        comments: list[dict[str, Any]] = []
        all_count = 0
        next_cursor: int | str = 0
        seen_ids: set[str] = set()
        while True:
            params = {
                "oid": aid,
                "type": 1,
                "mode": 3,
                "next": next_cursor,
                "ps": 20,
            }
            query = self._sign_wbi(params)
            url = f"https://api.bilibili.com/x/v2/reply/wbi/main?{query}"
            data = self._request_json(url, referer=f"https://www.bilibili.com/video/{bvid}")
            if int(data.get("code", -1)) != 0:
                break
            payload = data.get("data", {}) or {}
            cursor = payload.get("cursor", {}) or {}
            if "all_count" in cursor and cursor.get("all_count") is not None:
                try:
                    all_count = int(cursor.get("all_count"))
                except Exception:  # noqa: BLE001
                    all_count = 0
            page_replies = payload.get("replies", []) or []
            for reply in page_replies:
                rid = str(reply.get("rpid_str") or reply.get("rpid") or "")
                if not rid or rid in seen_ids:
                    continue
                seen_ids.add(rid)
                comments.append(reply)
                if len(comments) >= max_comments:
                    return comments, all_count
            if bool(cursor.get("is_end")):
                break
            new_next = cursor.get("next")
            if new_next is None or str(new_next) == str(next_cursor):
                break
            next_cursor = new_next
            time.sleep(self.sleep)
        return comments, all_count


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input concert list xlsx")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--max-videos", type=int, default=3)
    parser.add_argument("--max-comments-per-video", type=int, default=200)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--dict",
        dest="dictionary",
        default=str(Path(__file__).with_name("kano_sem_dimension_dict.json")),
    )
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
    )


def normalize_concert_df(input_path: Path, years: list[int]) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    required_cols = [
        "演出ID",
        "演出名称",
        "艺人/团体",
        "城市",
        "场馆名称",
        "开演日期",
        "开演时间",
        "演出类型",
        "最低票价",
        "最高票价",
        "票价档位数",
        "开票时间",
        "售罄状态",
        "售票平台",
        "页面URL",
        "爬取时间",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df.copy()
    df["开演日期"] = pd.to_datetime(df["开演日期"], errors="coerce")
    df["开票时间"] = pd.to_datetime(df["开票时间"], errors="coerce")
    df["爬取时间"] = pd.to_datetime(df["爬取时间"], errors="coerce")
    df["开演年份"] = df["开演日期"].dt.year
    df["最低票价"] = pd.to_numeric(df["最低票价"], errors="coerce")
    df["最高票价"] = pd.to_numeric(df["最高票价"], errors="coerce")
    df["票价档位数"] = pd.to_numeric(df["票价档位数"], errors="coerce")
    df["售罄状态"] = (
        df["售罄状态"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )
    df = df[df["开演年份"].isin(years)].copy()
    df = df.sort_values(["开演日期", "演出ID"]).reset_index(drop=True)
    return df


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
    t = RE_WHITESPACE.sub(" ", t)
    return t.strip()


def parse_cn_numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"--", "-", "None", "null", "nan"}:
        return 0.0
    text = text.rstrip("+")
    multiplier = 1.0
    if text.endswith("万"):
        multiplier = 10000.0
        text = text[:-1]
    elif text.endswith("亿"):
        multiplier = 100000000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except Exception:  # noqa: BLE001
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if not numbers:
            return 0.0
        return float(numbers[0]) * multiplier


def to_datetime_from_unix(timestamp: Any) -> pd.Timestamp | pd.NaT:
    try:
        ts_int = int(timestamp)
        if ts_int <= 0:
            return pd.NaT
        return pd.to_datetime(ts_int, unit="s", utc=True).tz_convert("Asia/Shanghai").tz_localize(None)
    except Exception:  # noqa: BLE001
        return pd.NaT


def extract_show_core_tokens(show_name: str) -> list[str]:
    tokens = []
    for token in RE_TOKEN.findall(str(show_name or "")):
        token = token.strip()
        if not token or len(token) <= 1:
            continue
        if token in SHOW_NAME_STOPWORDS:
            continue
        tokens.append(token)
    dedup_tokens: list[str] = []
    for token in tokens:
        if token not in dedup_tokens:
            dedup_tokens.append(token)
    return dedup_tokens


def compute_candidate_score(
    candidate: dict[str, Any],
    show: pd.Series,
    core_tokens: list[str],
) -> dict[str, Any]:
    title = clean_html_text(candidate.get("title", ""))
    desc = clean_html_text(candidate.get("description", ""))
    tags = clean_html_text(candidate.get("tag", ""))
    joined_text = f"{title} {desc} {tags}"
    text_lower = joined_text.lower()

    artist = str(show["艺人/团体"] or "")
    city = str(show["城市"] or "")
    show_date = show["开演日期"]

    artist_hit_score = 5 if artist and artist in joined_text else 0
    city_hit_score = 3 if city and city in joined_text else 0
    core_hit_count = 0
    for token in core_tokens:
        if token.lower() in text_lower:
            core_hit_count += 1
    core_token_hit_score = min(core_hit_count * 2, 6)

    pubdate = to_datetime_from_unix(candidate.get("pubdate"))
    date_score = 0
    date_gap_days: float | None = None
    if pd.notna(pubdate) and pd.notna(show_date):
        date_gap_days = float(abs((pubdate.normalize() - show_date.normalize()).days))
        if date_gap_days <= 45:
            date_score = 2
        elif date_gap_days <= 120:
            date_score = 1

    engagement = parse_cn_numeric(candidate.get("play")) + parse_cn_numeric(candidate.get("review"))
    engagement_score = 0.2 * math.log1p(max(0.0, engagement))
    total_score = artist_hit_score + city_hit_score + core_token_hit_score + date_score + engagement_score

    return {
        "title_clean": title,
        "description_clean": desc,
        "pubdate": pubdate,
        "artist_hit_score": artist_hit_score,
        "city_hit_score": city_hit_score,
        "core_token_hit_score": core_token_hit_score,
        "date_score": date_score,
        "engagement_score": round(engagement_score, 6),
        "total_score": round(total_score, 6),
        "date_gap_days": date_gap_days,
        "core_hit_count": core_hit_count,
    }


def match_dimensions(text: str, dimension_keywords: dict[str, list[str]]) -> list[str]:
    tags: list[str] = []
    low_text = text.lower()
    for dim, keywords in dimension_keywords.items():
        for keyword in keywords:
            kw = str(keyword).strip()
            if not kw:
                continue
            if kw.lower() in low_text:
                tags.append(dim)
                break
    return tags


def hash_mid(mid: Any) -> str:
    mid_str = str(mid or "")
    return hashlib.sha256((mid_str + MID_HASH_SALT).encode("utf-8")).hexdigest()


def compute_sentiment(
    text: str,
    pos_words: list[str],
    neg_words: list[str],
    negations: list[str],
) -> tuple[str, float]:
    normalized = text.lower()
    pos_count = 0
    neg_count = 0

    def is_negated(idx: int) -> bool:
        start = max(0, idx - 4)
        window = normalized[start:idx]
        return any(n in window for n in negations)

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
    if score > 0.1:
        label = "pos"
    elif score < -0.1:
        label = "neg"
    else:
        label = "neu"
    return label, round(float(score), 6)


def map_kano_roles(tags: list[str], role_map: dict[str, list[str]]) -> str:
    results: list[str] = []
    tag_set = set(tags)
    for role in ["must_be", "one_dim", "attractive"]:
        dims = set(role_map.get(role, []))
        if tag_set.intersection(dims):
            results.append(role)
    return ";".join(results)


def run_dimension_unit_cases(
    dimension_keywords: dict[str, list[str]],
    pos_words: list[str],
    neg_words: list[str],
    negations: list[str],
) -> tuple[bool, list[dict[str, Any]]]:
    cases = [
        ("security_check", "今天安检排队太久，入场很慢"),
        ("free_supply", "场馆发放了饮用水和雨衣，补给很及时"),
        ("controlled_lightstick", "统一荧光棒效果不错，灯牌很整齐"),
        ("network_signal", "场馆内网络没信号，wifi也连不上"),
        ("temperature_control", "空调太弱，场内闷热"),
        ("hygiene", "卫生间比较干净，没有异味"),
        ("photo_checkin", "打卡拍照装置很出片，背景墙好看"),
        ("shuttle_service", "散场接驳和地铁疏散很顺畅"),
        ("satisfaction", "整体非常满意，值回票价"),
        ("recommendation", "我会推荐朋友来，下次还会来"),
    ]
    results: list[dict[str, Any]] = []
    passed = True
    for expected_dim, text in cases:
        tags = match_dimensions(text, dimension_keywords)
        sentiment_label, sentiment_score = compute_sentiment(text, pos_words, neg_words, negations)
        hit = expected_dim in tags
        if not hit:
            passed = False
        results.append(
            {
                "expected_dim": expected_dim,
                "text": text,
                "hit": hit,
                "tags": ";".join(tags),
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
            }
        )
    return passed, results


def build_sem_ready(
    shows_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    matched_video_count: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, show in shows_df.iterrows():
        show_id = str(show["演出ID"])
        g = labeled_df[labeled_df["show_id"] == show_id] if not labeled_df.empty else labeled_df
        total_comments = int(len(g))
        row: dict[str, Any] = {
            "show_id": show_id,
            "show_name": show["演出名称"],
            "artist": show["艺人/团体"],
            "city": show["城市"],
            "venue": show["场馆名称"],
            "concert_date": show["开演日期"],
            "platform": show["售票平台"],
            "ticket_min": show["最低票价"],
            "ticket_max": show["最高票价"],
            "matched_videos_cnt": int(matched_video_count.get(show_id, 0)),
            "total_comments": total_comments,
        }
        if total_comments > 0:
            tag_lists = g["dimension_tags"].fillna("").astype(str).str.split(";")
        else:
            tag_lists = pd.Series([], dtype="object")

        for dim in ALL_DIMS:
            if total_comments == 0:
                mention_cnt = 0
                mention_ratio = 0.0
                neg_ratio = 0.0
                mean_sentiment = 0.0
            else:
                mask = tag_lists.apply(lambda tags, target=dim: target in tags if isinstance(tags, list) else False)
                mention_cnt = int(mask.sum())
                mention_ratio = float(mention_cnt / total_comments) if total_comments else 0.0
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
            round(
                float(
                    (
                        g[
                            g["dimension_tags"]
                            .fillna("")
                            .astype(str)
                            .str.contains(r"(?:^|;)satisfaction(?:$|;)", regex=True)
                        ]["sentiment_label"]
                        == "pos"
                    ).mean()
                ),
                6,
            )
            if sat_cnt > 0
            else 0.0
        )
        row["rec_pos_ratio"] = (
            round(
                float(
                    (
                        g[
                            g["dimension_tags"]
                            .fillna("")
                            .astype(str)
                            .str.contains(r"(?:^|;)recommendation(?:$|;)", regex=True)
                        ]["sentiment_label"]
                        == "pos"
                    ).mean()
                ),
                6,
            )
            if rec_cnt > 0
            else 0.0
        )
        row["data_gap"] = int(row["matched_videos_cnt"] == 0 or total_comments == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
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
    pos_words = sentiment_cfg.get("positive", [])
    neg_words = sentiment_cfg.get("negative", [])
    negations = sentiment_cfg.get("negations", [])

    print(f"[1/7] reading and normalizing concerts from: {input_path}")
    shows_df = normalize_concert_df(input_path, args.years)
    total_shows = len(shows_df)
    if total_shows == 0:
        raise RuntimeError("no shows left after year filtering")
    print(f"      shows in scope: {total_shows}")

    print("[2/7] initializing bilibili client and signed search")
    client = BilibiliClient(retry=args.retry, sleep=args.sleep, timeout=args.timeout)
    _ = client._ensure_wbi_key()

    raw_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    selected_video_rows: list[dict[str, Any]] = []
    seen_comment_ids_global: set[str] = set()
    matched_video_count: dict[str, int] = {}
    search_success_count = 0

    print("[3/7] searching candidates, scoring, selecting videos, and fetching comments")
    for idx, show in shows_df.iterrows():
        show_id = str(show["演出ID"])
        show_name = str(show["演出名称"])
        artist = str(show["艺人/团体"])
        city = str(show["城市"])
        query = f"{artist} {show_name} {city} 演唱会"
        core_tokens = extract_show_core_tokens(show_name)
        search_error = None

        try:
            candidates = client.search_videos(query, page=1)
            search_success_count += 1
        except Exception as exc:  # noqa: BLE001
            candidates = []
            search_error = str(exc)

        scored_candidates: list[dict[str, Any]] = []
        for rank, cand in enumerate(candidates[:20], start=1):
            aid = cand.get("aid")
            bvid = cand.get("bvid")
            aid_int = int(aid) if str(aid).isdigit() else None
            scores = compute_candidate_score(cand, show, core_tokens)
            row = {
                "show_id": show_id,
                "show_name": show_name,
                "artist": artist,
                "city": city,
                "concert_date": show["开演日期"],
                "query": query,
                "candidate_rank_api": rank,
                "aid": aid_int,
                "bvid": str(bvid or ""),
                "video_title": scores["title_clean"],
                "video_pubdate": scores["pubdate"],
                "artist_hit_score": scores["artist_hit_score"],
                "city_hit_score": scores["city_hit_score"],
                "core_token_hit_score": scores["core_token_hit_score"],
                "date_score": scores["date_score"],
                "engagement_score": scores["engagement_score"],
                "total_score": scores["total_score"],
                "date_gap_days": scores["date_gap_days"],
                "core_hit_count": scores["core_hit_count"],
                "selected": 0,
                "reject_reason": "",
            }
            scored_candidates.append(row)

        scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)

        selected: list[dict[str, Any]] = []
        used_bvid: set[str] = set()
        for cand in scored_candidates:
            reason = ""
            aid = cand.get("aid")
            bvid = str(cand.get("bvid") or "")
            if not aid or not bvid:
                reason = "missing_aid_or_bvid"
            elif cand["artist_hit_score"] <= 0:
                reason = "artist_not_matched"
            elif cand["date_score"] <= 0 and cand["city_hit_score"] <= 0:
                reason = "date_and_city_not_matched"
            elif cand["total_score"] < 7:
                reason = "low_total_score"
            elif bvid in used_bvid:
                reason = "duplicate_bvid"
            else:
                cand["selected"] = 1
                selected.append(cand)
                used_bvid.add(bvid)
            cand["reject_reason"] = reason

        if not scored_candidates:
            audit_rows.append(
                {
                    "show_id": show_id,
                    "show_name": show_name,
                    "artist": artist,
                    "city": city,
                    "concert_date": show["开演日期"],
                    "query": query,
                    "candidate_rank_api": 0,
                    "aid": None,
                    "bvid": "",
                    "video_title": "",
                    "video_pubdate": pd.NaT,
                    "artist_hit_score": 0,
                    "city_hit_score": 0,
                    "core_token_hit_score": 0,
                    "date_score": 0,
                    "engagement_score": 0.0,
                    "total_score": 0.0,
                    "date_gap_days": None,
                    "core_hit_count": 0,
                    "selected": 0,
                    "reject_reason": "search_failed" if search_error else "no_search_result",
                }
            )
        else:
            if not selected:
                scored_candidates[0]["reject_reason"] = (
                    scored_candidates[0]["reject_reason"] or "no_candidate_passed_strict_filter"
                )
            audit_rows.extend(scored_candidates)

        selected = sorted(selected, key=lambda x: x["total_score"], reverse=True)[: args.max_videos]
        matched_video_count[show_id] = len(selected)
        selected_video_rows.extend(selected)

        for video in selected:
            aid = int(video["aid"])
            bvid = str(video["bvid"])
            comment_list, all_count = client.fetch_comments(
                aid=aid,
                bvid=bvid,
                max_comments=args.max_comments_per_video,
            )
            source_url = f"https://www.bilibili.com/video/{bvid}"
            video_pubdate = pd.to_datetime(video.get("video_pubdate"), errors="coerce")
            for reply in comment_list:
                comment_id = str(reply.get("rpid_str") or reply.get("rpid") or "")
                if not comment_id or comment_id in seen_comment_ids_global:
                    continue
                seen_comment_ids_global.add(comment_id)
                parent_id = str(reply.get("parent_str") or reply.get("parent") or "")
                ctime = to_datetime_from_unix(reply.get("ctime"))
                content_raw = str((reply.get("content") or {}).get("message") or "").strip()
                like = int(reply.get("like") or 0)
                rcount = int(reply.get("rcount") or reply.get("count") or 0)
                member = reply.get("member") or {}
                mid_str = str(reply.get("mid_str") or member.get("mid") or "")
                raw_rows.append(
                    {
                        "show_id": show_id,
                        "show_name": show_name,
                        "artist": artist,
                        "city": city,
                        "concert_date": show["开演日期"],
                        "platform": show["售票平台"],
                        "bvid": bvid,
                        "aid": aid,
                        "video_title": video["video_title"],
                        "video_pubdate": video_pubdate,
                        "video_comment_total_hint": all_count,
                        "comment_id": comment_id,
                        "parent_id": parent_id,
                        "ctime": ctime,
                        "like": like,
                        "rcount": rcount,
                        "content_raw": content_raw,
                        "source_url": source_url,
                        "mid_hash": hash_mid(mid_str),
                    }
                )
            time.sleep(args.sleep)

        if (idx + 1) % 10 == 0 or idx + 1 == total_shows:
            print(
                "      progress "
                f"{idx + 1}/{total_shows} | selected_videos={len(selected_video_rows)} "
                f"| raw_comments={len(raw_rows)}"
            )

    print("[4/7] building raw/labeled datasets")
    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        labeled_df = raw_df.copy()
        for col in ["content_clean", "dimension_tags", "sentiment_label", "sentiment_score", "kano_role_prior"]:
            labeled_df[col] = []
    else:
        raw_df["ctime"] = pd.to_datetime(raw_df["ctime"], errors="coerce")
        raw_df["video_pubdate"] = pd.to_datetime(raw_df["video_pubdate"], errors="coerce")
        raw_df = raw_df.sort_values(["show_id", "bvid", "ctime", "comment_id"]).reset_index(drop=True)
        labeled_df = raw_df.copy()
        labeled_df["content_clean"] = labeled_df["content_raw"].map(clean_comment_text)
        labeled_df["dimension_tags"] = labeled_df["content_clean"].map(
            lambda t: ";".join(match_dimensions(t, dimension_keywords))
        )
        sentiment_outputs = labeled_df["content_clean"].map(
            lambda t: compute_sentiment(t, pos_words, neg_words, negations)
        )
        labeled_df["sentiment_label"] = sentiment_outputs.map(lambda x: x[0])
        labeled_df["sentiment_score"] = sentiment_outputs.map(lambda x: x[1])
        labeled_df["kano_role_prior"] = labeled_df["dimension_tags"].map(
            lambda s: map_kano_roles([x for x in str(s).split(";") if x], role_map)
        )

    print("[5/7] aggregating SEM-ready metrics by show_id")
    sem_ready_df = build_sem_ready(shows_df, labeled_df, matched_video_count)
    sem_ready_df = sem_ready_df.sort_values(["concert_date", "show_id"]).reset_index(drop=True)

    print("[6/7] preparing audit table")
    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        audit_df["concert_date"] = pd.to_datetime(audit_df["concert_date"], errors="coerce")
        audit_df["video_pubdate"] = pd.to_datetime(audit_df["video_pubdate"], errors="coerce")
        audit_df = audit_df.sort_values(
            ["concert_date", "show_id", "selected", "total_score"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)

    print("[7/7] writing excel outputs")
    raw_path = out_dir / "concert_comments_raw_bilibili_2024_2025.xlsx"
    labeled_path = out_dir / "concert_comments_labeled_kano_sem_2024_2025.xlsx"
    sem_path = out_dir / "concert_sem_ready_metrics_2024_2025.xlsx"
    audit_path = out_dir / "concert_video_match_audit_2024_2025.xlsx"

    raw_df.to_excel(raw_path, index=False)
    labeled_df.to_excel(labeled_path, index=False)
    sem_ready_df.to_excel(sem_path, index=False)
    audit_df.to_excel(audit_path, index=False)

    print("[validation] running acceptance checks")
    match_ratio = (sem_ready_df["matched_videos_cnt"] > 0).mean() if not sem_ready_df.empty else 0.0
    nonzero_comment_shows = int((sem_ready_df["total_comments"] > 0).sum()) if not sem_ready_df.empty else 0
    duplicate_comment_cnt = int(raw_df.duplicated(subset=["comment_id"]).sum()) if not raw_df.empty else 0
    sem_unique_ok = bool(sem_ready_df["show_id"].is_unique) if not sem_ready_df.empty else True
    sem_row_ok = int(len(sem_ready_df)) == int(total_shows)
    unit_pass, unit_rows = run_dimension_unit_cases(dimension_keywords, pos_words, neg_words, negations)

    print(
        f"[validation] api_search_success={search_success_count}/{total_shows}; "
        f"match_ratio={match_ratio:.4f}; target>=0.95"
    )
    print(
        f"[validation] nonzero_comment_shows={nonzero_comment_shows}; "
        "target>=60"
    )
    print(f"[validation] duplicate_comment_id={duplicate_comment_cnt}; target=0")
    print(f"[validation] sem_unique_show_id={sem_unique_ok}; sem_rows_ok={sem_row_ok}")
    print(f"[validation] dimension_unit_cases_pass={unit_pass} ({sum(r['hit'] for r in unit_rows)}/10)")

    print("[done] outputs:")
    print(f"  - {raw_path}")
    print(f"  - {labeled_path}")
    print(f"  - {sem_path}")
    print(f"  - {audit_path}")


if __name__ == "__main__":
    main()
