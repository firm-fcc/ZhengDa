## 方案标题
**正大杯：基于 B 站公开评论的演唱会维度化抓取与 Kano+SEM 外部证据构建（2024-2025 全量）**

## 摘要
- 按你确认的口径执行：`B站全自动`、`2024-2025 全量85场`、`Q13八维+满意/推荐`、`中等深度`（每场前3视频、每视频最多200条）、`严格匹配`、`全链路三层交付`。
- 技术路线使用 B 站 `wbi` 签名接口（已验证可通），避免票务平台高反爬导致的不可复现问题。
- 交付 4 个核心 Excel：原始评论、维度标注明细、SEM汇总表、视频匹配审计表，全部落到 `data/`。

## 公开接口/文件变更
- 新增主脚本：[kano_sem_comment_pipeline.py](D:/fcc/ZhengDa/scripts/kano_sem_comment_pipeline.py)
- 新增维度词典配置：[kano_sem_dimension_dict.json](D:/fcc/ZhengDa/scripts/kano_sem_dimension_dict.json)
- 生成结果文件：[concert_comments_raw_bilibili_2024_2025.xlsx](D:/fcc/ZhengDa/data/concert_comments_raw_bilibili_2024_2025.xlsx)
- 生成结果文件：[concert_comments_labeled_kano_sem_2024_2025.xlsx](D:/fcc/ZhengDa/data/concert_comments_labeled_kano_sem_2024_2025.xlsx)
- 生成结果文件：[concert_sem_ready_metrics_2024_2025.xlsx](D:/fcc/ZhengDa/data/concert_sem_ready_metrics_2024_2025.xlsx)
- 生成结果文件：[concert_video_match_audit_2024_2025.xlsx](D:/fcc/ZhengDa/data/concert_video_match_audit_2024_2025.xlsx)

## 实施计划（可直接交付实现）
1. 输入与过滤：读取 `concert_list_20260223_212443(2).xlsx`，保留 2024-2025 场次，标准化字段类型（日期、票价、布尔）。
2. B 站签名模块：实现 `nav -> wbi_key -> w_rid` 的签名流程，统一请求封装（重试3次、指数退避、超时20s、失败日志）。
3. 视频检索：对每场构造检索词 `艺人 + 演出名称 + 城市`，调用 `x/web-interface/wbi/search/type` 取前20条候选。
4. 严格匹配打分：候选仅保留 `aid+bvid` 完整项；分数固定为 `artist命中(5) + 城市命中(3) + 演出名核心词命中(每词2，上限6) + 日期接近(<=45天:+2, <=120天:+1) + 互动强度加分(0.2*log1p(play+review))`，每场取 Top3 且 `bvid` 去重。
5. 评论抓取：对每个选中视频调用 `x/v2/reply/wbi/main`，参数固定 `type=1, mode=3, ps=20`，按 `next` 翻页直到 `is_end=True` 或达到 `200条/视频`。
6. 原始落表：保留 `show_id, show_name, artist, city, concert_date, platform, bvid, aid, video_title, video_pubdate, comment_id, parent_id, ctime, like, rcount, content_raw, source_url`；用户标识仅保留 `mid_hash=sha256(mid_str+固定盐)`。
7. 文本清洗：去HTML、URL、@提及、表情符号、冗余空白；生成 `content_clean`。
8. 维度映射：按词典对每条评论多标签匹配，标签集合固定为 `security_check, free_supply, controlled_lightstick, network_signal, temperature_control, hygiene, photo_checkin, shuttle_service, satisfaction, recommendation`。
9. 情感判定：使用规则词典（正负词+否定词窗口翻转）产出 `sentiment_label(pos/neu/neg)` 与 `sentiment_score[-1,1]`。
10. Kano桥接字段：新增 `kano_role_prior`（先验映射）用于论文解释：`must_be={security_check,network_signal,temperature_control,hygiene,shuttle_service}`、`one_dim={free_supply,controlled_lightstick}`、`attractive={photo_checkin}`，`satisfaction/recommendation` 作为结果变量证据。
11. SEM汇总表：按 `show_id` 聚合输出每维度 `mention_cnt, mention_ratio, neg_ratio, mean_sentiment`，并输出 `sat_pos_ratio, rec_pos_ratio, total_comments, matched_videos_cnt`。
12. 审计表输出：保存每场 Top3 候选的分数明细和未命中原因（便于方法章节复现与附录说明）。
13. 一键运行命令固定为：`python D:/fcc/ZhengDa/scripts/kano_sem_comment_pipeline.py --input "D:/OneDrive/桌面/concert_list_20260223_212443(2).xlsx" --out-dir "D:/fcc/ZhengDa/data" --years 2024 2025 --max-videos 3 --max-comments-per-video 200 --retry 3 --sleep 0.35`。

## 数据表结构（最终固定）
- `raw` 表：一行一条评论，包含评论原文与视频/演出关联键，供追溯。
- `labeled` 表：在 `raw` 基础上增加 `dimension_tags`、`sentiment_label`、`sentiment_score`、`kano_role_prior`。
- `sem_ready` 表：一行一场演出，含 8 个 Kano 服务维度指标 + 满意/推荐指标。
- `audit` 表：一行一个候选视频，含匹配得分拆解与是否被选中。

## 测试与验收场景
1. 接口可用性：签名搜索返回 `code=0`，评论接口可分页返回。
2. 匹配质量：85场中视频命中率目标 `>=95%`；未命中必须在审计表记录原因。
3. 抓取有效性：非零评论场次目标 `>=60`；评论ID全局去重后无重复主键。
4. 标签正确性：构造10条规则样例文本，8个服务维度与满意/推荐均可命中。
5. 汇总一致性：`sem_ready` 中 `show_id` 唯一且行数等于输入演出数；指标列无类型错误。
6. 可复现性：同参数重复运行可稳定生成同结构文件；差异仅体现在平台新增评论量。

## 明确假设与默认值
- 仅抓取公开可访问评论，不绕过登录/验证码，不抓取私密数据。
- 以 B 站为评论来源代理“观演体验舆情”，并在报告中注明平台偏差。
- 同名不同日期场次视为不同演出样本（按 `演出ID` 区分）。
- 无评论或无匹配视频的演出仍保留在 `sem_ready`，相关指标置0并标记 `data_gap=1`。
- 当前环境不依赖额外第三方库安装（优先使用标准库 + 已有 `pandas/openpyxl`）。
