#!/bin/bash
# mw-v3 monitor: check DB progress + push Feishu (with roast) + update abstract.md
# Runs via cron every 30 min, independent of any Claude Code session

WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/c547084f-7f92-4437-acfd-0c4401e274e2"
EXP_ID="thinkdepthai-qwen3.5-plus-mw-v3"
DB_URL="postgresql://postgres:postgres@localhost:5433/SOTA-Agents"
ABSTRACT="/home/nn/SOTA-agents/RolloutRunner/logs/mw-v3/abstract.md"
TOTAL=105
UV="/home/nn/.local/bin/uv"

# --- All logic in Python for complex text handling ---
RESULT=$($UV run --project /home/nn/SOTA-agents/RolloutRunner python3 << 'PYEOF'
import sqlalchemy, subprocess, json, random, datetime

DB_URL = "postgresql://postgres:postgres@localhost:5433/SOTA-Agents"
EXP_ID = "thinkdepthai-qwen3.5-plus-mw-v3"
TOTAL = 105

# ====== 1. Query DB ======
engine = sqlalchemy.create_engine(DB_URL)
with engine.connect() as conn:
    r = conn.execute(sqlalchemy.text("""
        SELECT
            COALESCE(SUM(CASE WHEN stage='init' THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN stage='rollout' THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN stage='judged' THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN correct THEN 1 ELSE 0 END), 0)
        FROM evaluation_data WHERE exp_id=:eid
    """), {"eid": EXP_ID}).fetchone()
    init, rollout, judged, correct = r

    # Get avg time_cost and avg tokens for completed samples
    stats = conn.execute(sqlalchemy.text("""
        SELECT
            AVG((meta->'cost_metrics'->>'time_cost')::float),
            AVG((meta->'cost_metrics'->>'total_tokens')::float),
            COUNT(*)
        FROM evaluation_data
        WHERE exp_id=:eid AND stage != 'init'
          AND meta->'cost_metrics' IS NOT NULL
    """), {"eid": EXP_ID}).fetchone()
    avg_time = stats[0] or 0
    avg_tokens = stats[1] or 0
    stats_count = stats[2] or 0

done = rollout + judged
failed = done - correct
remaining = init
pct = done * 100 / TOTAL if TOTAL > 0 else 0
ac = correct * 100 / done if done > 0 else 0

# ====== 2. Tmux status ======
try:
    tmux_out = subprocess.check_output(
        ["tmux", "capture-pane", "-t", "mw-v3", "-p", "-S", "-10"],
        stderr=subprocess.DEVNULL, text=True
    )
    lines = [l for l in tmux_out.strip().split("\n") if any(k in l for k in ["INFO","WARNING","ERROR"])][-5:]
    has_429 = any("429" in l for l in lines)
    proc_alive = subprocess.call(["pgrep", "-f", "run_mw_v3_experiment"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
except Exception:
    lines = []
    has_429 = False
    proc_alive = False

proc_status = "进程存活" if proc_alive else "进程已退出!"

# ====== 3. Performance roast ======
def gen_roast():
    """Generate a roast based on current metrics."""

    # --- Progress roasts ---
    if done == 0:
        progress_roasts = [
            "📋 工位已到，咖啡已泡，就是不干活",
            "📋 打卡签到完毕，开始进入冥想模式",
            "📋 「我来了，但我的产出还在路上」",
            "📋 出勤率 100%，产出率 0%，完美摸鱼",
            "📋 已成功占用一个工位和三个 API 连接",
            "📋 上班第一件事：被保安拦（429）",
            "📋 简历写得很好，下次别来了",
        ]
        if has_429:
            progress_roasts += [
                "📋 刚上班就被门禁拦了，HR 说工牌没激活",
                "📋 疯狂刷卡进不去门，保安已经认识你了",
                "📋 429 三连击，API 保安：「你哪个部门的？」",
            ]
        return random.choice(progress_roasts)

    if pct < 10:
        return random.choice([
            f"📊 完成 {pct:.0f}%，相当于开会 2 小时后打开了 IDE",
            f"📊 {done}/{TOTAL}，这个速度，实习生都比你快",
            f"📊 终于开始干活了？还以为你转管理岗了",
            f"📊 产出约等于写了个 Hello World 就去茶水间了",
        ])
    if pct < 30:
        return random.choice([
            f"📊 {pct:.0f}%，勉强算是过了试用期",
            f"📊 有点产出了，但离 KPI 还差十万八千里",
            f"📊 这个速度提交周报，老板会以为是月报",
            f"📊 终于有存在感了，虽然还是倒数第一",
        ])
    if pct < 60:
        return random.choice([
            f"📊 {pct:.0f}%，进入了「看起来在努力」的阶段",
            f"📊 半程了，比大部分同事的年度 OKR 都强",
            f"📊 不快不慢，完美卡在「不会被优化」的线上",
            f"📊 中场休息？不不不，你没有休息的权利",
        ])
    if pct < 90:
        return random.choice([
            f"📊 {pct:.0f}%! 冲刺阶段，老板开始关注你了",
            f"📊 快了快了，年终奖在向你招手",
            f"📊 这波要是能收尾，绩效至少 B+",
            f"📊 最后一公里，别掉链子啊",
        ])
    if done < TOTAL:
        return random.choice([
            f"📊 {pct:.0f}%! 终点在望，别浪别浪",
            f"📊 就差临门一脚了，踢偏了算你的",
            f"📊 HR 已经在写你的表彰邮件了，别翻车",
        ])
    return "🏆 全部完成！请到 HR 处领取「本月最佳牛马」奖杯"

    # --- Accuracy roasts ---
def gen_accuracy_roast():
    if done == 0:
        return ""
    if ac >= 60:
        return random.choice([
            f"🎯 准确率 {ac:.1f}%，这是要卷死其他 Agent 的节奏",
            f"🎯 {ac:.1f}%，学霸本霸，其他人可以回家了",
            f"🎯 正确率拉满，建议直接晋升 P8",
        ])
    if ac >= 40:
        return random.choice([
            f"🎯 准确率 {ac:.1f}%，及格线边缘疯狂试探",
            f"🎯 {ac:.1f}%，对了一半错一半，跟抛硬币差不多",
            f"🎯 这个准确率，建议下次带个计算器",
        ])
    if ac >= 20:
        return random.choice([
            f"🎯 准确率 {ac:.1f}%... 你确定不是在随机输出？",
            f"🎯 {ac:.1f}%，错的比对的多，要不要考虑反着选",
            f"🎯 这成绩，实习转正怕是没戏了",
        ])
    return random.choice([
        f"🎯 准确率 {ac:.1f}%，建议去买彩票，反正运气也不好",
        f"🎯 {ac:.1f}%... 我词穷了，真的",
        f"🎯 这个正确率，怀疑你在用 /dev/urandom 做推理",
    ])

    # --- Speed roasts ---
def gen_speed_roast():
    if stats_count == 0:
        return ""
    mins = avg_time / 60
    if mins > 20:
        return random.choice([
            f"🐌 平均 {mins:.0f} 分钟/题，乌龟看了都着急",
            f"🐌 每题 {mins:.0f} 分钟，你是在用算盘跑大模型吗",
            f"🐌 这个速度，deadline 都过了你还在 loading",
        ])
    if mins > 10:
        return random.choice([
            f"⏱️ 平均 {mins:.0f} 分钟/题，不快不慢，佛系打工",
            f"⏱️ {mins:.0f} 分钟一题，工时填起来挺好看的",
        ])
    return random.choice([
        f"⚡ 平均 {mins:.0f} 分钟/题，速度还行，就是费钱",
        f"⚡ {mins:.0f} 分钟闪电出答案，质量另说",
    ])

    # --- Cost roasts ---
def gen_cost_roast():
    if stats_count == 0 or avg_tokens == 0:
        return ""
    k = avg_tokens / 1000
    if k > 100:
        return random.choice([
            f"💸 平均 {k:.0f}K tokens/题，公司败家子实锤",
            f"💸 {k:.0f}K tokens，你在写论文还是在排障？",
            f"💸 token 用量感人，财务已经在找你谈话了",
        ])
    if k > 50:
        return random.choice([
            f"💰 平均 {k:.0f}K tokens/题，不便宜但还没破产",
            f"💰 {k:.0f}K tokens，花钱还算有节制",
        ])
    return random.choice([
        f"🪙 平均 {k:.0f}K tokens/题，降本增效模范员工",
        f"🪙 {k:.0f}K tokens，抠门得很，但省钱是美德",
    ])

# Assemble roast
roast_lines = [gen_roast()]
ar = gen_accuracy_roast()
if ar: roast_lines.append(ar)
sr = gen_speed_roast()
if sr: roast_lines.append(sr)
cr = gen_cost_roast()
if cr: roast_lines.append(cr)

# 429 special
if has_429 and done > 0:
    roast_lines.append(random.choice([
        "🚫 又 429 了，API 保安：「兄弟你今天第几次了」",
        "🚫 429 限流中，建议去楼下星巴克坐会儿",
        "🚫 被限流了，正好摸会鱼",
    ]))

if not proc_alive:
    roast_lines.append("💀 进程挂了！赶紧来捞一下！！")

roast = "\n".join(roast_lines)

# ====== 4. Grade ======
if done == 0:
    grade = "N/A (尚未产出，无法评级)"
elif done == TOTAL and ac >= 50:
    grade = "S — 年度最佳牛马 🏆"
elif ac >= 50:
    grade = "A — 卷王潜质，继续保持"
elif ac >= 35:
    grade = "B — 中规中矩，不开不裁"
elif ac >= 20:
    grade = "C — 建议回炉重造"
else:
    grade = "D — 绩效面谈通知已发送"

# ====== Output ======
# Line 1: abstract.md note
note_parts = [proc_status]
if has_429: note_parts.append("429限流")
if done > 0: note_parts.append(f"AC@1={ac:.1f}%")
abstract_note = ", ".join(note_parts)

# Line 2+: Feishu card content
now_str = datetime.datetime.now().strftime("%H:%M %b %d")
ac_str = f"AC@1={ac:.1f}%" if done > 0 else ""

feishu_content = (
    f"**进度**: {done}/{TOTAL} ({pct:.0f}%)\n"
    f"**成功**: {correct} | **失败**: {failed} | **剩余**: {remaining}\n"
    f"**{proc_status}** {('| ' + ac_str) if ac_str else ''}\n"
    f"\n---\n"
    f"**📝 绩效考核：thinkdepthai（mw-v3）**\n"
    f"**综合评级**: {grade}\n\n"
    f"{roast}\n"
    f"\n---\n"
    f"⏰ 每 30 分钟自动巡检 (cron)"
)

# Determine color/emoji
if done == TOTAL:
    color, emoji = "green", "✅"
elif done > 0:
    color, emoji = "blue", "🔄"
else:
    color, emoji = "orange", "⏳"

# Output as JSON for bash to consume
print(json.dumps({
    "abstract_note": abstract_note,
    "feishu_content": feishu_content,
    "color": color,
    "emoji": emoji,
    "now_str": now_str,
    "done": done,
    "total": TOTAL,
    "correct": correct,
    "failed": failed,
    "remaining": remaining,
    "grade": grade,
    "all_done": done == TOTAL,
}))
PYEOF
)

# Parse Python output
if [ -z "$RESULT" ]; then
  echo "[$(date)] Python script failed" >> "$ABSTRACT"
  exit 1
fi

NOW=$(date '+%H:%M (%-b %d)')
DONE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['done'])")
CORRECT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['correct'])")
FAILED=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['failed'])")
REMAINING=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['remaining'])")
ABSTRACT_NOTE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['abstract_note'])")
FEISHU_CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['feishu_content'])")
COLOR=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['color'])")
EMOJI=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['emoji'])")
NOW_STR=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['now_str'])")
ALL_DONE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['all_done'])")

# --- Update abstract.md ---
echo "| $NOW | $DONE/$TOTAL | $CORRECT | $FAILED | $REMAINING | $ABSTRACT_NOTE |" >> "$ABSTRACT"

# --- Push Feishu (use Python for proper JSON escaping) ---
$UV run --project /home/nn/SOTA-agents/RolloutRunner python3 -c "
import json, urllib.request

content = $(echo "$RESULT" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['feishu_content']))")
color = '$COLOR'
emoji = '$EMOJI'
now_str = '$NOW_STR'

payload = {
    'msg_type': 'interactive',
    'card': {
        'header': {
            'title': {'tag': 'plain_text', 'content': f'{emoji} mw-v3 巡检 | {now_str}'},
            'template': color
        },
        'elements': [
            {'tag': 'markdown', 'content': content}
        ]
    }
}

req = urllib.request.Request(
    '$WEBHOOK',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'}
)
urllib.request.urlopen(req)
" 2>/dev/null

# --- Auto-remove cron when done ---
if [ "$ALL_DONE" = "True" ]; then
  crontab -l 2>/dev/null | grep -v "mw_v3_monitor" | crontab -
  echo "[$(date)] All $TOTAL samples done. Cron removed." >> "$ABSTRACT"
fi
