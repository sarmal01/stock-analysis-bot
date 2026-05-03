import os
import time
import json
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests
from google import genai
from datetime import datetime, timedelta, timezone

# セキュリティ設定
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# 分析対象の銘柄
TICKERS = ["^GSPC", "NVDA", "9432.T"]

def send_to_discord(res):
    """1銘柄ごとにDiscordへ送信する"""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return

    try:
        score_val = int(res['score'])
    except:
        score_val = 0

    emoji = "🚀" if score_val > 80 else "📈" if score_val > 60 else "⚠️" if score_val < 40 else "📊"
    
    jst = timezone(timedelta(hours=9))
    jst_now = datetime.now(jst).strftime('%Y/%m/%d %H:%M')
    
    content = (
        f"📅 **{jst_now} 分析レポート**\n"
        f"**{res.get('name', '不明')} ({res['ticker']})**\n"
        f"スコア: **{score_val}** {emoji}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"**【今後の展望】**\n{res['reason']}\n\n"
        f"**【注意すべきリスク】**\n{res['risk']}\n"
        f"━━━━━━━━━━━━━━━"
    )

    payload = {"content": content}
    requests.post(webhook_url, json=payload)
    time.sleep(2) # Discordの制限対策で少し長めに待機

def run_analysis():
    all_results = []
    print("--- 統合分析フェーズ開始 ---")
    
    for symbol in TICKERS:
        try:
            print(f"Waiting for safety...")
            time.sleep(20) 
            
            print(f"Analyzing {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # 1. テクニカル指標の計算
            df = ticker.history(period="3mo")
            if df.empty:
                print(f"  ❌ {symbol} データが取得できませんでした")
                continue

            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['SMA20'] = ta.sma(df['Close'], length=20)
            technical_data = df.tail(5)[['Close', 'Volume', 'RSI', 'SMA20']].to_csv()

            # 2. ニュースデータの取得
            news_text = ""
            try:
                news_list = ticker.news[:3]
                for n in news_list:
                    c = n.get('content', {})
                    news_text += f"■ {c.get('title')}\n要約: {c.get('summary', '')[:100]}...\n"
            except:
                news_text = "ニュースの取得に失敗しました。"

            # 3. Geminiによる分析 (モデル名を1.5-flashに修正)
            prompt = f"""
            銘柄 {symbol} について、投資家視点で【厳格に】分析してください。
            1. 回答は必ず日本語。
            2. スコアは0〜100で【相対的な差】を明確につけること。
            3. 提供された数値データ（RSI, SMA）を具体的に引用すること。

            {{
                "score": 整数,
                "reason": "今後の展望",
                "risk": "注意すべきリスク"
            }}

            【市場データ】\n{technical_data}
            【ニュース】\n{news_text}
            """
            
            response = client.models.generate_content(
                model="gemini-3-flash-preview", # 正式名称に修正
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            
            res_json = json.loads(response.text)
            res_json['ticker'] = symbol
            
            # 銘柄名の取得（失敗しても止まらないようにする）
            try:
                res_json['name'] = ticker.info.get('shortName', symbol)
            except:
                res_json['name'] = symbol
            
            # ベクトル化
            print(f"  ベクトル化中...")
            emb = client.models.embed_content(model="models/gemini-embedding-2", contents=res_json['reason'])
            res_json['embedding'] = emb.embeddings[0].values
            
            all_results.append(res_json)
            
            # Discordへ送信
            send_to_discord(res_json)
            print(f"  ✅ {symbol} 分析・送信完了")

        except Exception as e:
            print(f"  ❌ {symbol} エラー発生: {e}")

    if all_results:
        pd.DataFrame(all_results).to_json("stock_research_data.json", orient="records", force_ascii=False)
        print("✅ 全データの保存が完了しました。")

if __name__ == "__main__":
    run_analysis()
