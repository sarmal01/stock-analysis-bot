import os
import time
import json
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from google import genai

# セキュリティ設定：GitHubのSecretsから読み込む
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# 分析対象の銘柄
TICKERS = ["^GSPC", "NVDA", "9432.T"]

def run_analysis():
    all_results = []
    print("--- 統合分析フェーズ開始 ---")
    
    for symbol in TICKERS:
        try:
            # 1分あたりの制限を避けるため、各銘柄の前に20秒待機
            print(f"Waiting for safety...")
            time.sleep(20) 
            
            print(f"Analyzing {symbol}...")
            
            # モデルを安定性の高い 1.5-flash に変更
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite", 
                contents=f"銘柄 {symbol} の分析をして..."
            )
            print(f"Analyzing {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # 1. テクニカル指標の計算（3ヶ月分取得して直近5日分を使用）
            df = ticker.history(period="3mo")
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['SMA20'] = ta.sma(df['Close'], length=20)
            technical_data = df.tail(5)[['Close', 'Volume', 'RSI', 'SMA20']].to_csv()

            # 2. ニュースデータの取得
            news_list = ticker.news[:3]
            news_text = ""
            for n in news_list:
                c = n.get('content', {})
                news_text += f"■ {c.get('title')}\n要約: {c.get('summary', '')[:100]}...\n"

            # 3. Geminiによる統合分析（武田さんのプロンプトをベースに最適化）
            prompt = f"""
            銘柄 {symbol} の分析依頼。
            【数値と指標】\n{technical_data}
            【ニュース】\n{news_text}
            上記から、今後の展望を分析し JSON形式(score, reason, risk) で出力してください。
            """
            
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            
            res_json = json.loads(response.text)
            res_json['ticker'] = symbol
            res_json['name'] = ticker.info.get('shortName', symbol)
            
            # ベクトル化（類似度分析用）
            print(f"  ベクトル化中...")
            emb = client.models.embed_content(model="models/gemini-embedding-2", contents=res_json['reason'])
            res_json['embedding'] = emb.embeddings[0].values
            
            all_results.append(res_json)
            print(f"  ✅ {symbol} 分析完了")

        except Exception as e:
            print(f"  ❌ {symbol} エラー発生: {e}")

    # 保存処理
    if all_results:
        pd.DataFrame(all_results).to_json("stock_research_data.json", orient="records", force_ascii=False)
        print("✅ 全データの保存が完了しました。")

if __name__ == "__main__":
    run_analysis()
