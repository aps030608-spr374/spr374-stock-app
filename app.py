import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import google.generativeai as genai
from datetime import datetime

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="AI 智囊選股", layout="wide")
st.title("🧠 AI 智囊選股助手 (雲端部署版)")

# --- 側邊欄：API Key 與共用設定 (支援 Secrets) ---
st.sidebar.header("🔑 系統設定")

# 優先檢查是否設定了 Streamlit Secrets (雲端或本機機密檔)
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ API Key 已從系統變數載入")
else:
    # 如果沒設定 Secrets，才顯示輸入框 (適合分享給沒有 Key 的人)
    api_key = st.sidebar.text_input("Gemini API Key (AI功能必填)", type="password", help="請輸入 Google AI Studio 申請的 Key")

if api_key:
    # 自動清除前後空格，避免複製錯誤
    clean_key = api_key.strip()
    genai.configure(api_key=clean_key)

# --- 共用函數 ---
def get_stock_name(code):
    try:
        if code in twstock.codes:
            return twstock.codes[code].name
        return code
    except:
        return code

# --- 核心：取得即時與歷史混合數據 (含 SSL 防護網) ---
def get_mixed_data(code):
    # 1. 先抓歷史 (yfinance)
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if df.empty: return None, None
    except Exception as e:
        st.error(f"歷史資料下載失敗: {e}")
        return None, None

    latest = df.iloc[-1].copy()

    # 2. 嘗試抓即時 (twstock) 用來校正
    try:
        realtime_data = twstock.realtime.get(code)
        if realtime_data and realtime_data['success'] and realtime_data['realtime']['latest_trade_price']:
            rt_price = float(realtime_data['realtime']['latest_trade_price'])
            latest['Close'] = rt_price
    except Exception as e:
        print(f"⚠️ twstock 連線失敗 (已自動切換為歷史數據模式): {e}")
        pass
    
    return df, latest

# --- 功能一：AI 分析專用函數 ---
def calculate_technical_indicators(df):
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
    rsv = rsv.fillna(50)
    k, d = [50], [50]
    for i in range(1, len(df)):
        k.append(k[-1]*2/3 + rsv.iloc[i]/3)
        d.append(d[-1]*2/3 + k[-1]/3)
    df['K'], df['D'] = k, d

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss
    df['RSI_6'] = 100 - (100 / (1 + rs))
    return df

def ask_ai_analyst(ticker, name, df, latest):
    if not api_key:
        return "⚠️ 請先設定 API Key 才能啟動 AI 分析。"

    prev = df.iloc[-2]
    
    # 智能判斷成交量單位
    vol_raw = int(latest['Volume'])
    if vol_raw > 100000:
        vol_display = f"{int(vol_raw / 1000)} 張"
    else:
        vol_display = f"{vol_raw} (單位未確認)"

    change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    trend = "多頭排列" if latest['MA5'] > latest['MA20'] > latest['MA60'] else "整理或空頭"
    kd_status = "黃金交叉" if latest['K'] > latest['D'] and prev['K'] < prev['D'] else "無特殊交叉"
    data_date = latest.name.strftime('%Y-%m-%d') if hasattr(latest, 'name') else "最新交易日"

    prompt = f"""
    你是一位專業的台灣股市分析師。請根據以下 {name} ({ticker}) 的數據進行評估。
    
    【基本資訊】
    - 資料日期：{data_date}
    - 目前股價：{latest['Close']:.2f} (漲跌幅 {change_pct:.2f}%)
    - 成交量：{vol_display}
    
    【技術指標】
    - 均線狀態：MA5={latest['MA5']:.1f}, MA20={latest['MA20']:.1f}, MA60={latest['MA60']:.1f} ({trend})
    - KD指標：K={latest['K']:.1f}, D={latest['D']:.1f} ({kd_status})
    - RSI(6)：{latest['RSI_6']:.1f}

    【任務】請用繁體中文撰寫簡短分析：
    1. **盤勢解讀**：目前是強勢還是弱勢？
    2. **關鍵價位**：下方支撐在哪？上方壓力在哪？
    3. **操作建議**：空手者該買嗎？持有者該賣嗎？
    (請註明僅供參考)
    """
    try:
        # 使用最新的 gemini-2.5-flash 模型
        model = genai.GenerativeModel('gemini-2.5-flash')
        with st.spinner('🤖 AI 正在看盤分析中 (Model: Gemini 2.5)...'):
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 連線失敗：{str(e)}"

# --- 功能二：篩選器專用函數 ---
def calculate_kd_simple(df):
    try:
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
        rsv = rsv.fillna(50)
        k, d = [50], [50]
        for i in range(1, len(df)):
            k.append(k[-1]*2/3 + rsv.iloc[i]/3)
            d.append(d[-1]*2/3 + k[-1]/3)
        df['K'], df['D'] = k, d
        return df
    except:
        return df

# --- 介面佈局 ---
tab1, tab2 = st.tabs(["📊 個股 AI 診斷", "🌪️ 策略選股漏斗"])

# ==========================================
# 分頁 1: 個股 AI 診斷
# ==========================================
with tab1:
    st.subheader("個股全方位診斷 + AI 建議")
    col1, col2 = st.columns([1, 3])
    with col1:
        input_ticker = st.text_input("輸入股票代碼", value="3481", key="ai_ticker")
        run_ai = st.button("✨ 啟動 AI 分析", type="primary")
    
    if run_ai:
        stock_code = input_ticker.replace(".TW", "").replace(".tw", "").strip()
        stock_name = get_stock_name(stock_code)
        
        df, latest = get_mixed_data(stock_code)
        
        if df is None:
            st.error("找不到此股票資料，請確認代碼是否正確。")
        else:
            df = calculate_technical_indicators(df)
            latest_with_indicators = df.iloc[-1].copy()
            latest_with_indicators['Close'] = latest['Close']
            
            st.metric(f"{stock_code} {stock_name}", f"{latest['Close']:.2f}")
            st.line_chart(df[['Close', 'MA20', 'MA60']], color=["#ffffff", "#ffaa00", "#00aaff"])
            
            ai_result = ask_ai_analyst(stock_code, stock_name, df, latest_with_indicators)
            st.info("🤖 AI 分析師觀點：")
            st.markdown(ai_result)

# ==========================================
# 分頁 2: 策略選股漏斗
# ==========================================
with tab2:
    st.subheader("兩階段策略選股 (價格快篩 -> 技術精選)")
    
    with st.expander("⚙️ 設定篩選條件", expanded=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            list_mode = st.radio("掃描範圍", ("🚀 熱門股 (快)", "🐢 全台股 (慢)"))
            min_p = st.number_input("最低價", min_value=0.0, value=10.0)
            max_p = st.number_input("最高價", min_value=0.0, value=200.0)
        with col_m2:
            use_kd = st.checkbox("KD 黃金交叉", value=True)
            use_vol = st.checkbox("爆量", value=True)
            vol_pct = st.slider("爆量增幅 %", 10, 100, 20) / 100

    if st.button("🚀 開始掃描"):
        if list_mode == "🚀 熱門股 (快)":
            raw_list = ["2330", "2317", "2454", "2308", "2303", "2881", "2412", "2382", "3008", "2882", "2603", "2609", "2615", "3231", "2357", "2324", "3481", "2409", "6116"]
        else:
            st.info("載入全台股清單中...")
            raw_list = [c for c, i in twstock.codes.items() if i.type == '股票' and i.market == '上市']

        ticker_list_tw = [f"{x}.TW" for x in raw_list]
        st.write(f"目標掃描：{len(raw_list)} 檔")
        
        progress_bar = st.progress(0)
        try:
            batch_data = yf.download(ticker_list_tw, period="1d", progress=False)
            
            if len(ticker_list_tw) > 1:
                if 'Close' not in batch_data:
                    st.error("無法取得股價資料，請稍後再試。")
                    st.stop()
                current_prices = batch_data['Close'].iloc[-1]
            else:
                current_prices = pd.Series({ticker_list_tw[0]: batch_data['Close'].iloc[-1]})

            qualified = []
            for code_tw in ticker_list_tw:
                try:
                    if code_tw in current_prices:
                        p = current_prices[code_tw]
                        if min_p <= p <= max_p:
                            clean_code = code_tw.replace(".TW", "")
                            qualified.append((clean_code, get_stock_name(clean_code), p))
                except: continue
            
            st.success(f"✅ 價格符合：{len(qualified)} 檔 (進入第二階段)")
            progress_bar.progress(50)

            final_results = []
            if qualified:
                for i, (code, name, price) in enumerate(qualified):
                    progress_bar.progress(0.5 + 0.5 * ((i+1)/len(qualified)))
                    try:
                        df = yf.download(f"{code}.TW", period="3mo", progress=False)
                        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                        if len(df) < 20: continue
                        
                        df = calculate_kd_simple(df)
                        today, prev = df.iloc[-1], df.iloc[-2]

                        match_kd = (prev['K'] < prev['D'] and today['K'] > today['D']) if use_kd else True
                        match_vol = (today['Volume'] > prev['Volume'] * (1 + vol_pct)) if use_vol else True

                        if match_kd and match_vol:
                            final_results.append({
                                "代碼": code, "名稱": name, "現價": f"{today['Close']:.2f}",
                                "K值": f"{today['K']:.1f}", "成交量": int(today['Volume']), "訊號": "🌟入選"
                            })
                    except: continue

            progress_bar.progress(100)
            if final_results:
                st.balloons()
                st.dataframe(pd.DataFrame(final_results), use_container_width=True)
            else:
                st.warning("無符合技術條件的股票")
        except Exception as e:
            st.error(f"發生錯誤：{str(e)}")