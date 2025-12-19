import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import google.generativeai as genai
from datetime import datetime

# --- 1. 網頁基本設定 (手機版建議用預設 layout，但這裡維持 wide 讓圖表大一點) ---
st.set_page_config(page_title="AI 選股、各股分析 V2.0", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <h1 style='text-align: center; font-size: 28px; margin-bottom: 20px;'>
        📱 AI 選股 V2.0
    </h1>
    """, 
    unsafe_allow_html=True
)

# --- 側邊欄：只放不會常動的設定 ---
with st.sidebar:
    st.header("⚙️ 系統設定")
    # 優先讀取 Secrets
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key 已載入 ✅")
    else:
        api_key = st.text_input("輸入 Gemini API Key", type="password")

    if api_key:
        genai.configure(api_key=api_key.strip())
    
    st.info("💡 手機版操作提示：\n點擊圖表可放大查看\n點擊表格標題可排序")

# --- 共用函數 ---
def get_stock_name(code):
    try:
        return twstock.codes[code].name if code in twstock.codes else code
    except:
        return code

# --- 核心數據函數 (含防護網) ---
def get_mixed_data(code):
    try:
        # 手機版減少數據量，改抓 3 個月加快速度
        df = yf.download(f"{code}.TW", period="3mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.empty: return None, None
    except: return None, None

    latest = df.iloc[-1].copy()
    try:
        realtime_data = twstock.realtime.get(code)
        if realtime_data and realtime_data['success'] and realtime_data['realtime']['latest_trade_price']:
            latest['Close'] = float(realtime_data['realtime']['latest_trade_price'])
    except: pass
    
    return df, latest

# --- 技術指標 ---
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
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss
    df['RSI_6'] = 100 - (100 / (1 + rs))
    return df

# --- AI 分析 ---
def ask_ai_analyst(ticker, name, df, latest):
    if not api_key: return "⚠️ 請先設定 API Key"
    
    # 簡化 Prompt 以加快手機生成速度
    prompt = f"""
    分析台股 {name}({ticker})。
    數據：股價{latest['Close']:.2f}, MA5={latest['MA5']:.1f}, MA20={latest['MA20']:.1f}, KD(K={latest['K']:.1f}), RSI={latest['RSI_6']:.1f}。
    請用繁體中文，針對手機閱讀優化(列點、簡短)，分析：
    1.趨勢(強/弱/盤) 2.支撐/壓力位 3.操作建議(買/賣/觀望)
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        with st.spinner('🤖 AI 分析中...'):
            response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"連線失敗：{str(e)}"

# --- 篩選計算 ---
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
    except: return df

# --- 介面佈局 (手機版優化) ---
# 使用 emoji 縮短標籤長度，適應手機寬度
tab1, tab2 = st.tabs(["🔍 個股診斷", "🌪️ 策略選股"])

# ==========================================
# 分頁 1: 個股診斷 (手機版)
# ==========================================
with tab1:
    # 1. 搜尋列 (置頂)
    col_input, col_btn = st.columns([2, 1])
    with col_input:
        input_ticker = st.text_input("股票代碼", value="2330", label_visibility="collapsed", placeholder="輸入代碼 (如 2330)")
    with col_btn:
        run_ai = st.button("分析", type="primary", use_container_width=True) # 按鈕填滿寬度

    if run_ai:
        stock_code = input_ticker.replace(".TW", "").strip()
        stock_name = get_stock_name(stock_code)
        
        df, latest = get_mixed_data(stock_code)
        
        if df is None:
            st.toast("找不到資料，請檢查代碼", icon="❌")
        else:
            df = calculate_technical_indicators(df)
            
            # 2. 顯示大字報價格 (手機最需要一眼看到價錢)
            latest_price = latest['Close']
            prev_close = df['Close'].iloc[-2]
            diff = latest_price - prev_close
            diff_pct = (diff / prev_close) * 100
            color = "normal"
            if diff > 0: color = "off" # Streamlit metric 綠色是 normal...這裡用預設就好
            
            st.metric(
                label=f"{stock_name} ({stock_code})",
                value=f"{latest_price:.2f}",
                delta=f"{diff:.2f} ({diff_pct:.2f}%)"
            )

            # 3. K線圖 (手機版不宜太高)
            st.line_chart(df[['Close', 'MA5', 'MA20']], height=250, color=["#ffffff", "#ffaa00", "#00aaff"])
            
            # 4. AI 結果 (用 Expander 預設展開，但可以收起來)
            with st.expander("🤖 AI 分析報告 (點擊收合)", expanded=True):
                ai_result = ask_ai_analyst(stock_code, stock_name, df, df.iloc[-1])
                st.markdown(ai_result)

# ==========================================
# 分頁 2: 策略選股 (手機版)
# ==========================================
with tab2:
    # 1. 將佔空間的設定收進摺疊選單
    with st.expander("⚙️ 設定篩選條件 (點擊展開)", expanded=False):
        list_mode = st.radio("範圍", ("🚀 熱門股", "🐢 全台股"))
        
        c1, c2 = st.columns(2)
        with c1: min_p = st.number_input("最低價", 0.0, value=20.0)
        with c2: max_p = st.number_input("最高價", 0.0, value=150.0)
        
        st.caption("技術條件")
        c3, c4 = st.columns(2)
        with c3: use_kd = st.checkbox("KD金叉", True)
        with c4: use_vol = st.checkbox("爆量", True)
        
    # 2. 大按鈕
    if st.button("🚀 開始掃描", type="primary", use_container_width=True):
        st.toast("正在掃描中...", icon="⏳")
        
        # (簡化的掃描邏輯，保持原本功能)
        if list_mode.startswith("🚀"):
            raw_list = ["2330", "2317", "2454", "2308", "2303", "2881", "2412", "2382", "3008", "2603", "2609", "2615", "3231", "3481", "2409", "6116"]
        else:
            raw_list = [c for c, i in twstock.codes.items() if i.type == '股票' and i.market == '上市']

        ticker_list_tw = [f"{x}.TW" for x in raw_list]
        
        # 價格快篩
        try:
            batch_data = yf.download(ticker_list_tw, period="1d", progress=False)
            if 'Close' not in batch_data: st.stop()
            
            prices = batch_data['Close'].iloc[-1] if len(ticker_list_tw) > 1 else pd.Series({ticker_list_tw[0]: batch_data['Close'].iloc[-1]})
            
            qualified = []
            for code in ticker_list_tw:
                try:
                    p = prices[code]
                    if min_p <= p <= max_p:
                        clean = code.replace(".TW", "")
                        qualified.append((clean, get_stock_name(clean)))
                except: continue
            
            # 技術篩選
            final = []
            bar = st.progress(0)
            
            for i, (code, name) in enumerate(qualified):
                bar.progress((i+1)/len(qualified))
                try:
                    df = yf.download(f"{code}.TW", period="3mo", progress=False)
                    if df.empty or len(df) < 20: continue
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                    
                    df = calculate_kd_simple(df)
                    cur, prev = df.iloc[-1], df.iloc[-2]
                    
                    match_kd = (prev['K'] < prev['D'] and cur['K'] > cur['D']) if use_kd else True
                    match_vol = (cur['Volume'] > prev['Volume'] * 1.2) if use_vol else True
                    
                    if match_kd and match_vol:
                        final.append({
                            "代碼": code, "股名": name, # 手機版欄位名要短
                            "價": f"{cur['Close']:.1f}",
                            "K": f"{cur['K']:.0f}"
                        })
                except: continue
            
            bar.empty()
            if final:
                st.toast(f"找到 {len(final)} 檔！", icon="🎉")
                st.dataframe(pd.DataFrame(final), use_container_width=True, hide_index=True)
            else:
                st.toast("無符合條件股票", icon="⚠️")
                
        except Exception as e:
            st.error(str(e))