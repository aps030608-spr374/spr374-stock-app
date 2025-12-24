import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import numpy as np

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="AI 掌上股市 Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    '<h1 style="font-size: 24px; white-space: nowrap; margin-bottom: 20px;">📱 AI 掌上股市 Pro V11.8 (記憶體優化版)</h1>', 
    unsafe_allow_html=True
)

# --- 初始化 Session State (這是解決跳頁的關鍵！) ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'advisor_results' not in st.session_state:
    st.session_state.advisor_results = []

# --- 側邊欄：系統設定 ---
with st.sidebar:
    st.header("⚙️ 引擎設定")
    ai_provider = st.radio("AI 核心", ("Google Gemini (免費)", "DeepSeek (付費)"), index=0)
    st.divider()
    
    api_key = ""
    selected_model = ""
    
    if ai_provider == "Google Gemini (免費)":
        default_key = st.secrets.get("GEMINI_API_KEY", "")
        api_key = st.text_input("Gemini API Key", value=default_key, type="password")
        if api_key: st.caption(f"🔑 ...{api_key[-4:]}")
        selected_model = st.selectbox("Model", ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"], index=0)
    else:
        default_ds_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        api_key = st.text_input("DeepSeek API Key", value=default_ds_key, type="password")
        selected_model = st.selectbox("Model", ["deepseek-chat", "deepseek-reasoner"], index=0)

    st.info("💡 V11.8 修復：\n1. 加入記憶體機制，點擊分析不會再跳掉。\n2. 個股診斷加入緩衝，減少 429 機率。")

# --- 核心 API ---
def call_ai_engine(prompt, provider, model_name, key):
    if not key: return "⚠️ 請設定 API Key"
    
    # 強制緩衝 1 秒，避免手速太快觸發 429
    time.sleep(1)
    
    if provider == "Google Gemini (免費)":
        clean_model = model_name.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key}"
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.7}}
        try:
            res = requests.post(url, headers=headers, json=data, timeout=30)
            if res.status_code == 200: return res.json()['candidates'][0]['content']['parts'][0]['text']
            elif res.status_code == 429: return "❌ 速度太快 (429)，請稍等 5 秒再試。"
            else: return f"❌ Google Error ({res.status_code})"
        except Exception as e: return f"連線錯誤: {str(e)}"
    else:
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        data = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "stream": False}
        try:
            res = requests.post(url, headers=headers, json=data, timeout=60)
            if res.status_code == 200: return res.json()['choices'][0]['message']['content']
            elif "Insufficient Balance" in res.text: return "❌ DeepSeek 餘額不足"
            else: return f"❌ DeepSeek Error ({res.status_code})"
        except Exception as e: return f"連線錯誤: {str(e)}"

# --- 數據函數 ---
def get_stock_name(code):
    try: return twstock.codes[code].name if code in twstock.codes else code
    except: return code

def get_market_cap_robust(code, current_price):
    try:
        ticker = yf.Ticker(f"{code}.TW")
        mkt_cap = ticker.fast_info.market_cap
        if mkt_cap and mkt_cap > 0: return round(mkt_cap / 100000000, 1)
    except: pass
    try:
        if code in twstock.codes:
            cap = twstock.codes[code].capital
            if cap: return round((float(cap)/10 * current_price) / 100000000, 1)
    except: pass
    return 0

@st.cache_data
def get_tw_stock_list():
    stock_list = []
    for code, info in twstock.codes.items():
        if info.type == "股票" and info.market == "上市":
            stock_list.append(f"{code} {info.name}")
    return stock_list

def get_stock_news(code, name):
    try:
        url = f"https://news.google.com/rss/search?q={name}+{code}+stock+when:14d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        res = requests.get(url, timeout=3)
        soup = BeautifulSoup(res.content, features="xml")
        items = soup.findAll('item')
        news_list = []
        for item in items[:3]:
            title = item.title.text
            news_list.append(f"- {title}")
        return "\n".join(news_list)
    except: return "無新聞"

def get_mixed_data(code):
    try:
        df = yf.download(f"{code}.TW", period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.empty: return None, None
    except: return None, None
    latest = df.iloc[-1].copy()
    try:
        realtime = twstock.realtime.get(code)
        if realtime and realtime['success']:
            latest['Close'] = float(realtime['realtime']['latest_trade_price'])
    except: pass
    return df, latest

def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
    rsv = rsv.fillna(50)
    k, d = [50], [50]
    for i in range(1, len(df)):
        k.append(k[-1]*2/3 + rsv.iloc[i]/3)
        d.append(d[-1]*2/3 + k[-1]/3)
    df['K'], df['D'] = k, d
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    rs = gain / loss
    df['RSI_6'] = 100 - (100 / (1 + rs))
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HV'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252) * 100
    df['HV'] = df['HV'].fillna(0)
    df['Std'] = df['Close'].rolling(20).std()
    df['BB_Up'] = df['MA20'] + 2 * df['Std']
    df['BB_Low'] = df['MA20'] - 2 * df['Std']
    df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / df['MA20']
    df['Box_High_20'] = df['High'].shift(1).rolling(20).max()
    return df

# --- 主介面 ---
tab1, tab2, tab3 = st.tabs(["🔍 個股診斷", "🌪️ 策略選股", "🤖 AI 智能投顧"])

# Tab 1: 個股診斷
with tab1:
    all_stocks = get_tw_stock_list()
    default_idx = 0
    for i, s in enumerate(all_stocks):
        if "2330" in s: default_idx = i; break
        
    c1, c2 = st.columns([3, 1])
    with c1: selected_stock = st.selectbox("搜尋股票", all_stocks, index=default_idx, label_visibility="collapsed")
    with c2: run_btn = st.button("分析", type="primary", use_container_width=True)

    if run_btn:
        code = selected_stock.split(" ")[0]
        name = get_stock_name(code)
        df, latest_raw = get_mixed_data(code)
        
        if df is None: st.toast("無資料", icon="❌")
        else:
            df = calculate_indicators(df)
            latest = df.iloc[-1].copy()
            latest['Close'] = latest_raw['Close']
            cap = get_market_cap_robust(code, latest['Close'])
            
            # 使用 container 確保佈局穩定
            with st.container():
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("股價", f"{latest['Close']:.2f}")
                with m2: st.metric("HV波動", f"{latest['HV']:.1f}%")
                with m3: st.caption(f"市值 {cap} 億")
                
                st.line_chart(df[['Close', 'MA20', 'BB_Up', 'BB_Low']], height=250)
                
                news_text = get_stock_news(code, name)
                
                # 直接呼叫 AI (這裡不需要 session state，因為按鈕本身就是觸發者)
                prompt = f"分析 {name}({code})。價{latest['Close']}，HV{latest['HV']:.1f}%，MA({latest['MA5']:.1f}/{latest['MA20']:.1f})。新聞：{news_text}。請給出操作建議。"
                with st.spinner("AI 思考中..."):
                    res = call_ai_engine(prompt, ai_provider, selected_model, api_key)
                    st.info(res)

# Tab 2: 策略選股
with tab2:
    with st.expander("⚙️ 篩選條件", expanded=True):
        list_mode = st.radio("範圍", ("🚀 熱門股", "🐢 全台股 (慢)"))
        c1, c2 = st.columns(2)
        with c1: min_p = st.number_input("Min $", 0.0, value=10.0)
        with c2: max_p = st.number_input("Max $", 0.0, value=3000.0)
        st.markdown("---")
        use_warrant = st.checkbox("🎯 權證飆速", False)
        use_0050 = st.checkbox("🏆 0050潛力", False)
        use_box = st.checkbox("📦 突破箱體", False)
        use_ma = st.checkbox("📈 均線多頭", False)
        use_bb = st.checkbox("⚡ 布林爆發", False)
        
    if st.button("🚀 開始掃描", type="primary", use_container_width=True):
        st.session_state.scan_results = [] # 清空舊資料
        
        if list_mode.startswith("🚀"):
            raw_list = ["2330", "2317", "2454", "2308", "2303", "2881", "2412", "3008", "2603", "3037", "3481", "2409"]
        else:
            raw_list = [c for c, i in twstock.codes.items() if i.type == '股票' and i.market == '上市']
            
        bar = st.progress(0)
        status = st.empty()
        
        tickers = [f"{x}.TW" for x in raw_list]
        try:
            batch = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]
            qualified = []
            for code in raw_list:
                try:
                    p = batch.get(f"{code}.TW", np.nan)
                    if not np.isnan(p) and min_p <= p <= max_p:
                        qualified.append((code, get_stock_name(code), p))
                except: continue
        except: qualified = []

        for i, (code, name, price) in enumerate(qualified):
            bar.progress((i+1)/len(qualified))
            status.text(f"計算指標: {code} {name}")
            
            try:
                cap = get_market_cap_robust(code, price)
                if use_0050 and not (use_warrant or use_box or use_ma) and cap < 300: continue
                df = yf.download(f"{code}.TW", period="1y", progress=False)
                if len(df) < 60: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = calculate_indicators(df)
                cur = df.iloc[-1]
                
                reasons = []
                if use_0050 and cap > 300 and cur['Close'] > cur['MA20']: reasons.append(f"🏆權值")
                if use_warrant and cur['HV'] > 25 and cur['Close'] > cur['Box_High_20']: reasons.append(f"🔥飆速")
                if use_box and cur['Close'] > cur['Box_High_20']: reasons.append("📦破箱")
                if use_ma and cur['MA20'] > cur['MA60'] and cur['Close'] > cur['MA20']: reasons.append("📈多頭")
                if use_bb and df.iloc[-2]['BB_Width'] < 0.15 and cur['Close'] > cur['BB_Up']: reasons.append("⚡布林")
                
                if (use_warrant or use_0050 or use_box or use_ma or use_bb) and not reasons: continue
                
                # 存入 Session State
                st.session_state.scan_results.append({
                    "code": code, "name": name, "price": price, 
                    "reasons": " ".join(reasons), "df": df
                })
            except: continue
            
        bar.empty()
        status.empty()

    # --- 顯示結果 (讀取 Session State) ---
    if st.session_state.scan_results:
        st.success(f"篩選出 {len(st.session_state.scan_results)} 檔。請點擊分析。")
        for item in st.session_state.scan_results:
            with st.container():
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1: st.markdown(f"### 🎯 {item['name']} ({item['code']}) - ${item['price']:.2f}")
                with c2: st.caption(item['reasons'])
                
                # 按鈕
                if st.button(f"🤖 AI 分析 {item['name']}", key=f"btn_scan_{item['code']}"):
                    prompt = f"分析 {item['name']} ({item['code']})。現價 {item['price']}。{item['reasons']}。請給短線操作建議。"
                    with st.spinner("AI 分析中..."):
                        advice = call_ai_engine(prompt, ai_provider, selected_model, api_key)
                        st.info(advice)
                
                st.line_chart(item['df'][['Close', 'MA20', 'BB_Up', 'BB_Low']].iloc[-60:], height=150)
                st.divider()
    elif not st.session_state.scan_results and st.button("清除結果", key="clean_scan"):
        st.write("無資料")

# Tab 3: AI 智能投顧
with tab3:
    st.header("🤖 AI 智能投顧")
    with st.form("ai_adv"):
        c1, c2 = st.columns(2)
        with c1: strat = st.radio("屬性", ("🔥 短期", "🌳 長期"))
        with c2: 
            p_min = st.number_input("Min", value=10.0)
            p_max = st.number_input("Max", value=200.0)
        sub = st.form_submit_button("🚀 篩選候選股")
        
    if sub:
        st.session_state.advisor_results = [] # 清空
        targets = ["2330", "2317", "2454", "2308", "2303", "2881", "2412", "3008", "2603", "3037", "2379", "3034", "3045", "4938", "3017"]
        bar = st.progress(0)
        
        for i, code in enumerate(targets):
            bar.progress((i+1)/len(targets))
            try:
                df = yf.download(f"{code}.TW", period="6mo", progress=False)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                cur = df['Close'].iloc[-1]
                if not (p_min <= cur <= p_max): continue
                
                df = calculate_indicators(df)
                st.session_state.advisor_results.append({
                    "code": code, "name": get_stock_name(code), "df": df, "strat": strat
                })
            except: continue
        bar.empty()
        
    # --- 顯示結果 (讀取 Session State) ---
    if st.session_state.advisor_results:
        st.success(f"篩選出 {len(st.session_state.advisor_results)} 檔。請手動分析。")
        for stock in st.session_state.advisor_results:
            with st.container():
                st.markdown(f"**{stock['name']} ({stock['code']})**")
                if st.button(f"分析 {stock['name']}", key=f"adv_{stock['code']}"):
                    prompt = f"針對 {stock['name']} 給出{stock['strat']}建議。參考技術指標。"
                    with st.spinner("AI 運算中..."):
                        res = call_ai_engine(prompt, ai_provider, selected_model, api_key)
                        st.info(res)
                st.line_chart(stock['df']['Close'].iloc[-60:], height=100)
                st.divider()