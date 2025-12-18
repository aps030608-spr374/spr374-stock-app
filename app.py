import streamlit as st
import yfinance as yf
import pandas as pd
import twstock  # 引入 twstock 用來查中文名

# --- 設定網頁 ---
st.set_page_config(page_title="超級選股漏斗", layout="wide")
st.title("🌪️ 兩階段選股：價格快篩 -> 技術精選 (含名稱)")

# --- 側邊欄：設定條件 ---
st.sidebar.header("1. 選擇族群")
default_list = "2330, 2317, 2454, 2308, 2303, 2881, 2412, 2382, 3008, 2882, 2886, 2891, 1216, 2002, 2884, 2207, 1101, 2892, 5880, 5871, 2357, 2885, 3231, 2345, 3045, 2912, 4904, 2880, 2883, 2887, 2603, 3034, 3711, 2379, 3037, 2327, 2408, 2395, 2609, 2615, 4938, 1590, 5876, 2801, 6669, 6505, 3017, 2301, 1605, 9910, 3481, 2409, 6116, 2481, 2356, 2353"
user_tickers = st.sidebar.text_area("觀察名單 (逗號隔開)", default_list, height=150)

st.sidebar.header("2. 第一層：價格篩選")
min_price = st.sidebar.number_input("最低價 (元)", value=50)
max_price = st.sidebar.number_input("最高價 (元)", value=150)

st.sidebar.header("3. 第二層：技術指標")
use_kd = st.sidebar.checkbox("KD 黃金交叉", value=True)
use_vol = st.sidebar.checkbox("爆量 (成交量增幅)", value=True)
vol_pct = st.sidebar.slider("增幅 %", 10, 100, 20) / 100

# --- 輔助函數：查股票中文名 ---
def get_stock_name(code):
    try:
        # twstock.codes 是一個字典，可以直接用代碼查資料
        if code in twstock.codes:
            return twstock.codes[code].name
        else:
            return code # 查不到就回傳代碼
    except:
        return code

# --- KD 計算函數 ---
def calculate_kd(df):
    try:
        low_min = df['Low'].rolling(window=9).min()
        high_max = df['High'].rolling(window=9).max()
        rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
        rsv = rsv.fillna(50)
        
        k_values = [50]
        d_values = [50]
        
        for i in range(1, len(df)):
            k = (2/3) * k_values[-1] + (1/3) * rsv.iloc[i]
            d = (2/3) * d_values[-1] + (1/3) * k
            k_values.append(k)
            d_values.append(d)
        
        df['K'] = k_values
        df['D'] = d_values
        return df
    except:
        return df

# --- 主程式 ---
if st.button("🚀 開始兩階段篩選"):
    
    # 0. 整理代碼清單
    raw_list = [x.strip() for x in user_tickers.split(",") if x.strip()]
    # 自動補上 .TW 給 yfinance 用
    ticker_list_tw = [f"{x}.TW" if not x.upper().endswith(".TW") else x for x in raw_list]
    
    st.write(f"### 🏁 階段一：價格快篩 (共 {len(raw_list)} 檔)")
    
    try:
        # 批次下載最新股價 (只抓 1 天，速度最快)
        batch_data = yf.download(ticker_list_tw, period="1d", progress=False)
        
        # 處理 yfinance 回傳格式
        if len(ticker_list_tw) > 1:
            current_prices = batch_data['Close'].iloc[-1]
        else:
            current_prices = pd.Series({ticker_list_tw[0]: batch_data['Close'].iloc[-1]})

        # 篩選符合價格區間的股票
        price_qualified_tickers = []
        
        for code_tw in ticker_list_tw:
            try:
                price = current_prices[code_tw]
                
                # 價格判斷
                if min_price <= price <= max_price:
                    # 取得純數字代碼 (去掉 .TW) 用來查名字
                    clean_code = code_tw.replace(".TW", "").replace(".tw", "")
                    stock_name = get_stock_name(clean_code) # 查中文名
                    
                    # 存入清單：(代碼, 名稱, 價格)
                    price_qualified_tickers.append((clean_code, stock_name, price))
            except:
                continue

        # 顯示第一階段結果
        st.success(f"✅ 價格符合 ({min_price}~{max_price}元)：共 {len(price_qualified_tickers)} 檔")
        
        with st.expander("👀 查看通過價格篩選的名單"):
            # 建立表格顯示
            if price_qualified_tickers:
                price_df = pd.DataFrame(price_qualified_tickers, columns=["代碼", "名稱", "目前股價"])
                # 股價格式化小數點
                price_df["目前股價"] = price_df["目前股價"].map("{:.2f}".format)
                st.dataframe(price_df, use_container_width=True)
            else:
                st.write("無符合資料")

    except Exception as e:
        st.error(f"下載資料時發生錯誤：{e}")
        price_qualified_tickers = []

    # --- 階段二：技術指標精選 ---
    if price_qualified_tickers:
        st.write("---")
        st.write(f"### 🔬 階段二：技術分析掃描 (針對剩下的 {len(price_qualified_tickers)} 檔)")
        
        final_results = []
        progress_bar = st.progress(0)
        total = len(price_qualified_tickers)
        
        # 這裡的迴圈會同時拿到 code (代碼) 和 name (名稱)
        for i, (code, name, price) in enumerate(price_qualified_tickers):
            progress_bar.progress((i + 1) / total)
            
            try:
                stock_id = f"{code}.TW"
                df = yf.download(stock_id, period="3mo", progress=False)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                if df.empty or len(df) < 20: continue

                df = calculate_kd(df)
                today = df.iloc[-1]
                yesterday = df.iloc[-2]
                
                match_kd = True
                if use_kd:
                    match_kd = (yesterday['K'] < yesterday['D']) and (today['K'] > today['D'])
                
                match_vol = True
                if use_vol:
                    match_vol = today['Volume'] > (yesterday['Volume'] * (1 + vol_pct))
                
                if match_kd and match_vol:
                    final_results.append({
                        "代碼": code,
                        "名稱": name,  # 這裡加入名稱
                        "收盤價": f"{today['Close']:.2f}",
                        "K值": f"{today['K']:.2f}",
                        "D值": f"{today['D']:.2f}",
                        "成交量": int(today['Volume']),
                        "訊號": "🌟 入選"
                    })
                    
            except:
                continue
        
        if final_results:
            st.balloons()
            st.markdown(f"### 🎉 最終精選：{len(final_results)} 檔")
            st.dataframe(pd.DataFrame(final_results), use_container_width=True)
        else:
            st.warning("⚠️ 價格過濾後，沒有股票符合技術指標條件。")
    else:
        st.warning("⚠️ 沒有股票符合價格區間，無法進行第二階段篩選。")

else:
    st.info("👈 設定好區間後，按上面的按鈕開始！")