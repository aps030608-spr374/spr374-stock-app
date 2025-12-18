import streamlit as st
import yfinance as yf
import pandas as pd
import twstock

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="台股全方位掃描器", layout="wide")
st.title("🌪️ 台股全方位掃描器：價格快篩 -> 技術精選")
st.markdown("---")

# --- 2. 側邊欄：設定篩選條件 ---
st.sidebar.header("🎯 第一步：選擇掃描範圍")

# 模式切換：熱門股 vs 全台股
list_mode = st.sidebar.radio(
    "請選擇名單來源：",
    ("🚀 熱門股 (速度快, 測試用)", "🐢 全台上市股票 (約980檔, 需時較久)")
)

if list_mode == "🚀 熱門股 (速度快, 測試用)":
    # 預設熱門股清單 (包含電子、傳產、金融、航運)
    default_list = "2330, 2317, 2454, 2308, 2303, 2881, 2412, 2382, 3008, 2882, 2886, 2891, 1216, 2002, 2884, 2207, 1101, 2892, 5880, 5871, 2357, 2885, 3231, 2345, 3045, 2912, 4904, 2880, 2883, 2887, 2603, 3034, 3711, 2379, 3037, 2327, 2408, 2395, 2609, 2615, 4938, 1590, 5876, 2801, 6669, 6505, 3017, 2301, 1605, 9910, 3481, 2409, 6116, 2481, 2356, 2353"
    user_tickers = st.sidebar.text_area("觀察名單 (可手動增減)", default_list, height=150)
else:
    # 自動抓取 twstock 內的所有上市股票
    st.sidebar.info("正在載入全台股名單...請稍候")
    # 過濾條件：type='股票' 且 market='上市'
    all_listed = [code for code, info in twstock.codes.items() if info.type == '股票' and info.market == '上市']
    all_listed_str = ", ".join(all_listed)
    user_tickers = st.sidebar.text_area("已載入全上市名單 (建議勿手動修改)", all_listed_str, height=150)
    st.sidebar.warning(f"⚠️ 共 {len(all_listed)} 檔。第一階段價格下載約需 1-2 分鐘，請耐心等待。")

st.sidebar.markdown("---")
st.sidebar.header("💰 第二步：價格過濾 (第一層)")
min_price = st.sidebar.number_input("最低價 (元)", value=20)
max_price = st.sidebar.number_input("最高價 (元)", value=100)

st.sidebar.markdown("---")
st.sidebar.header("📈 第三步：技術指標 (第二層)")
use_kd = st.sidebar.checkbox("開啟 KD 黃金交叉篩選", value=True)
use_vol = st.sidebar.checkbox("開啟 爆量篩選", value=True)
vol_pct = st.sidebar.slider("成交量增幅至少 %", 10, 100, 20) / 100

# --- 3. 核心函數區 ---

# 查股票中文名稱
def get_stock_name(code):
    try:
        if code in twstock.codes:
            return twstock.codes[code].name
        return code
    except:
        return code

# 計算 KD 值
def calculate_kd(df):
    try:
        # 計算 RSV
        low_min = df['Low'].rolling(window=9).min()
        high_max = df['High'].rolling(window=9).max()
        rsv = 100 * (df['Close'] - low_min) / (high_max - low_min)
        rsv = rsv.fillna(50)
        
        # 計算 K, D
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

# --- 4. 主程式執行邏輯 ---
if st.button("🚀 開始執行篩選"):
    
    # --- 步驟 0: 準備清單 ---
    raw_list = [x.strip() for x in user_tickers.split(",") if x.strip()]
    # 確保代碼有 .TW (yfinance 需要)
    ticker_list_tw = [f"{x}.TW" if not x.upper().endswith(".TW") else x for x in raw_list]
    
    st.subheader(f"🏁 階段一：價格快篩 (目標掃描：{len(raw_list)} 檔)")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # --- 步驟 1: 批次下載價格 (Batch Download) ---
    status_text.text("正在批次下載最新股價... (全台股模式會停在這裡比較久，是正常的)")
    
    try:
        # 一次抓取所有股票的「最新一天」資料
        batch_data = yf.download(ticker_list_tw, period="1d", progress=False)
        progress_bar.progress(30)
        
        # 整理 current_prices (處理 Series 或 DataFrame 的差異)
        if len(ticker_list_tw) > 1:
            # 檢查是否有抓到資料
            if 'Close' in batch_data:
                current_prices = batch_data['Close'].iloc[-1]
            else:
                st.error("無法取得股價資料，可能是網路問題或代碼錯誤。")
                st.stop()
        else:
            # 單檔股票處理
            current_prices = pd.Series({ticker_list_tw[0]: batch_data['Close'].iloc[-1]})

        # --- 篩選符合價格區間的股票 ---
        price_qualified_tickers = []
        
        for code_tw in ticker_list_tw:
            try:
                # 某些股票可能沒抓到資料(下市或錯誤)，用 try 接住
                if code_tw in current_prices:
                    price = current_prices[code_tw]
                    
                    if min_price <= price <= max_price:
                        # 取得純代碼
                        clean_code = code_tw.replace(".TW", "").replace(".tw", "")
                        name = get_stock_name(clean_code)
                        price_qualified_tickers.append((clean_code, name, price))
            except:
                continue
        
        progress_bar.progress(50)
        status_text.text(f"價格篩選完成！剩餘 {len(price_qualified_tickers)} 檔進入第二階段...")
        
        # 顯示第一階段結果 (可摺疊)
        st.success(f"✅ 符合價格區間 ({min_price}~{max_price}元)：共 {len(price_qualified_tickers)} 檔")
        with st.expander("👀 點擊查看【通過價格篩選】的名單"):
            if price_qualified_tickers:
                p_df = pd.DataFrame(price_qualified_tickers, columns=["代碼", "名稱", "現價"])
                p_df["現價"] = p_df["現價"].map("{:.2f}".format)
                st.dataframe(p_df, use_container_width=True)
            else:
                st.write("無符合資料")

    except Exception as e:
        st.error(f"發生錯誤：{e}")
        st.stop()

    # --- 步驟 2: 技術指標精選 (迴圈處理) ---
    if price_qualified_tickers:
        st.write("---")
        st.subheader(f"🔬 階段二：技術指標運算 (KD & 成交量)")
        
        final_results = []
        total_q = len(price_qualified_tickers)
        
        # 建立一個顯示區域
        scan_status = st.empty()
        
        for i, (code, name, price) in enumerate(price_qualified_tickers):
            # 更新進度條 (從 50% 開始跑)
            current_progress = 0.5 + 0.5 * ((i + 1) / total_q)
            progress_bar.progress(current_progress)
            scan_status.text(f"正在分析技術面：{code} {name} ... ({i+1}/{total_q})")
            
            try:
                # 下載歷史資料 (3個月)
                df = yf.download(f"{code}.TW", period="3mo", progress=False)
                
                # 清理 MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # 資料太少就跳過
                if df.empty or len(df) < 20: continue

                # 計算 KD
                df = calculate_kd(df)
                today = df.iloc[-1]
                yesterday = df.iloc[-2]
                
                # 判定 1: KD 黃金交叉
                match_kd = True
                kd_msg = "無"
                if use_kd:
                    # 昨K < 昨D  AND  今K > 今D
                    is_gc = (yesterday['K'] < yesterday['D']) and (today['K'] > today['D'])
                    match_kd = is_gc
                    kd_msg = "✅ 黃金交叉" if is_gc else "❌"

                # 判定 2: 爆量
                match_vol = True
                vol_msg = "無"
                if use_vol:
                    # 今日量 > 昨日量 * (1 + 增幅)
                    target_vol = yesterday['Volume'] * (1 + vol_pct)
                    is_vol_up = today['Volume'] > target_vol
                    match_vol = is_vol_up
                    vol_msg = "✅ 爆量" if is_vol_up else "❌"
                
                # 綜合判定
                if match_kd and match_vol:
                    final_results.append({
                        "代碼": code,
                        "名稱": name,
                        "收盤價": f"{today['Close']:.2f}",
                        "K值": f"{today['K']:.2f}",
                        "D值": f"{today['D']:.2f}",
                        "成交量": int(today['Volume']),
                        "KD狀態": kd_msg,
                        "成交量狀態": vol_msg
                    })
                    
            except Exception as e:
                continue
        
        # 掃描結束
        progress_bar.progress(100)
        scan_status.empty() # 清除文字
        
        if final_results:
            st.balloons() # 慶祝動畫
            st.markdown(f"### 🎉 最終篩選結果：共 {len(final_results)} 檔潛力股")
            
            # 整理並顯示最終表格
            res_df = pd.DataFrame(final_results)
            st.dataframe(res_df, use_container_width=True)
            
            st.success("分析完成！請參考上方數據進行決策。")
        else:
            st.warning("⚠️ 價格符合，但沒有股票符合您的技術指標條件。試著放寬「成交量增幅」或關閉 KD 篩選。")

    else:
        st.warning("第一階段價格篩選後無股票入選，請調整價格區間。")

else:
    st.info("👈 請在左側設定條件，並點擊按鈕開始掃描。")