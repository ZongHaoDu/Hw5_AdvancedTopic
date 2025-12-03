# This file will contain the UI elements, like the sidebar.
import streamlit as st

def display_sidebar():
    st.sidebar.title("關於此應用")
    st.sidebar.info(
        "這是一個高階 AI 文本偵測器 Demo，它使用多種語言學和機器學習模型來分析文本，並評估其是否可能由 AI 生成。"
    )

    st.sidebar.title("核心偵測演算法")
    
    with st.sidebar.expander("1. Perplexity (困惑度)"):
        st.write("""
        **理論**: Perplexity 衡量一個機率模型對新樣本的預測能力的好壞。在 NLP 中，它測量模型對一個文本序列的「驚訝程度」。
        - **人類寫作**: 通常在句法和語義上更具變化性，可能導致 Perplexity 較高或波動較大。
        - **AI 生成**: 傾向於選擇高機率、可預測的詞彙，因此 Perplexity 通常較低且平穩。
        **圖表**: 「PP 時間序列圖」顯示文本各部分的 Perplexity 值，幫助觀察其穩定性。
        """)

    with st.sidebar.expander("2. Burstiness (句子節奏)"):
        st.write("""
        **理論**: Burstiness 描述事件發生的叢集程度。在文本中，它指句子長度的變化節奏。
        - **人類寫作**: 傾向於混合使用長句和短句，句子長度分布不均勻，導致 Burstiness 值較高。
        - **AI 生成**: 傾向於生成結構和長度更均勻的句子，因此 Burstiness 值較低。
        **圖表**: 「句長分布直方圖」顯示了文本中句子長度的分布情況。
        """)

    with st.sidebar.expander("3. Stylometry (風格學)"):
        st.write("""
        **理論**: 風格學分析作者的寫作風格。我們關注兩個指標：
        - **詞彙多樣性 (TTR - Type-Token Ratio)**: 衡量用詞的豐富程度。TTR = (獨立詞彙數 / 總詞彙數)。AI 可能會重複使用某些詞彙，導致 TTR 偏低。
        - **詞性 (POS) 分布**: 分析名詞、動詞、形容詞等詞性的使用比例。人類和 AI 在此可能有不同的模式。
        """)

    with st.sidebar.expander("4. Zipf's Law (長尾分布)"):
        st.write("""
        **理論**: Zipf's Law 是一個經驗法則，它指出在自然語言中，任何單詞的頻率與其在頻率表中的排名成反比。
        - **人類寫作**: 通常非常貼合 Zipf 曲線，呈現出典型的「長尾」特徵 (少數詞彙高頻，大量詞彙低頻)。
        - **AI 生成**: 可能會過度使用中頻詞彙，而低頻的長尾詞彙不足，導致曲線與標準 Zipf 曲線偏離。
        **圖表**: 在 log-log 尺度上比較文本的詞頻分布與理想的 Zipf 曲線。
        """)

    with st.sidebar.expander("5. Semantic Drift (語意漂移)"):
        st.write("""
        **理論**: 這個指標衡量文本中語義連貫性的變化。
        - **人類寫作**: 在段落或主題轉換時，語意上可能會有自然的「跳躍」，但整體上保持連貫。
        - **AI 生成**: 可能會過於平滑，句子之間的語義距離非常一致；或者在上下文中產生不連貫的跳躍。
        **圖表**: 「語意軌跡散佈圖」通過 PCA 將句子向量降維到 2D 空間，視覺化句子之間語義關係的路徑。
        """)

# def display_test_samples():
#     st.sidebar.title("AI vs. 人類挑戰")
#     st.sidebar.write("試著猜猜看，下面哪段文字是 AI 生成的？點擊展開閱讀。")

#     # --- 文本範例 ---
#     human_text = """
#     說真的，我上週末去爬那座山，一開始還覺得天氣不錯，沒想到半路就給我下大雨！超狼狽的，鞋子跟褲子全都濕透了，而且根本沒帶傘，只能躲在一個超小的涼亭下面等雨停。旁邊還有一對情侶在吵架，真的是...有夠尷尬。不過雨停了之後，空氣聞起來超清新，還看到了彩虹，那一刻又覺得好像一切都值了。人生嘛，不就是這樣，起起落落的。
#     """

#     ai_text = """
#     綜合考量各項因素，上週末的山區健行活動在初期階段的天氣條件尚屬良好。然而，行程中段遭遇了未預期的強降雨，導致參與者的衣物和鞋履完全濕透。由於缺乏適當的雨具，參與者不得不暫時於一個小型遮蔽結構下避雨。此期間，觀察到鄰近的兩位個體發生了口頭爭執，營造了一種社交上不甚舒適的氛圍。降雨停止後，環境空氣品質顯著提升，並觀測到大氣光學現象——彩虹，這為整個體驗帶來了正面的情感轉折。此事件可視為一個隱喻，說明生活中不可預測的挑戰與其後可能出現的意外收穫並存。
#     """
    
#     # 為了讓使用者無法輕易根據順序猜出來，我們隨機決定哪個先顯示
#     import random
    
#     # 使用 session state 來確保 choice 在 reruns 之間保持不變
#     if 'challenge_order' not in st.session_state:
#         st.session_state.challenge_order = random.choice(['human_first', 'ai_first'])

#     if st.session_state.challenge_order == 'human_first':
#         with st.sidebar.expander("文本範例 1"):
#             st.markdown(f"_{human_text}_")

#         with st.sidebar.expander("文本範例 2"):
#             st.markdown(f"_{ai_text}_")
#     else:
#         with st.sidebar.expander("文本範例 1"):
#             st.markdown(f"_{ai_text}_")

#         with st.sidebar.expander("文本範例 2"):
#             st.markdown(f"_{human_text}_")
            
#     # 提供一個按鈕讓使用者揭曉答案
#     if st.sidebar.button("揭曉答案"):
#         if st.session_state.challenge_order == 'human_first':
#             st.sidebar.success("答案：\n- **文本 1** 是人類寫的\n- **文本 2** 是 AI 生成的")
#         else:
#             st.sidebar.success("答案：\n- **文本 1** 是 AI 生成的\n- **文本 2** 是人類寫的")
#         # 答完後可以清除 state，讓下一次挑戰重新隨機
#         del st.session_state.challenge_order