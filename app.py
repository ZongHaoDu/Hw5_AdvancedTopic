import streamlit as st
import analysis
import plotting
import ui
import numpy as np

def main():
    st.set_page_config(layout="wide", page_title="Advanced AI Text Detector")
    
    # Use columns for a cleaner layout
    col1, col2 = st.columns([1, 2])

    with col1:
        ui.display_sidebar()
        # ui.display_test_samples()

    with col2:
        st.title("高階 AI 文本偵測器 (Advanced AI Text Detector)")
        st.header("請在此處輸入您要分析的文本")
        text_input = st.text_area("Text to analyze", height=250, label_visibility="collapsed", placeholder="貼上文本於此 (Paste text here)...")

        if st.button("開始分析 (Analyze)"):
            if text_input:
                # This is a long process, so the spinner is important
                with st.spinner("正在深度分析文本...過程可能需要數分鐘，請耐心等候... (Performing deep analysis... this may take several minutes, please wait...)"):
                    
                    # --- 1. Run all analyses ---
                    all_metrics = {}
                    
                    avg_ppl, ppl_scores = analysis.calculate_perplexity(text_input)
                    all_metrics['avg_perplexity'] = avg_ppl
                    
                    burstiness_score, sent_lengths = analysis.calculate_burstiness(text_input)
                    all_metrics['burstiness'] = burstiness_score
                    
                    ttr_score, pos_dist = analysis.calculate_stylometry(text_input)
                    all_metrics['ttr'] = ttr_score
                    
                    zipf_data = analysis.calculate_zipf(text_input)
                    
                    semantic_data = analysis.calculate_semantic_drift(text_input)
                    if semantic_data:
                        all_metrics['avg_drift'] = semantic_data.get('avg_drift')

                    # --- 2. Calculate Final Score ---
                    final_score = analysis.calculate_final_score(all_metrics)

                    # Handle potential NaN score if metrics are zero or invalid
                    if final_score is None or np.isnan(final_score):
                        st.warning("Could not reliably compute a final score, likely due to very short or unusual input text. Score has been defaulted to 0.")
                        final_score = 0

                    st.success("分析完成！(Analysis Complete!)")
                    st.divider()

                    # --- 3. Display Final Score ---
                    st.header("綜合 AI 疑似度 (Comprehensive AI Likelihood)")
                    
                    # Determine color based on score
                    if final_score < 40:
                        color = "green"
                    elif final_score < 70:
                        color = "orange"
                    else:
                        color = "red"
                    
                    # Custom HTML for larger text and color
                    st.markdown(f"""
                    <style>
                    .big-font {{
                        font-size:32px !important;
                        font-weight: bold;
                        color: {color};
                    }}
                    </style>
                    <p class="big-font">{final_score:.2f}%</p>
                    """, unsafe_allow_html=True)
                    
                    st.progress(int(final_score))
                    st.info("此分數為綜合所有指標的啟發式評估，分數越高，由 AI 生成的可能性越大。僅供參考。")

                    st.divider()

                    # --- 4. Display Individual Metrics ---
                    st.header("各項指標細節 (Metric Details)")
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                    with res_col1:
                        st.metric(label="Avg. Perplexity", value=f"{avg_ppl:.2f}")
                    with res_col2:
                        st.metric(label="Burstiness", value=f"{burstiness_score:.4f}")
                    with res_col3:
                        st.metric(label="Lexical Diversity (TTR)", value=f"{ttr_score:.4f}")
                    if semantic_data:
                        with res_col4:
                            st.metric(label="Semantic Drift", value=f"{semantic_data['avg_drift']:.4f}")
                    
                    st.divider()

                    # --- 5. Display Plots ---
                    st.subheader("1. Perplexity (困惑度) 時間序列圖")
                    perplexity_fig = plotting.plot_perplexity(ppl_scores, avg_ppl)
                    st.plotly_chart(perplexity_fig, use_container_width=True)
                    st.info("Perplexity 衡量模型對文本的「驚訝程度」。AI 生成的文本通常更可預測，因此 Perplexity 較低且平穩。")

                    st.divider()

                    st.subheader("2. 句長分布 (Sentence Length Distribution)")
                    burstiness_fig = plotting.plot_burstiness(sent_lengths)
                    st.plotly_chart(burstiness_fig, use_container_width=True)
                    st.info("人類寫作的句子長度通常變化較大 (高 Burstiness)，而 AI 生成的文本則更趨於一致 (低 Burstiness)。")
                    
                    st.divider()

                    st.subheader("3. 詞性分布 (Part-of-Speech Distribution)")
                    pos_fig = plotting.plot_pos_distribution(pos_dist)
                    st.plotly_chart(pos_fig, use_container_width=True)
                    
                    st.divider()

                    st.subheader("4. Zipf's Law (長尾分布)")
                    zipf_fig = plotting.plot_zipf(zipf_data)
                    st.plotly_chart(zipf_fig, use_container_width=True)
                    st.info("此圖比較了文本的實際詞頻分布（藍點）與理想的 Zipf 曲線（紅線）。AI 生成的文本可能缺乏低頻的「長尾」詞彙。")

                    st.divider()

                    st.subheader("5. 語意軌跡 (Semantic Trajectory)")
                    semantic_fig = plotting.plot_semantic_drift(semantic_data)
                    st.plotly_chart(semantic_fig, use_container_width=True)
                    st.info("此圖將每個句子視覺化為 2D 空間中的一個點。AI 生成的文本可能有更平滑、可預測的軌跡。")

            else:
                st.warning("請輸入文本以進行分析 (Please enter text to analyze)")




if __name__ == "__main__":
    main()
