import streamlit as st
import numpy as np

def render(model, df, index_en, index_fr):

    # Search controls
    col1, col2, col3 = st.columns([5, 1.2, 1])

    with col1:
        query = st.text_input(
            "Search",
            placeholder="e.g. water quality monitoring Ontario  /  qualité de l'eau Ontario",
            label_visibility="visible"
        )
    with col2:
        lang = st.selectbox(
            "Language",
            options=["en", "fr"],
            format_func=lambda x: "EN English Embedding" if x == "en" else "FR French Embedding",
            label_visibility="visible",
            help="Compare the query vector with:\n\n EN → English embeddings\n\n FR → French embeddings"
        )
    with col3:
        top_k = st.selectbox(
            "Results",
            options=[5, 10, 15],
            label_visibility="visible",
            help="Number of results to display"
        )

    # Search logic
    if query:
        with st.spinner("🔍 Searching..."):
            vec   = model.encode(f"query: {query}")
            vec   = vec / np.linalg.norm(vec)
            vec   = vec.astype('float32').reshape(1, -1)
            index = index_en if lang == 'en' else index_fr
            scores, indices = index.search(vec, top_k)
            indices = indices[0].tolist()
            scores  = scores[0].tolist()

        st.markdown(f"**{len(indices)} results** for *'{query}'* in **{'English' if lang == 'en' else 'French'}** index")
        st.divider()

        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            row = df.iloc[idx]
            score_color = (
                "#4CAF50" if score >= 0.85 else
                "#FF9800" if score >= 0.75 else
                "#F44336"
            )
            desc = str(row['desc_fr']) if lang == 'fr' else str(row['desc_en'])

            st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span class="rank-badge">#{rank}</span>
                        <span class="score-badge" style="background-color:{score_color}">
                            Score: {score:.4f}
                        </span>
                    </div>
                    <div class="title-text">🇨🇦 {row['title_en']}</div>
                    <div class="title-text" style="color:#aaaaaa; font-size:14px;">
                        🇫🇷 {row['title_fr']}
                    </div>
                    <div class="desc-text">{desc[:300]}...</div>
                    <div class="org-text">
                        🏛️ {row['org']} &nbsp;|&nbsp; 📂 {row['subject']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(float(score))

    else:
        st.markdown("""
            <div style="text-align:center; padding:50px 0; color:#444444;">
                <div style="font-size:40px;">🔍</div>
                <div style="font-size:16px; margin-top:10px;">Type a query above to search</div>
                <div style="font-size:13px; margin-top:8px; color:#555555;">
                    Try: "water quality Ontario" · "forest fire satellite" · "climate change Canada"<br>
                    En français: "qualité de l'eau" · "incendies de forêt" · "changements climatiques"
                </div>
            </div>
        """, unsafe_allow_html=True)