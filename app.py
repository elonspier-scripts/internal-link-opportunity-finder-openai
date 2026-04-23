import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
import plotly.express as px
import plotly.graph_objects as go

# Pagina instellingen
st.set_page_config(page_title="Semantic Link Finder & Plotly Matrix", layout="wide")

st.title("🔗 Semantic Link Finder & Matrix")
st.markdown("Vind linkkansen en visualiseer de verbanden tussen je Topic Hubs met Plotly.")

# ========================================================
# 1. SIDEBAR CONFIGURATIE
# ========================================================
st.sidebar.header("🔑 API Instellingen")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.secrets.get("OPENAI_KEY", ""))

st.sidebar.divider()
st.sidebar.header("⚙️ Analyse Instellingen")
min_score = st.sidebar.slider("Minimale Similarity Score (%)", 50, 95, 80) / 100
max_links = st.sidebar.slider("Max targets per URL", 1, 10, 5)

# ========================================================
# 2. HELPERS
# ========================================================
def clean_url_for_text(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    clean_path = re.sub(r'[-_/]', ' ', path)
    return clean_path

def get_embeddings(text_list, api_key):
    client = OpenAI(api_key=api_key)
    embeddings = []
    progress_bar = st.progress(0)
    total_items = len(text_list)
    batch_size = 100
    for i in range(0, total_items, batch_size):
        batch = text_list[i:i+batch_size]
        try:
            response = client.embeddings.create(input=batch, model='text-embedding-3-small')
            embeddings.extend([data.embedding for data in response.data])
        except Exception as e:
            st.error(f"Fout: {e}")
            st.stop()
        progress_bar.progress(min((i + batch_size) / total_items, 1.0))
    progress_bar.empty()
    return np.array(embeddings)

def extract_topical_category(text):
    words = re.findall(r'\w{4,}', str(text).lower())
    stop_words = {'deze', 'voor', 'naar', 'met', 'door', 'geen', 'over', 'mijn', 'niet'}
    filtered = [w for w in words if w not in stop_words]
    unique_words = list(dict.fromkeys(filtered))
    return " / ".join(unique_words[:2]).upper() if unique_words else "ALGEMEEN"

def apply_color(val):
    if isinstance(val, (int, float, np.integer)):
        if val <= 60: return 'color: #dc3545; font-weight: bold;'
        elif val <= 84: return 'color: #ffc107; font-weight: bold;'
        else: return 'color: #28a745; font-weight: bold;'
    return ''

# ========================================================
# 3. INPUT
# ========================================================
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
with col2:
    focus_urls_input = st.text_area("Focus URL's", height=150, placeholder="https://site.nl/doel")

# ========================================================
# 4. ANALYSE
# ========================================================
if st.button("🚀 Start Analyse", type="primary"):
    if not openai_key or not uploaded_file or not focus_urls_input:
        st.error("Vul alle velden in.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            url_col = df.columns[0]
            content_cols = df.columns[1:]
            focus_urls = [x.strip() for x in focus_urls_input.split('\n') if x.strip()]
            
            df = df[df[url_col].str.strip() != ""].copy()
            df['url_text'] = df[url_col].apply(clean_url_for_text)
            df['combined_text'] = df.apply(lambda row: str(row['url_text']) + ' ' + ' '.join(row[content_cols].values.astype(str)), axis=1)
            df['Topical_Category'] = df['combined_text'].apply(extract_topical_category)

            # Mapping URL -> Categorie
            url_to_cat = dict(zip(df[url_col], df['Topical_Category']))

            st.info("⚙️ Embeddings genereren...")
            vectors = get_embeddings(df['combined_text'].tolist(), openai_key)
            sim_matrix = cosine_similarity(vectors)

            results = []
            for focus_url in focus_urls:
                if focus_url not in df[url_col].values: continue
                idx_source = df.index[df[url_col] == focus_url].tolist()[0]
                source_topic = df.iloc[idx_source]['Topical_Category']
                similarities = sim_matrix[idx_source]
                sorted_indices = np.argsort(similarities)[::-1]
                
                count = 0
                for idx_target in sorted_indices:
                    target_url = df.iloc[idx_target][url_col]
                    score = float(similarities[idx_target])
                    if focus_url != target_url and score >= min_score:
                        results.append({
                            'Source Category': source_topic,
                            'Focus URL': focus_url,
                            'Target URL': target_url,
                            'Target Category': url_to_cat.get(target_url, "OVERIG"),
                            'Score (%)': round(score * 100)
                        })
                        count += 1
                        if count >= max_links: break

            if results:
                res_df = pd.DataFrame(results)
                
                # --- PLOTLY MATRIX SECTIE ---
                st.subheader("📊 Cross-Linking Matrix (Intensity Heatmap)")
                
                # Pivot table maken
                matrix = pd.crosstab(res_df['Source Category'], res_df['Target Category'])
                all_cats = sorted(list(set(res_df['Source Category']).union(set(res_df['Target Category']))))
                matrix = matrix.reindex(index=all_cats, columns=all_cats, fill_value=0)

                # Plotly Heatmap configuratie
                fig = px.imshow(
                    matrix,
                    labels=dict(x="Target Category (Linkt naar)", y="Source Category (Linkt vanaf)", color="Aantal Links"),
                    x=matrix.columns,
                    y=matrix.index,
                    color_continuous_scale='Blues', # Intensity look
                    aspect="auto",
                    text_auto=True # Toon getallen in de cellen
                )

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="ONTAVNGENDE CATEGORIE (Target)",
                    yaxis_title="VERZENDENDE CATEGORIE (Source)",
                    font=dict(color="white")
                )

                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()

                # --- HUBS SECTIE ---
                st.subheader("🏗️ Topic Hubs")
                hub_ranking = res_df.groupby('Source Category')['Score (%)'].mean().sort_values(ascending=False)
                for cat in hub_ranking.index:
                    avg_s = round(hub_ranking[cat])
                    with st.expander(f"📁 {cat} (Gem. score: {avg_s}%)"):
                        cat_disp = res_df[res_df['Source Category'] == cat].copy()
                        cat_disp.loc[cat_disp.duplicated('Focus URL'), 'Focus URL'] = ""
                        st.dataframe(
                            cat_disp[['Focus URL', 'Target URL', 'Score (%)']].style.map(apply_color, subset=['Score (%)']),
                            use_container_width=True
                        )

                # Download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Data')
                    matrix.to_excel(writer, sheet_name='Matrix')
                st.download_button("📥 Download Excel", output.getvalue(), "link_finder_results.xlsx")

            else:
                st.warning("Geen resultaten gevonden.")
        except Exception as e:
            st.error(f"Fout: {e}")
