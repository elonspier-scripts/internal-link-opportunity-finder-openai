import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import re
import io
import json
from urllib.parse import urlparse

# 🚀 1. PERFORMANCE & STYLING CONFIG
# Verhoogt de limiet voor grote matrices en voorkomt de StreamlitAPIException
pd.set_option("styler.render.max_elements", 1000000)

st.set_page_config(page_title="SEO Link Matrix Pro", layout="wide")

# ========================================================
# 2. UI STYLING (Deep Black / Cyber Blue)
# ========================================================
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1e1e1e; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #0a0a0a !important; color: #00a2ff !important; border: 1px solid #222 !important;
    }
    .stDataFrame, div[data-testid="stTable"] { 
        background-color: #000 !important; border: 1px solid #1e1e1e !important; border-radius: 4px;
    }
    h1, h2, h3 { color: #00a2ff !important; font-family: 'Inter', sans-serif; }
    .stButton>button { 
        background: linear-gradient(135deg, #0044ff 0%, #00a2ff 100%);
        color: white; border: none; padding: 12px; font-weight: bold; width: 100%;
        box-shadow: 0 4px 20px rgba(0, 162, 255, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# ========================================================
# 3. INITIALISATIE
# ========================================================
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

with st.sidebar:
    st.title("⚙️ Configuratie")
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key_val")
    st.divider()
    cluster_threshold = st.slider("Minimale Cluster Match % (Hubs)", 50, 95, 80) / 100
    score_threshold = st.slider("Minimale Link Match % (Links)", 50, 95, 80) / 100
    links_per_page = st.slider("Aantal links per URL", 1, 10, 5)

# ========================================================
# 4. HELPERS (Met Caching voor snelheid)
# ========================================================
@st.cache_data(show_spinner=False)
def clean_path(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    return re.sub(r'[-_/]', ' ', path)

@st.cache_data(show_spinner=False)
def get_folder(url):
    """Extraheert de eerste hoofdmap (top-level folder). /energie/stroom/ -> /energie/"""
    try:
        path = urlparse(str(url)).path
        clean_path = path.strip('/')
        if not clean_path:
            return '/'
        first_folder = clean_path.split('/')[0]
        return f"/{first_folder}/"
    except:
        return "/"

def get_embeddings(texts, key, batch_size=500):
    """Haalt embeddings op in blokken (batches) om Rate Limits te voorkomen"""
    client = OpenAI(api_key=key)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        res = client.embeddings.create(input=batch, model='text-embedding-3-small')
        all_embeddings.extend([d.embedding for d in res.data])
    return np.array(all_embeddings)

def get_ai_cluster_names_bulk(clusters_dict, key):
    """Bedenkt namen voor alle clusters in één enkele API call (Bliksemsnel)"""
    client = OpenAI(api_key=key)
    payload = ""
    for cid, texts in clusters_dict.items():
        sample = "\n".join(texts[:10])
        payload += f"Cluster ID {cid}:\n{sample}\n\n"
        
    prompt = f"""
    Je bent een SEO expert. Bedenk voor ELK cluster één overkoepelende categorie-naam.
    Regels: Max 5 woorden, korte SEO-relevante naam.
    OUTPUT: Strict JSON, Key = Cluster ID, Value = Naam.
    Data: {payload}
    """
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={ "type": "json_object" }
    )
    try:
        result = json.loads(res.choices[0].message.content)
        return {int(k): str(v) for k, v in result.items()}
    except:
        return {}

def color_score(v):
    if not isinstance(v, (int, float)): return ''
    if v >= 85: return 'color: #28a745; font-weight: bold;'
    elif v >= 70: return 'color: #ffc107; font-weight: bold;'
    else: return 'color: #dc3545; font-weight: bold;'

# ========================================================
# 5. DE ANALYSE ENGINE
# ========================================================
st.title("🔗 SEO Link Intelligence Matrix")
t_tool, t_inst = st.tabs(["🚀 Analyse Tool", "📖 Instructies"])

with t_tool:
    c1, c2 = st.columns([1, 1])
    with c1:
        file = st.file_uploader("1. Upload Website CSV", type=['csv'])
    with c2:
        urls_txt = st.text_area("2. Focus URL's (één per regel)", height=100)

    if st.button("🚀 GENEREER INTELLIGENCE MATRIX"):
        if not api_key or not file or not urls_txt:
            st.error("⚠️ Oeps! API Key, CSV en Focus URL's zijn verplicht.")
        else:
            try:
                with st.spinner("AI is bezig met berekeningen... Even geduld."):
                    raw_df = pd.read_csv(file)
                    url_col = raw_df.columns[0]
                    focus_list = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                    
                    # Opschonen
                    df = raw_df.dropna(subset=[url_col]).copy()
                    df = df.fillna("").loc[df[url_col].astype(str).str.strip() != ""]
                    
                    # Context voorbereiden
                    df['text'] = df[url_col].astype(str).apply(clean_path) + " " + df.iloc[:, 1].astype(str)
                    
                    # 1. Embeddings
                    vecs = get_embeddings(df['text'].tolist(), api_key)
                    
                    # 2. Clustering (De 'emmertjes')
                    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0-cluster_threshold, metric='cosine', linkage='average')
                    df['Cluster_ID'] = clustering.fit_predict(vecs)
                    
                    # 3. Namen geven (Bulk)
                    c_to_n = {cid: df[df['Cluster_ID'] == cid]['text'].tolist() for cid in df['Cluster_ID'].unique()}
                    c_names = get_ai_cluster_names_bulk(c_to_n, api_key)
                    df['Category'] = df['Cluster_ID'].apply(lambda x: c_names.get(x, "ALGEMEEN"))
                    
                    # 4. Links berekenen (Numpy versnelling)
                    sims = cosine_similarity(vecs)
                    urls_arr = df[url_col].values
                    cat_map = dict(zip(df[url_col], df['Category']))

                    results = []
                    for f_url in focus_list:
                        if f_url not in df[url_col].values: continue
                        idx_src = df.index[df[url_col] == f_url].tolist()[0]
                        src_cat = df.iloc[idx_src]['Category']
                        
                        scores = sims[idx_src]
                        top_idx = np.argsort(scores)[::-1]
                        
                        added = 0
                        for t_idx in top_idx:
                            s = float(scores[t_idx])
                            if s < score_threshold: break # Stop als het niet meer relevant is
                            
                            t_url = urls_arr[t_idx]
                            if f_url != t_url:
                                results.append({
                                    'From Hub': src_cat,
                                    'From Folder': get_folder(f_url),
                                    'Focus URL': f_url,
                                    'To Hub': cat_map.get(t_url, "ALGEMEEN"),
                                    'To Folder': get_folder(t_url),
                                    'Target URL': t_url,
                                    'Score': s * 100
                                })
                                added += 1
                                if added >= links_per_page: break

                    st.session_state.df_results = pd.DataFrame(results)
                    st.rerun()
            except Exception as e:
                st.error(f"Fout: {e}")

# ========================================================
# 6. OUTPUT & MATRICES
# ========================================================
if st.session_state.df_results is not None:
    data = st.session_state.df_results
    st.divider()
    
    m_hub, m_folder = st.tabs(["🗂️ Semantische Hub Matrix", "📁 Technische Folder Matrix"])

    def style_mx(v, m):
        if v == 0: return 'background-color: #0a0a0a; color: #222222; text-align: center;'
        return f'background-color: rgba(0, 162, 255, {0.2 + 0.8*(v/m)}); color: #fff; font-weight: bold; text-align: center;'

    with m_hub:
        mx_h = pd.crosstab(data['From Hub'], data['To Hub'])
        mx_h = mx_h.reindex(index=mx_h.sum(axis=1).sort_values(ascending=False).index, columns=mx_h.sum(axis=0).sort_values(ascending=False).index, fill_value=0)
        st.dataframe(mx_h.style.map(lambda v: style_mx(v, mx_h.values.max() if mx_h.values.max()>0 else 1)), width='content', on_select="rerun", selection_mode="single-row", key="sel_h")
        
        sel_h = st.session_state.get("sel_h")
        if sel_h and sel_h.get("selection", {}).get("rows"):
            h_cat = mx_h.index[sel_h["selection"]["rows"][0]]
            st.dataframe(data[data['From Hub'] == h_cat][['Focus URL', 'To Hub', 'Target URL', 'Score']].style.map(color_score, subset=['Score']), width='content', hide_index=True)

    with m_folder:
        mx_f = pd.crosstab(data['From Folder'], data['To Folder'])
        mx_f = mx_f.reindex(index=mx_f.sum(axis=1).sort_values(ascending=False).index, columns=mx_f.sum(axis=0).sort_values(ascending=False).index, fill_value=0)
        st.dataframe(mx_f.style.map(lambda v: style_mx(v, mx_f.values.max() if mx_f.values.max()>0 else 1)), width='content', on_select="rerun", selection_mode="single-row", key="sel_f")
        
        sel_f = st.session_state.get("sel_f")
        if sel_f and sel_f.get("selection", {}).get("rows"):
            f_cat = mx_f.index[sel_f["selection"]["rows"][0]]
            st.dataframe(data[data['From Folder'] == f_cat][['Focus URL', 'To Folder', 'Target URL', 'Score']].style.map(color_score, subset=['Score']), width='content', hide_index=True)

    # 7. OVERZICHT & EXPORT
    st.divider()
    st.subheader("🏗️ Topic Hubs Detail")
    h_stats = data.groupby('From Hub')['Score'].mean().sort_values(ascending=False)
    t1, t2, t3 = st.tabs(["🟢 Sterk (>= 85%)", "🟡 Gemiddeld (70-84%)", "🔴 Zwak (< 70%)"])
    
    def render_h(stats):
        for h, a in stats.items():
            with st.expander(f"📁 {h} ({round(a)}%)"):
                st.dataframe(data[data['From Hub'] == h][['Focus URL', 'Target URL', 'Score']].style.map(color_score, subset=['Score']), width='content', hide_index=True)

    with t1: render_h(h_stats[h_stats >= 85])
    with t2: render_h(h_stats[(h_stats >= 70) & (h_stats < 85)])
    with t3: render_h(h_stats[h_stats < 70])

    csv = io.StringIO()
    data.to_csv(csv, index=False, sep=';')
    st.download_button("📥 Download Link Matrix", csv.getvalue(), "seo_matrix.csv", "text/csv")
