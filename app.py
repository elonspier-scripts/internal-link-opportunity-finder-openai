import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
from urllib.parse import urlparse

# ========================================================
# 1. UI CONFIGURATIE (Deep Black / Cyber Blue)
# ========================================================
st.set_page_config(page_title="SEO Link Matrix Pro", layout="wide")

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
# 2. INITIALISATIE
# ========================================================
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

with st.sidebar:
    st.title("⚙️ Configuratie")
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key_val")
    st.divider()
    score_threshold = st.slider("Minimale Match %", 50, 95, 80) / 100
    links_per_page = st.slider("Aantal links per URL", 1, 10, 5)

# ========================================================
# 3. HELPERS
# ========================================================
def clean_path(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    return re.sub(r'[-_/]', ' ', path)

def get_folder(url):
    """Extraheert de eerste hoofdmap uit de URL"""
    try:
        path = urlparse(str(url)).path
        clean_p = path.strip('/')
        if not clean_p:
            return '/'
        return f"/{clean_p.split('/')[0]}/"
    except:
        return "/"

def get_embeddings(texts, key):
    client = OpenAI(api_key=key)
    res = client.embeddings.create(input=texts, model='text-embedding-3-small')
    return np.array([d.embedding for d in res.data])

def get_cat(text):
    words = re.findall(r'\w{4,}', str(text).lower())
    stop = {'deze', 'voor', 'naar', 'met', 'door', 'geen', 'over', 'mijn'}
    filtered = [w for w in words if w not in stop]
    unique = list(dict.fromkeys(filtered))
    return " / ".join(unique[:2]).upper() if unique else "ALGEMEEN"

def color_score(v):
    if not isinstance(v, (int, float)): return ''
    if v >= 85: return 'color: #28a745; font-weight: bold;'
    elif v >= 70: return 'color: #ffc107; font-weight: bold;'
    else: return 'color: #dc3545; font-weight: bold;'

# ========================================================
# 4. DASHBOARD TABS
# ========================================================
st.title("🔗 SEO Link Intelligence Matrix")

tab_tool, tab_folder, tab_inst = st.tabs(["🚀 Analyse Tool", "📂 Folder Matrix", "📖 Instructies"])

with tab_inst:
    st.header("Hoe gebruik je deze tool?")
    st.markdown("""
    ### 1. Voorbereiding van het CSV-bestand
    Lever een CSV-bestand aan met de URL's in de eerste kolom.
    
    ### 2. OpenAI API Key
    Voer je API Key in de sidebar in.

    ### 3. Focus URL's
    Plak de URL's die je wilt analyseren in het tekstveld.

    ### 4. Folder Matrix
    De nieuwe **Folder Matrix** tab toont linkmogelijkheden op basis van de URL-structuur (bijv. van `/blog/` naar `/producten/`).
    """)

with tab_tool:
    c1, c2 = st.columns([1, 1])
    with c1:
        file = st.file_uploader("1. Upload Website CSV", type=['csv'], key="csv_uploader")
    with c2:
        urls_txt = st.text_area("2. Focus URL's (één per regel)", key="urls_input", height=100)

    if st.button("🚀 GENEREER INTELLIGENCE MATRIX", width="stretch"):
        missing = []
        if not api_key: missing.append("OpenAI API Key")
        if not file: missing.append("CSV-bestand")
        if not urls_txt: missing.append("Focus URL's")

        if missing:
            st.error(f"⚠️ De volgende velden ontbreken: {', '.join(missing)}")
        else:
            try:
                with st.spinner("Bezig met semantische analyse..."):
                    raw_df = pd.read_csv(file)
                    url_col = raw_df.columns[0]
                    focus_list = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                    
                    clean_df = raw_df.dropna(subset=[url_col]).copy()
                    clean_df = clean_df.fillna("")
                    clean_df = clean_df[clean_df[url_col].astype(str).str.strip() != ""]
                    
                    clean_df['text'] = clean_df[url_col].astype(str).apply(clean_path) + " " + clean_df.iloc[:, 1].astype(str)
                    clean_df['Category'] = clean_df['text'].apply(get_cat)
                    clean_df['Folder'] = clean_df[url_col].apply(get_folder)
                    
                    cat_lookup = dict(zip(clean_df[url_col], clean_df['Category']))
                    folder_lookup = dict(zip(clean_df[url_col], clean_df['Folder']))

                    vecs = get_embeddings(clean_df['text'].tolist(), api_key)
                    sims = cosine_similarity(vecs)

                    found = []
                    for f_url in focus_list:
                        if f_url not in clean_df[url_col].values: continue
                        idx_src = clean_df.index[clean_df[url_col] == f_url].tolist()[0]
                        src_cat = clean_df.iloc[idx_src]['Category']
                        src_folder = clean_df.iloc[idx_src]['Folder']
                        
                        scores = sims[idx_src]
                        top_idx = np.argsort(scores)[::-1]
                        
                        added = 0
                        for t_idx in top_idx:
                            t_url = clean_df.iloc[t_idx][url_col]
                            s = float(scores[t_idx])
                            if f_url != t_url and s >= score_threshold:
                                found.append({
                                    'From Hub': src_cat,
                                    'From Folder': src_folder,
                                    'Focus URL': f_url,
                                    'To Hub': cat_lookup.get(t_url, "ALGEMEEN"),
                                    'To Folder': folder_lookup.get(t_url, "/"),
                                    'Target URL': t_url,
                                    'Score': s * 100
                                })
                                added += 1
                                if added >= links_per_page: break

                    st.session_state.df_results = pd.DataFrame(found)
                    st.rerun()

            except Exception as e:
                st.error(f"Systeemfout: {e}")
                
    # Hub Matrix (Originele weergave onder de knop)
    if st.session_state.df_results is not None:
        data = st.session_state.df_results
        matrix_hubs = pd.crosstab(data['From Hub'], data['To Hub'])
        
        st.divider()
        st.subheader("📊 Cross-Linking Matrix (Topic Hubs)")
        
        max_val = matrix_hubs.values.max() if matrix_hubs.values.max() > 0 else 1
        def style_matrix(val):
            if val == 0: return 'background-color: #0a0a0a; color: #222222; text-align: center;'
            intensity = 0.2 + 0.8 * (val / max_val)
            return f'background-color: rgba(0, 162, 255, {intensity}); color: #ffffff; font-weight: bold; text-align: center;'

        st.dataframe(matrix_hubs.style.map(style_matrix), width='stretch', on_select="rerun", selection_mode="single-row", key="hub_selector")

        sel_hub = st.session_state.get("hub_selector")
        if sel_hub and sel_hub.get("selection", {}).get("rows"):
            f_cat = matrix_hubs.index[sel_hub["selection"]["rows"][0]]
            st.markdown(f"### 🎯 Hub Details: `{f_cat}`")
            f_data = data[data['From Hub'] == f_cat][['Focus URL', 'To Hub', 'Target URL', 'Score']]
            st.dataframe(f_data.style.map(color_score, subset=['Score']), width='stretch', hide_index=True)

# ========================================================
# 5. FOLDER MATRIX TAB
# ========================================================
with tab_folder:
    if st.session_state.df_results is not None:
        data = st.session_state.df_results
        matrix_folder = pd.crosstab(data['From Folder'], data['To Folder'])
        
        st.subheader("📂 URL Path / Folder Matrix")
        st.info("Deze matrix toont linkmogelijkheden tussen verschillende directory-niveaus.")

        max_f_val = matrix_folder.values.max() if matrix_folder.values.max() > 0 else 1
        def style_f_matrix(val):
            if val == 0: return 'background-color: #0a0a0a; color: #222222; text-align: center;'
            intensity = 0.2 + 0.8 * (val / max_f_val)
            return f'background-color: rgba(0, 255, 162, {intensity}); color: #ffffff; font-weight: bold; text-align: center;'

        st.dataframe(
            matrix_folder.style.map(style_f_matrix), 
            width='stretch', 
            on_select="rerun", 
            selection_mode="single-row", 
            key="folder_selector"
        )

        sel_folder = st.session_state.get("folder_selector")
        if sel_folder and sel_folder.get("selection", {}).get("rows"):
            f_path = matrix_folder.index[sel_folder["selection"]["rows"][0]]
            st.markdown(f"### 🎯 Folder Details: `{f_path}`")
            f_data = data[data['From Folder'] == f_path][['Focus URL', 'To Folder', 'Target URL', 'Score']]
            st.dataframe(f_data.style.map(color_score, subset=['Score']), width='stretch', hide_index=True)
    else:
        st.warning("Voer eerst een analyse uit in de 'Analyse Tool' tab.")

# ========================================================
# 6. EXPORT & OVERZICHT (Onderaan)
# ========================================================
if st.session_state.df_results is not None:
    st.divider()
    csv_buffer = io.StringIO()
    st.session_state.df_results.to_csv(csv_buffer, index=False, sep=';')
    st.download_button("📥 Download Volledige Matrix (CSV)", data=csv_buffer.getvalue(), file_name="seo_folder_matrix.csv", mime="text/csv", width="stretch")
