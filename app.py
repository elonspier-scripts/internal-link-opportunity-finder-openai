import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
from urllib.parse import urlparse

# 👉 Toegevoegd: voorkomt crash bij grote tabellen
pd.set_option("styler.render.max_elements", 1000000)

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

# 👉 Toegevoegd
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

tab_tool, tab_inst = st.tabs(["🚀 Analyse Tool", "📖 Instructies"])

with tab_inst:
    st.header("Hoe gebruik je deze tool?")
    st.markdown("""
    ### 1. Voorbereiding van het CSV-bestand
    Lever een CSV-bestand aan (bijv. een export uit Screaming Frog of een eigen lijst) met de volgende structuur:
    * **Kolom A (eerste kolom):** Moet de volledige URL's bevatten.
    * **Overige kolommen:** Hier mag content staan zoals de Page Title, H1 of de hoofdtekst. De tool gebruikt deze data om de context te begrijpen.
    
    ### 2. OpenAI API Key
    Voer je eigen OpenAI API Key in de sidebar aan de linkerkant in. De tool maakt gebruik van de `text-embedding-3-small` engine voor razendsnelle en goedkope analyses.

    ### 3. Focus URL's
    Plak in het tekstveld de URL's die je wilt analyseren. Dit zijn de pagina's waarvoor je interne linkmogelijkheden wilt vinden. Gebruik één URL per regel.

    ### 4. De Matrix gebruiken
    * Na de analyse verschijnt een **Cross-Linking Matrix**. 
    * De matrix is standaard gesorteerd op relevantie (de hubs met de meeste kansen staan bovenaan).
    * Klik op een **rij** in de matrix om direct alle specifieke link-kansen voor die hub te openen.
    """)

with tab_tool:
    c1, c2 = st.columns([1, 1])
    with c1:
        file = st.file_uploader("1. Upload Website CSV", type=['csv'], key="csv_uploader")
    with c2:
        urls_txt = st.text_area("2. Focus URL's (één per regel)", key="urls_input", height=100)

    # ========================================================
    # 5. DE ANALYSE ENGINE
    # ========================================================
    if st.button("🚀 GENEREER INTELLIGENCE MATRIX"):
        missing = []
        if not api_key: missing.append("OpenAI API Key (in de sidebar)")
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
                    cat_lookup = dict(zip(clean_df[url_col], clean_df['Category']))

                    vecs = get_embeddings(clean_df['text'].tolist(), api_key)
                    sims = cosine_similarity(vecs)

                    found = []
                    for f_url in focus_list:
                        if f_url not in clean_df[url_col].values: continue
                        idx_src = clean_df.index[clean_df[url_col] == f_url].tolist()[0]
                        src_cat = clean_df.iloc[idx_src]['Category']
                        
                        scores = sims[idx_src]
                        top_idx = np.argsort(scores)[::-1]
                        
                        added = 0
                        for t_idx in top_idx:
                            t_url = clean_df.iloc[t_idx][url_col]
                            s = float(scores[t_idx])
                            if f_url != t_url and s >= score_threshold:
                                found.append({
                                    'From Hub': src_cat,
                                    'From Folder': get_folder(f_url),
                                    'Focus URL': f_url,
                                    'To Hub': cat_lookup.get(t_url, "ALGEMEEN"),
                                    'To Folder': get_folder(t_url),
                                    'Target URL': t_url,
                                    'Score': s * 100
                                })
                                added += 1
                                if added >= links_per_page: break

                    st.session_state.df_results = pd.DataFrame(found)
                    st.rerun()

            except Exception as e:
                st.error(f"Systeemfout: {e}")

    # ========================================================
    # 6. MATRIX MET TABS (AANGEPAST)
    # ========================================================
    if st.session_state.df_results is not None:
        data = st.session_state.df_results
        st.divider()
        
        tab_matrix_hub, tab_matrix_folder = st.tabs(["🗂️ Woord-Cluster Matrix", "📁 Technische Folder Matrix"])

        def style_matrix_cells(val, mx_val):
            if val == 0:
                return 'background-color: #0a0a0a; color: #222222; text-align: center;'
            intensity = 0.2 + 0.8 * (val / mx_val)
            return f'background-color: rgba(0, 162, 255, {intensity}); color: #ffffff; font-weight: bold; text-align: center;'

        with tab_matrix_hub:
            matrix_hub = pd.crosstab(data['From Hub'], data['To Hub'])
            row_order = matrix_hub.sum(axis=1).sort_values(ascending=False).index
            col_order = matrix_hub.sum(axis=0).sort_values(ascending=False).index
            matrix_hub = matrix_hub.reindex(index=row_order, columns=col_order, fill_value=0)

            st.dataframe(
                matrix_hub.style.map(lambda v: style_matrix_cells(v, matrix_hub.values.max() if matrix_hub.values.max() > 0 else 1)),
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="matrix_selector"
            )

        with tab_matrix_folder:
            matrix_folder = pd.crosstab(data['From Folder'], data['To Folder'])
            row_order = matrix_folder.sum(axis=1).sort_values(ascending=False).index
            col_order = matrix_folder.sum(axis=0).sort_values(ascending=False).index
            matrix_folder = matrix_folder.reindex(index=row_order, columns=col_order, fill_value=0)

            st.dataframe(
                matrix_folder.style.map(lambda v: style_matrix_cells(v, matrix_folder.values.max() if matrix_folder.values.max() > 0 else 1)),
                width='content',
                on_select="rerun",
                selection_mode="single-row",
                key="mx_folder_tab"
            )
