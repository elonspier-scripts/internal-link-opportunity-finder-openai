import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import io

# ========================================================
# 1. UI CONFIGURATIE (Zwart / Blauw / App-stijl)
# ========================================================
st.set_page_config(page_title="Internal Link Matrix Pro", layout="wide")

st.markdown("""
    <style>
    /* Achtergrond & Tekst */
    .stApp { background-color: #000000; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1e1e1e; }
    
    /* Input velden styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #0a0a0a !important; color: #00a2ff !important; border: 1px solid #222 !important;
    }

    /* Tabel en Matrix styling */
    .stDataFrame, div[data-testid="stTable"] { 
        background-color: #000 !important; border: 1px solid #1e1e1e !important; border-radius: 4px;
    }
    
    /* Headers */
    h1, h2, h3 { color: #00a2ff !important; font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }
    
    /* Grote Actie Knop */
    .stButton>button { 
        background: linear-gradient(135deg, #0062ff 0%, #00a2ff 100%);
        color: white; border: none; padding: 12px; font-weight: bold; width: 100%;
        box-shadow: 0 4px 15px rgba(0, 162, 255, 0.3);
    }
    
    /* Verstop branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ========================================================
# 2. INITIALISATIE & SIDEBAR
# ========================================================
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1243/1243933.png", width=50) # Optioneel icoontje
    st.header("App Config")
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key_input")
    st.divider()
    score_threshold = st.slider("Minimale Match %", 50, 95, 80) / 100
    links_per_page = st.slider("Aantal links per URL", 1, 10, 5)

# ========================================================
# 3. HELPERS
# ========================================================
def clean_path(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    return re.sub(r'[-_/]', ' ', path)

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

# ========================================================
# 4. DASHBOARD INPUT
# ========================================================
st.title("SEO Link Intelligence Matrix")

c1, c2 = st.columns([1, 1])
with c1:
    file = st.file_uploader("1. Upload Website CSV (URL kolom = A)", type=['csv'], key="csv_file")
with c2:
    urls_txt = st.text_area("2. Focus URL's (één per regel)", key="focus_area")

# ========================================================
# 5. ENGINE
# ========================================================
if st.button("🚀 GENEREER INTELLIGENCE MATRIX"):
    if not (api_key and file and urls_txt):
        st.error("⚠️ Actie vereist: Vul de API Key in, upload een bestand en voer URL's in.")
    else:
        try:
            with st.spinner("Analyseert semantische structuren..."):
                raw_df = pd.read_csv(file)
                url_col = raw_df.columns[0]
                focus_list = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                
                # Pre-processing
                clean_df = raw_df[raw_df[url_col].str.strip() != ""].copy()
                clean_df['text'] = clean_df[url_col].apply(clean_path) + " " + clean_df.iloc[:, 1].astype(str)
                clean_df['Category'] = clean_df['text'].apply(get_cat)
                cat_lookup = dict(zip(clean_df[url_col], clean_df['Category']))

                # Vectors
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
                                'Focus URL': f_url,
                                'To Hub': cat_lookup.get(t_url, "ALGEMEEN"),
                                'Target URL': t_url,
                                'Score': round(s * 100)
                            })
                            added += 1
                            if added >= links_per_page: break

                st.session_state.df_results = pd.DataFrame(found)
                st.rerun()

        except Exception as e:
            st.error(f"Systeemfout: {e}")

# ========================================================
# 6. INTERACTIEVE OUTPUT (DE MATRIX)
# ========================================================
if st.session_state.df_results is not None:
    data = st.session_state.df_results
    
    # Maak de Matrix (Target Hubs = Kolommen = Bovenaan)
    matrix = pd.crosstab(data['From Hub'], data['To Hub'])
    all_hubs = sorted(list(set(data['From Hub']).union(set(data['To Hub']))))
    matrix = matrix.reindex(index=all_hubs, columns=all_hubs, fill_value=0)

    st.divider()
    st.subheader("📊 Cross-Linking Matrix")
    st.info("💡 KLIKBAAR: Selecteer een **rij** in de tabel hieronder om alle uitgaande links van die specifieke hub te bekijken.")

    # Matrix tonen met single-row selectie
    selection = st.dataframe(
        matrix.style.background_gradient(cmap='Blues', axis=None),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="matrix_selector"
    )

    # Wanneer er een rij wordt geselecteerd
    if selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        f_cat = matrix.index[selected_idx]
        
        st.markdown(f"### 🎯 Uitgaande links vanuit: `{f_cat}`")
        
        # Filter op de geselecteerde rij (From Hub)
        filtered = data[data['From Hub'] == f_cat]
        
        if not filtered.empty:
            def color_score(v):
                c = '#28a745' if v >= 85 else '#ffc107' if v >= 70 else '#dc3545'
                return f'color: {c}; font-weight: bold'

            # Tabel met focus URLs, waar ze heen gaan en de target URL
            st.dataframe(
                filtered[['Focus URL', 'To Hub', 'Target URL', 'Score']].sort_values('Score', ascending=False).style.map(color_score, subset=['Score']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Geen uitgaande links gevonden voor deze hub in de huidige selectie.")
    else:
        # Als er niets is aangeklikt, laat dan de Top-down lijst zien (standaard weergave)
        st.divider()
        st.subheader("🏗️ Alle Topic Hubs (Top-Down Overzicht)")
        for hub in sorted(data['From Hub'].unique()):
            with st.expander(f"📁 HUB: {hub}"):
                hub_df = data[data['From Hub'] == hub][['Focus URL', 'To Hub', 'Target URL', 'Score']]
                
                def color_score_fallback(v):
                    c = '#28a745' if v >= 85 else '#ffc107' if v >= 70 else '#dc3545'
                    return f'color: {c}; font-weight: bold'
                
                st.dataframe(
                    hub_df.sort_values('Score', ascending=False).style.map(color_score_fallback, subset=['Score']), 
                    use_container_width=True, 
                    hide_index=True
                )
