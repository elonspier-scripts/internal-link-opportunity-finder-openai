import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import io

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
    # We gebruiken een key om de API-sleutel te bewaren
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
    """Bepaalt de kleur van de percentage-scores in de tabellen"""
    if not isinstance(v, (int, float)): return ''
    if v >= 85: return 'color: #28a745; font-weight: bold;'
    elif v >= 70: return 'color: #ffc107; font-weight: bold;'
    else: return 'color: #dc3545; font-weight: bold;'

# ========================================================
# 4. DASHBOARD INPUT
# ========================================================
st.title("🔗 SEO Link Intelligence Matrix")

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
                
                clean_df = raw_df[raw_df[url_col].str.strip() != ""].copy()
                clean_df['text'] = clean_df[url_col].apply(clean_path) + " " + clean_df.iloc[:, 1].astype(str)
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
                                'Focus URL': f_url,
                                'To Hub': cat_lookup.get(t_url, "ALGEMEEN"),
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
# 6. INTERACTIEVE MATRIX & OUTPUT
# ========================================================
if st.session_state.df_results is not None:
    data = st.session_state.df_results
    
    # Matrix bouwen
    matrix = pd.crosstab(data['From Hub'], data['To Hub'])
    all_hubs = sorted(list(set(data['From Hub']).union(set(data['To Hub']))))
    matrix = matrix.reindex(index=all_hubs, columns=all_hubs, fill_value=0)

    st.divider()
    st.subheader("📊 Cross-Linking Matrix (Intensity)")
    st.info("💡 Klik op een rij (de verwijzende Hub) om de link-kansen te bekijken.")

    # Custom Matrix Styling (Zonder matplotlib, perfect voor de App Dark Mode)
    max_val = matrix.values.max() if matrix.values.max() > 0 else 1
    def style_matrix_cells(val):
        if val == 0:
            return 'background-color: #0a0a0a; color: #222222; text-align: center;'
        else:
            intensity = 0.2 + 0.8 * (val / max_val)
            return f'background-color: rgba(0, 162, 255, {intensity}); color: #ffffff; font-weight: bold; text-align: center;'

    styled_matrix = matrix.style.map(style_matrix_cells)

    # Matrix weergeven
    st.dataframe(
        styled_matrix,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="matrix_selector"
    )

    # Details tonen na klik op matrix
    selection = st.session_state.get("matrix_selector")
    if selection and selection.get("selection", {}).get("rows"):
        selected_idx = selection["selection"]["rows"][0]
        f_cat = matrix.index[selected_idx]
        
        st.markdown(f"### 🎯 Uitgaande links vanuit: `{f_cat}`")
        filtered = data[data['From Hub'] == f_cat]
        
        # Sorteer op Focus URL en maak duplicates leeg
        display_filtered = filtered[['Focus URL', 'To Hub', 'Target URL', 'Score']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
        display_filtered.loc[display_filtered.duplicated('Focus URL'), 'Focus URL'] = ""
        
        # Tabel weergeven met de Score-kleuren toegepast
        st.dataframe(
            display_filtered.style.map(color_score, subset=['Score']),
            use_container_width=True,
            hide_index=True,
            column_config={"Score": st.column_config.NumberColumn(format="%d%%")}
        )

    # ========================================================
    # 7. TOPIC HUBS OVERZICHT
    # ========================================================
    st.divider()
    st.subheader("🏗️ Topic Hubs Overzicht")
    
    hubs = sorted(data['From Hub'].unique())
    for hub in hubs:
        hub_df = data[data['From Hub'] == hub]
        avg_score = round(hub_df['Score'].mean())
        
        # Score in de titel
        with st.expander(f"📁 HUB: {hub} ({avg_score}%)"):
            
            # Sorteer en verberg duplicates
            display_hub = hub_df[['Focus URL', 'To Hub', 'Target URL', 'Score']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
            display_hub.loc[display_hub.duplicated('Focus URL'), 'Focus URL'] = ""
            
            # Tabel weergeven met de Score-kleuren toegepast
            st.dataframe(
                display_hub.style.map(color_score, subset=['Score']),
                use_container_width=True,
                hide_index=True,
                column_config={"Score": st.column_config.NumberColumn(format="%d%%")}
            )

    # ========================================================
    # 8. EXPORT EXCEL
    # ========================================================
    st.divider()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Link_Kansen')
        matrix.to_excel(writer, sheet_name='Matrix_Overzicht')
        
    st.download_button(
        label="📥 Download Volledige Analyse (Excel)",
        data=output.getvalue(),
        file_name="seo_internal_links_matrix.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
