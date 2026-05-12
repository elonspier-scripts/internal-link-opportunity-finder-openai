import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
import re
import io
import hashlib
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ========================================================
# 1. UI CONFIGURATION (Deep Black / Cyber Blue)
# ========================================================
st.set_page_config(page_title="SEO Link Opportunity Finder", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1e1e1e; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
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
# 2. INITIALISATION
# ========================================================
if 'df_results' not in st.session_state:
    st.session_state.df_results = None
if 'analysis_signature' not in st.session_state:
    st.session_state.analysis_signature = None

with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key_val")
    st.divider()
    score_threshold = st.slider("Minimum Match threshold % (display filter)", 50, 95, 80) / 100
    links_per_page = st.slider("Links per URL", 1, 10, 5)
    check_existing_links = st.toggle(
        "Check and hide existing links",
        value=False,
        help="Checks if recommended links already exist on the page (boilerplate excluded) and hides matches that already exist."
    )

# ========================================================
# 3. HELPERS
# ========================================================
def clean_path(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    return re.sub(r'[-_/]', ' ', path)

def get_folder(url):
    """Extraheert de eerste hoofdmap (top-level folder) uit een URL"""
    try:
        path = urlparse(str(url)).path
        clean_p = path.strip('/')
        if not clean_p:
            return '/'
        first_folder = clean_p.split('/')[0]
        return f"/{first_folder}/"
    except:
        return "/"

def parse_sitemap_xml(xml_text):
    urls = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return urls, []

    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    url_locs = root.findall('.//sm:url/sm:loc', ns)
    if not url_locs:
        url_locs = root.findall('.//url/loc')
    for loc in url_locs:
        if loc.text and loc.text.strip():
            urls.append(loc.text.strip())

    child_sitemaps = []
    sm_locs = root.findall('.//sm:sitemap/sm:loc', ns)
    if not sm_locs:
        sm_locs = root.findall('.//sitemap/loc')
    for loc in sm_locs:
        if loc.text and loc.text.strip():
            child_sitemaps.append(loc.text.strip())

    return urls, child_sitemaps

@st.cache_data(show_spinner=False)
def fetch_sitemap_urls(sitemap_url, max_sitemaps=25):
    pending = [sitemap_url]
    visited = set()
    collected_urls = []

    while pending and len(visited) < max_sitemaps:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)

        response = requests.get(current, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        urls, child_sitemaps = parse_sitemap_xml(response.text)
        if urls:
            collected_urls.extend(urls)
        for child in child_sitemaps:
            if child not in visited:
                pending.append(child)

    deduped = list(dict.fromkeys([u for u in collected_urls if str(u).strip()]))
    return deduped

def normalize_compare_url(url):
    if not url:
        return ""
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip('/')
    return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"

def extract_main_content_links(page_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(page_url, timeout=15, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    links = set()

    def has_boilerplate_class_or_id(anchor):
        class_pattern = re.compile(r'\b(footer|menu|sidebar|breadcrumb|breadcrumbs)\b', re.I)
        for parent in anchor.parents:
            if not getattr(parent, 'name', None):
                continue
            if parent.name in ('html', 'body'):
                continue

            classes = parent.get('class') or []
            class_text = " ".join(classes) if isinstance(classes, list) else str(classes)
            if class_pattern.search(class_text):
                return True

            parent_id = parent.get('id') or ""
            if class_pattern.search(str(parent_id)):
                return True

        return False

    for anchor in soup.find_all('a', href=True):
        if anchor.find_parent(['header', 'nav', 'footer', 'aside']):
            continue
        if anchor.find_parent(attrs={'role': re.compile(r'navigation', re.I)}):
            continue
        if has_boilerplate_class_or_id(anchor):
            continue

        href = (anchor.get('href') or '').strip()
        if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            continue
        absolute = requests.compat.urljoin(page_url, href)
        normalized = normalize_compare_url(absolute)
        if normalized:
            links.add(normalized)

    return links

@st.cache_data(show_spinner=False)
def get_cached_page_links(page_url):
    return sorted(extract_main_content_links(page_url))

@st.cache_data(show_spinner=False)
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def get_embeddings(texts, key, model_name, file_hash):
    """Fetches embeddings from OpenAI, cached to prevent duplicate API costs."""
    client = OpenAI(api_key=key)
    res = client.embeddings.create(input=list(texts), model=model_name)
    return np.array([d.embedding for d in res.data])

@st.cache_data(show_spinner=False)
def get_top_k_neighbors(vectors, file_hash, model_name, internal_k):
    n_rows = len(vectors)
    n_neighbors = min(max(2, internal_k + 1), n_rows)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    similarities = 1.0 - distances
    return indices, similarities

def get_cat(text):
    words = re.findall(r'\w{4,}', str(text).lower())
    stop = {'deze', 'voor', 'naar', 'met', 'door', 'geen', 'over', 'mijn'}
    filtered = [w for w in words if w not in stop]
    unique = list(dict.fromkeys(filtered))
    return " / ".join(unique[:2]).upper() if unique else "ALGEMEEN"

def color_score(v):
    # Strip emoji label for calculation if necessary, but keep CSS simple
    val = v
    if isinstance(v, str):
        try: val = float(v.replace('%', '').replace('⚠️', '').strip())
        except: return ''
    
    if val >= 85: return 'color: #28a745; font-weight: bold;'
    elif val >= 70: return 'color: #ffc107; font-weight: bold;'
    else: return 'color: #dc3545; font-weight: bold;'

# ========================================================
# 4. DASHBOARD TABS
# ========================================================
st.title("🔗 SEO Link Opportunity Finder")

tab_tool, tab_inst = st.tabs(["🚀 internal link Tool", "📖 Instructions"])

with tab_inst:
    st.header("How to use this tool")
    st.markdown("""
    ### 1. Preparing the CSV file
    Provide a CSV file (e.g., an export from Screaming Frog or your own list) with the following structure:
    * **Column A (first column):** Must contain the full URLs.
    * **Other columns:** May contain content such as the Page Title, H1, or the main body text.

    ### 2. Technical Limitations & Advice
    * **Max File Size:** Keep your CSV under **20MB**. Larger files may crash the app memory.
    * **Optimal Rows:** For the best performance, aim for **1,000 to 5,000 rows**. 
    * **Limits:** Analyzing more than **10,000 rows** is not advised as it significantly increases processing time and API costs.
    * **Data Cleaning:** For large sites, only upload "Internal HTML" URLs with a 200 status code.
    
    ### 3. OpenAI API Key
    Enter your own OpenAI API Key in the sidebar.

    ### 4. Focus URLs
    Paste the URLs you want to analyze into the text field. Use one URL per line.(The focus URL should be in the .csv you have uploaded aswell).

    ### 5. Using the Matrix (Bi-Directional)
    * After the analysis, a **Cross-Linking Matrix** will appear. 
    * Use the dropdown above each matrix to switch between **Outbound** (where your focus URL should link to) and **Inbound** (which pages should link to your focus URL).
    * Click on a **row** in the matrix to immediately open all specific linking opportunities.
    * The table will explicitly tell you which page you need to edit in your CMS.

    ### 6. Using the Topic Hub Overview
    * Click on a **Hub** in the Topic Hub Overview to open all specific linking opportunities.
    * Prioritize internal linking opportunities based on the relevance scores.
    
    ### 7. Export results to CSV
    * Use the **download button** at the bottom to export all Topic Hub results at once.
    * You can also download each individual table separately by hovering over it and clicking the download icon.
    """)

with tab_tool:
    c1, c2 = st.columns([1, 1])
    with c1:
        file = st.file_uploader("1. Upload Website CSV", type=['csv'], key="csv_uploader")
        sitemap_url = st.text_input("or Sitemap URL", placeholder="https://example.com/sitemap.xml", key="sitemap_input")
    with c2:
        urls_txt = st.text_area("2. Focus URL's (1 per row)", key="urls_input", height=100)

    # ========================================================
    # 5. THE ANALYSIS
    # ========================================================
    if st.button("🚀 Generate", width="stretch"):
        missing = []
        if not api_key: missing.append("OpenAI API Key (in the sidebar)")
        using_sitemap_input = bool(sitemap_url.strip())
        if not file and not using_sitemap_input: missing.append("CSV-file or Sitemap URL")
        if not using_sitemap_input and not urls_txt: missing.append("Focus URL's")

        if missing:
            st.error(f"⚠️ De volgende velden ontbreken: {', '.join(missing)}")
        else:
            try:
                with st.spinner("Analysing..."):
                    using_sitemap = bool(sitemap_url.strip())

                    if using_sitemap:
                        sitemap_urls = fetch_sitemap_urls(sitemap_url.strip())
                        if not sitemap_urls:
                            st.error("No URLs found in sitemap.")
                            st.stop()
                        raw_df = pd.DataFrame({"URL": sitemap_urls})
                        url_col = "URL"
                        file_hash = get_file_hash("\n".join(sitemap_urls).encode("utf-8"))
                    else:
                        file_bytes = file.getvalue()
                        file_hash = get_file_hash(file_bytes)
                        raw_df = pd.read_csv(io.BytesIO(file_bytes))
                        url_col = raw_df.columns[0]

                    focus_list = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                    embedding_model = 'text-embedding-3-small'
                    
                    clean_df = raw_df.dropna(subset=[url_col]).copy()
                    clean_df = clean_df.fillna("")
                    clean_df = clean_df[clean_df[url_col].astype(str).str.strip() != ""]
                    clean_df = clean_df.reset_index(drop=True)
                    
                    content_cols = [c for c in clean_df.columns if c != url_col]
                    clean_df['url_text'] = clean_df[url_col].astype(str).apply(clean_path)

                    if using_sitemap:
                        clean_df['text'] = clean_df['url_text'].str.replace(r"\s+", " ", regex=True).str.strip()
                    elif content_cols:
                        content_text = clean_df[content_cols].astype(str).agg(" ".join, axis=1)
                        clean_df['text'] = (clean_df['url_text'] + " " + content_text).str.replace(r"\s+", " ", regex=True).str.strip()
                    else:
                        clean_df['text'] = clean_df['url_text'].str.replace(r"\s+", " ", regex=True).str.strip()
                    clean_df['Category'] = clean_df['text'].apply(get_cat)
                    cat_lookup = dict(zip(clean_df[url_col], clean_df['Category']))

                    if using_sitemap and not focus_list:
                        focus_list = clean_df[url_col].astype(str).tolist()

                    analysis_signature = (
                        file_hash,
                        tuple(focus_list),
                        links_per_page,
                        check_existing_links,
                        embedding_model,
                        20
                    )

                    if st.session_state.df_results is not None and st.session_state.analysis_signature == analysis_signature:
                        st.info("Using cached analysis results for this dataset/settings.")
                        st.rerun()

                    vecs = get_embeddings(tuple(clean_df['text'].tolist()), api_key, embedding_model, file_hash)
                    internal_k = 20
                    neighbor_indices, neighbor_scores = get_top_k_neighbors(vecs, file_hash, embedding_model, internal_k)

                    found = []
                    base_score_threshold = 0.50
                    for f_url in focus_list:
                        if f_url not in clean_df[url_col].values: continue
                        idx_src = clean_df.index[clean_df[url_col] == f_url].tolist()[0]
                        src_cat = clean_df.iloc[idx_src]['Category']

                        added = 0
                        for t_idx, s in zip(neighbor_indices[idx_src], neighbor_scores[idx_src]):
                            t_url = clean_df.iloc[t_idx][url_col]
                            s = float(s)
                            if f_url != t_url and s >= base_score_threshold:
                                
                                # 1. OUTBOUND
                                found.append({
                                    'Direction': 'Outbound',
                                    'From Hub': src_cat,
                                    'From Folder': get_folder(f_url),
                                    'Focus URL': f_url,
                                    'To Hub': cat_lookup.get(t_url, "General"),
                                    'To Folder': get_folder(t_url),
                                    'Target URL': t_url,
                                    'Page to Edit (Source)': f_url,
                                    'Link Destination': t_url,
                                    'Score': s * 100
                                })
                                
                                # 2. INBOUND
                                found.append({
                                    'Direction': 'Inbound',
                                    'From Hub': cat_lookup.get(t_url, "General"),
                                    'From Folder': get_folder(t_url),
                                    'Focus URL': f_url,
                                    'To Hub': src_cat,
                                    'To Folder': get_folder(f_url),
                                    'Target URL': f_url,
                                    'Page to Edit (Source)': t_url,
                                    'Link Destination': f_url,
                                    'Score': s * 100
                                })
                                
                                added += 1
                                if added >= links_per_page: break

                    if check_existing_links and found:
                        source_pages = list(dict.fromkeys([row['Page to Edit (Source)'] for row in found if row.get('Page to Edit (Source)')]))
                        existing_links_map = {}

                        with st.status("Checking existing links on live pages...", expanded=False):
                            for source_url in source_pages:
                                try:
                                    existing_links_map[source_url] = set(get_cached_page_links(source_url))
                                except Exception:
                                    existing_links_map[source_url] = None

                        for row in found:
                            source_url = row.get('Page to Edit (Source)', '')
                            target_url = normalize_compare_url(row.get('Link Destination', ''))
                            link_set = existing_links_map.get(source_url)
                            if link_set is None:
                                row['Existing Link'] = "Unknown"
                            elif target_url and target_url in link_set:
                                row['Existing Link'] = "Yes"
                            else:
                                row['Existing Link'] = "No"
                    else:
                        for row in found:
                            row['Existing Link'] = "Not checked"

                    st.session_state.df_results = pd.DataFrame(found)
                    st.session_state.analysis_signature = analysis_signature
                    st.rerun()

            except Exception as e:
                st.error(f"Systeemfout: {e}")
                
    # ========================================================
    # 6. INTERACTIVE MATRIX & OUTPUT
    # ========================================================
    if st.session_state.df_results is not None:
        data = st.session_state.df_results.copy()

        if 'Existing Link' not in data.columns:
            data['Existing Link'] = "Not checked"

        data = data[data['Score'] >= (score_threshold * 100)].copy()
        if data.empty:
            st.info("No opportunities match the selected minimum score filter.")
            st.stop()

        hide_existing_links_display = st.toggle(
            "Hide existing links in results",
            value=check_existing_links,
            help="Filter out rows where the destination link already exists on the source page.",
            key="hide_existing_links_display"
        )

        if hide_existing_links_display:
            data = data[data['Existing Link'] != 'Yes'].copy()
            if data.empty:
                st.info("No opportunities left after hiding existing links.")
                st.stop()
        
        st.divider()
        st.subheader("📊 Cross-Linking Matrix")
        st.info("💡 Click on a row for more details. The matrix is in descending order with the most link opportunities first. Internal links with an similarity score of 95% (⚠️) or higher. The pages might compete with each other.")

        tab_matrix_hub, tab_matrix_folder = st.tabs(["🗂️ Semantic Hub Matrix", "📁 Path / Folder Matrix"])

        def style_matrix_cells(val, mx_val):
            if val == 0:
                return 'background-color: #0a0a0a; color: #222222; text-align: center;'
            else:
                intensity = 0.2 + 0.8 * (val / mx_val)
                return f'background-color: rgba(0, 162, 255, {intensity}); color: #ffffff; font-weight: bold; text-align: center;'

        # --- TAB 1: HUB MATRIX ---
        with tab_matrix_hub:
            dir_hub = st.selectbox("🔗 Select Link Direction:", ["Outbound", "Inbound"], key="dir_hub_select")
            data_hub = data[data['Direction'] == dir_hub]
            
            if not data_hub.empty:
                matrix_hub = pd.crosstab(data_hub['From Hub'], data_hub['To Hub'])
                row_order_hub = matrix_hub.sum(axis=1).sort_values(ascending=False).index
                col_order_hub = matrix_hub.sum(axis=0).sort_values(ascending=False).index
                matrix_hub = matrix_hub.reindex(index=row_order_hub, columns=col_order_hub, fill_value=0)

                max_val_hub = matrix_hub.values.max() if matrix_hub.values.max() > 0 else 1
                styled_matrix_hub = matrix_hub.style.map(lambda v: style_matrix_cells(v, max_val_hub))

                st.dataframe(styled_matrix_hub, width='stretch', on_select="rerun", selection_mode="multi-row", key="matrix_selector_hub")

                select_all_hubs = st.checkbox("Select all hub rows", value=False, key=f"select_all_hubs_{dir_hub.lower()}")

                hub_csv_buffer = io.StringIO()
                matrix_hub.to_csv(hub_csv_buffer, sep=';')
                st.download_button(
                    label=f"📥 Download Semantic Hub Matrix ({dir_hub})",
                    data=hub_csv_buffer.getvalue(),
                    file_name=f"semantic_hub_matrix_{dir_hub.lower()}.csv",
                    mime="text/csv",
                    key=f"download_hub_matrix_{dir_hub.lower()}"
                )

                selected_hubs = []
                if select_all_hubs:
                    selected_hubs = list(matrix_hub.index)
                else:
                    selection_hub = st.session_state.get("matrix_selector_hub")
                    if selection_hub and selection_hub.get("selection", {}).get("rows"):
                        selected_hubs = [matrix_hub.index[i] for i in selection_hub["selection"]["rows"] if i < len(matrix_hub.index)]

                if selected_hubs:
                    st.markdown(f"### 🎯 Links to place from Hub rows: `{len(selected_hubs)}` selected")
                    filtered = data_hub[data_hub['From Hub'].isin(selected_hubs)].copy()
                    display_filtered = filtered[['Focus URL', 'Page to Edit (Source)', 'To Hub', 'Link Destination', 'Score', 'Existing Link']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
                    display_filtered.loc[display_filtered.duplicated('Focus URL'), 'Focus URL'] = ""
                    
                    # Formatting logic for Warning Label
                    final_display = display_filtered[['Page to Edit (Source)', 'To Hub', 'Link Destination', 'Score', 'Existing Link']].copy()
                    final_display['Score'] = final_display['Score'].apply(lambda x: f"{int(x)}% ⚠️" if x >= 95 else f"{int(x)}%")
                    
                    st.dataframe(
                        final_display.style.map(color_score, subset=['Score']),
                        width='stretch',
                        hide_index=True
                    )
            else:
                st.warning(f"No {dir_hub.lower()} links found.")

        # --- TAB 2: FOLDER MATRIX ---
        with tab_matrix_folder:
            dir_folder = st.selectbox("🔗 Select Link Direction:", ["Outbound", "Inbound"], key="dir_folder_select")
            data_folder = data[data['Direction'] == dir_folder]
            
            if not data_folder.empty:
                matrix_folder = pd.crosstab(data_folder['From Folder'], data_folder['To Folder'])
                row_order_folder = matrix_folder.sum(axis=1).sort_values(ascending=False).index
                col_order_folder = matrix_folder.sum(axis=0).sort_values(ascending=False).index
                matrix_folder = matrix_folder.reindex(index=row_order_folder, columns=col_order_folder, fill_value=0)

                max_val_folder = matrix_folder.values.max() if matrix_folder.values.max() > 0 else 1
                styled_matrix_folder = matrix_folder.style.map(lambda v: style_matrix_cells(v, max_val_folder))

                st.dataframe(styled_matrix_folder, width='stretch', on_select="rerun", selection_mode="multi-row", key="matrix_selector_folder")

                select_all_folders = st.checkbox("Select all folder rows", value=False, key=f"select_all_folders_{dir_folder.lower()}")

                folder_csv_buffer = io.StringIO()
                matrix_folder.to_csv(folder_csv_buffer, sep=';')
                st.download_button(
                    label=f"📥 Download Folder Matrix ({dir_folder})",
                    data=folder_csv_buffer.getvalue(),
                    file_name=f"folder_matrix_{dir_folder.lower()}.csv",
                    mime="text/csv",
                    key=f"download_folder_matrix_{dir_folder.lower()}"
                )

                selected_folders = []
                if select_all_folders:
                    selected_folders = list(matrix_folder.index)
                else:
                    selection_folder = st.session_state.get("matrix_selector_folder")
                    if selection_folder and selection_folder.get("selection", {}).get("rows"):
                        selected_folders = [matrix_folder.index[i] for i in selection_folder["selection"]["rows"] if i < len(matrix_folder.index)]

                if selected_folders:
                    st.markdown(f"### 🎯 Links to place from Folder rows: `{len(selected_folders)}` selected")
                    filtered_folder = data_folder[data_folder['From Folder'].isin(selected_folders)].copy()
                    display_filtered_folder = filtered_folder[['Focus URL', 'Page to Edit (Source)', 'To Folder', 'Link Destination', 'Score', 'Existing Link']].sort_values(by=['Focus URL', 'Score'], ascending=[True, False]).copy()
                    display_filtered_folder.loc[display_filtered_folder.duplicated('Focus URL'), 'Focus URL'] = ""
                    
                    final_display_folder = display_filtered_folder[['Page to Edit (Source)', 'To Folder', 'Link Destination', 'Score', 'Existing Link']].copy()
                    final_display_folder['Score'] = final_display_folder['Score'].apply(lambda x: f"{int(x)}% ⚠️" if x >= 95 else f"{int(x)}%")

                    st.dataframe(
                        final_display_folder.style.map(color_score, subset=['Score']),
                        width='stretch',
                        hide_index=True
                    )
            else:
                st.warning(f"No {dir_folder.lower()} links found.")

        # ========================================================
        # 7. TOPIC HUBS OVERVIEW
        # ========================================================
        st.divider()
        st.subheader("🏗️ Topic Hubs Overview (Outbound)")
        st.info("This overview shows the standard outbound perspective for your focus URLs.")

        overview_data = data[data['Direction'] == 'Outbound'].copy()
        hub_stats = overview_data.groupby('From Hub')['Score'].mean().sort_values(ascending=False)

        tab_strong, tab_avg, tab_weak = st.tabs(["🟢 Strong (>= 85%)", "🟡 Average (70-84%)", "🔴 Weak (< 70%)"])

        def render_hub_group(hubs_series):
            if hubs_series.empty:
                st.info("Geen hubs gevonden.")
            else:
                for hub, avg_score in hubs_series.items():
                    hub_df = overview_data[overview_data['From Hub'] == hub].copy()
                    with st.expander(f"📁 HUB: {hub} ({round(avg_score)}%)"):
                        display_hub = hub_df[['Page to Edit (Source)', 'To Hub', 'Link Destination', 'Score', 'Existing Link']].sort_values(by=['Page to Edit (Source)', 'Score'], ascending=[True, False]).copy()
                        display_hub.loc[display_hub.duplicated('Page to Edit (Source)'), 'Page to Edit (Source)'] = ""
                        
                        # Apply warning label
                        display_hub['Score'] = display_hub['Score'].apply(lambda x: f"{int(x)}% ⚠️" if x >= 95 else f"{int(x)}%")
                        
                        st.dataframe(
                            display_hub.style.map(color_score, subset=['Score']),
                            width='stretch',
                            hide_index=True
                        )

        with tab_strong: render_hub_group(hub_stats[hub_stats >= 85])
        with tab_avg: render_hub_group(hub_stats[(hub_stats >= 70) & (hub_stats < 85)])
        with tab_weak: render_hub_group(hub_stats[hub_stats < 70])
        
        # ========================================================
        # 8. EXPORT CSV
        # ========================================================
        st.divider()
        export_df = data.copy()
        export_df = export_df.sort_values(by=['Direction', 'From Hub', 'Focus URL', 'Score'], ascending=[True, True, True, False])
        export_df['Score'] = export_df['Score'].apply(lambda x: f"{round(x)}% ⚠️" if x >= 95 else f"{round(x)}%")
        
        export_df.loc[export_df.duplicated(subset=['Direction', 'From Hub', 'Focus URL']), 'Focus URL'] = ""
        export_df.loc[export_df.duplicated(subset=['Direction', 'From Hub']), 'From Hub'] = ""
        export_df.loc[export_df.duplicated(subset=['Direction']), 'Direction'] = ""
        
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, sep=';')
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="seo_bidirectional_links_matrix.csv",
            mime="text/csv",
            width="stretch"
        )
