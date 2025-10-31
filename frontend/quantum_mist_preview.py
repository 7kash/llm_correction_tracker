import streamlit as st

st.set_page_config(page_title="Quantum Mist Preview", page_icon="✨", layout="centered")

# --- Theme tokens (your 4 picks) ---
COLORS = {
    "mist_shadow": "#CBD5E1",
    "ice_cyan": "#CFFAFE",
    "cool_slate": "#475569",
    "soft_blossom": "#FBCFE8",
    "text_strong": "#334155",   # slightly darker than Cool Slate for headings
    "surface_border": "#B8C4D4" # border derived from Mist Shadow
}

# --- Global styles ---
st.markdown(f"""
<style>
:root {{
  --bg: {COLORS['ice_cyan']};
  --surface: #FFFFFF;
  --surface-border: {COLORS['surface_border']};
  --chip: {COLORS['mist_shadow']};
  --text: {COLORS['cool_slate']};
  --text-strong: {COLORS['text_strong']};
  --primary: {COLORS['cool_slate']};
  --accent: {COLORS['soft_blossom']};
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: var(--bg);
}}

.block-container {{
  padding-top: 2.5rem;
  padding-bottom: 4rem;
}}

.nav {{
  background: linear-gradient(180deg, rgba(255,255,255,0.75), rgba(255,255,255,0.55));
  border: 1px solid var(--surface-border);
  color: var(--text);
  border-radius: 14px;
  padding: 14px 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}}

.nav .logo {{
  font-weight: 800;
  color: var(--text-strong);
  letter-spacing: .2px;
}}

.nav .links a {{
  color: var(--text);
  text-decoration: none;
  margin-left: 22px;
  font-weight: 600;
}}

.nav .links a:hover {{
  text-decoration: underline;
}}

.card {{
  margin-top: 16px;
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 18px;
  padding: 26px;
  box-shadow: 0 10px 24px rgba(17,24,39,0.06);
}}

.tag {{
  display: inline-block;
  padding: 8px 14px;
  border-radius: 999px;
  background: var(--chip);
  color: var(--text);
  border: 1px solid var(--surface-border);
  font-weight: 600;
}}

.h1 {{
  margin-top: 16px;
  margin-bottom: 8px;
  font-weight: 800;
  font-size: 30px;
  color: var(--text-strong);
  letter-spacing: .2px;
}}

.p {{
  color: var(--text);
  font-size: 16px;
  margin-bottom: 22px;
}}

.row {{
  display: flex; 
  gap: 14px;
  flex-wrap: wrap;
}}

.btn {{
  appearance: none;
  border: none;
  border-radius: 12px;
  padding: 14px 18px;
  font-weight: 700;
  cursor: pointer;
  transition: transform .06s ease, box-shadow .2s ease, background .2s ease;
  letter-spacing: .25px;
}}

.btn:active {{ transform: translateY(1px); }}

.btn-primary {{
  background: var(--primary);
  color: #FFFFFF;
  box-shadow: 0 6px 14px rgba(71,85,105,0.35);
}}

.btn-primary:hover {{ filter: brightness(1.05); }}

.btn-accent {{
  background: var(--accent);
  color: var(--text);
  border: 1px solid var(--surface-border);
  box-shadow: 0 6px 14px rgba(251, 207, 232, 0.35);
}}

.btn-accent:hover {{ filter: brightness(0.98); }}

.footer-note {{
  margin-top: 14px;
  color: var(--text);
  opacity: .8;
  font-size: 13px;
}}
</style>
""", unsafe_allow_html=True)

# --- Top nav ---
st.markdown("""
<div class="nav">
  <div class="logo">Logo</div>
  <div class="links">
    <a href="#">Home</a>
    <a href="#">About</a>
  </div>
</div>
""", unsafe_allow_html=True)

# --- Card preview ---
st.markdown(f"""
<div class="card">
  <span class="tag">Tag</span>
  <div class="h1">Card Title</div>
  <div class="p">This is some text inside of a card.</div>
  <div class="row">
    <button class="btn btn-primary">Primary Button</button>
    <button class="btn btn-accent">Accent Button</button>
  </div>
  <div class="footer-note">Mist Shadow #{COLORS['mist_shadow'][1:]}, Ice Cyan #{COLORS['ice_cyan'][1:]}, Cool Slate #{COLORS['cool_slate'][1:]}, Soft Blossom #{COLORS['soft_blossom'][1:]}</div>
</div>
""", unsafe_allow_html=True)

# Optional: little swatch row
with st.expander("Show color swatches"):
    cols = st.columns(4)
    names = ["Mist Shadow", "Ice Cyan", "Cool Slate", "Soft Blossom"]
    hexes = [COLORS['mist_shadow'], COLORS['ice_cyan'], COLORS['cool_slate'], COLORS['soft_blossom']]
    for c, name, hx in zip(cols, names, hexes):
        c.markdown(f"""
        <div style="border:1px solid #d9dee7;border-radius:12px;overflow:hidden">
          <div style="height:56px;background:{hx}"></div>
          <div style="padding:8px 10px;font-weight:600;color:#334155">{name}</div>
          <div style="padding:0 10px 10px 10px;color:#475569">{hx}</div>
        </div>
        """, unsafe_allow_html=True)
