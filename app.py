import os
import time
from pathlib import Path
from string import Template
from typing import Dict, Any, List, Optional

import streamlit as st
import yaml
import pandas as pd

import google.generativeai as genai
import openai
import anthropic

# ======================
# CONFIG
# ======================

st.set_page_config(
    page_title="Flower Agents Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

AGENTS_YAML_PATH = Path(__file__).parent / "agents.yaml"

# ======================
# CONSTANTS
# ======================

FLOWER_THEMES = [
    {
        "id": "sakura",
        "label_en": "Sakura Blossom",
        "label_zh": "春櫻之語",
        "primary": "#f472b6",
        "secondary": "#fdf2f8",
        "bg_light": "linear-gradient(135deg, #fff7fb 0%, #ffe4f1 100%)",
        "bg_dark": "linear-gradient(135deg, #3b0820 0%, #120010 100%)",
    },
    {
        "id": "rose",
        "label_en": "Rose Garden",
        "label_zh": "玫瑰花園",
        "primary": "#e11d48",
        "secondary": "#fee2e2",
        "bg_light": "linear-gradient(135deg, #fff1f2 0%, #fecaca 100%)",
        "bg_dark": "linear-gradient(135deg, #450a0a 0%, #111827 100%)",
    },
    {
        "id": "lavender",
        "label_en": "Lavender Field",
        "label_zh": "薰衣草之野",
        "primary": "#8b5cf6",
        "secondary": "#ede9fe",
        "bg_light": "linear-gradient(135deg, #f5f3ff 0%, #e0e7ff 100%)",
        "bg_dark": "linear-gradient(135deg, #1e1b4b 0%, #020617 100%)",
    },
    {
        "id": "sunflower",
        "label_en": "Sunflower Sky",
        "label_zh": "向日葵之光",
        "primary": "#eab308",
        "secondary": "#fef9c3",
        "bg_light": "linear-gradient(135deg, #fefce8 0%, #fef3c7 100%)",
        "bg_dark": "linear-gradient(135deg, #422006 0%, #030712 100%)",
    },
    {
        "id": "lotus",
        "label_en": "Lotus Pond",
        "label_zh": "靜水蓮花",
        "primary": "#22c55e",
        "secondary": "#dcfce7",
        "bg_light": "linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%)",
        "bg_dark": "linear-gradient(135deg, #052e16 0%, #0f172a 100%)",
    },
    {
        "id": "camellia",
        "label_en": "Camellia Mist",
        "label_zh": "山茶霧光",
        "primary": "#f97316",
        "secondary": "#ffedd5",
        "bg_light": "linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%)",
        "bg_dark": "linear-gradient(135deg, #431407 0%, #0b1120 100%)",
    },
    {
        "id": "orchid",
        "label_en": "Orchid Dream",
        "label_zh": "蘭花幽夢",
        "primary": "#a855f7",
        "secondary": "#fae8ff",
        "bg_light": "linear-gradient(135deg, #fdf4ff 0%, #e9d5ff 100%)",
        "bg_dark": "linear-gradient(135deg, #3b0764 0%, #020617 100%)",
    },
    {
        "id": "peony",
        "label_en": "Peony Silk",
        "label_zh": "牡丹錦雲",
        "primary": "#db2777",
        "secondary": "#fce7f3",
        "bg_light": "linear-gradient(135deg, #fdf2f8 0%, #fbcfe8 100%)",
        "bg_dark": "linear-gradient(135deg, #500724 0%, #020617 100%)",
    },
    {
        "id": "cherry_blossom_night",
        "label_en": "Cherry Blossom Night",
        "label_zh": "夜櫻流光",
        "primary": "#fb7185",
        "secondary": "#fecdd3",
        "bg_light": "linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%)",
        "bg_dark": "linear-gradient(135deg, #2e1065 0%, #020617 100%)",
    },
    {
        "id": "wisteria",
        "label_en": "Wisteria Veil",
        "label_zh": "紫藤輕紗",
        "primary": "#6366f1",
        "secondary": "#e0e7ff",
        "bg_light": "linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)",
        "bg_dark": "linear-gradient(135deg, #1e1b4b 0%, #020617 100%)",
    },
    {
        "id": "hydrangea",
        "label_en": "Hydrangea Dew",
        "label_zh": "繡球晨露",
        "primary": "#22d3ee",
        "secondary": "#cffafe",
        "bg_light": "linear-gradient(135deg, #ecfeff 0%, #cffafe 100%)",
        "bg_dark": "linear-gradient(135deg, #082f49 0%, #020617 100%)",
    },
    {
        "id": "magnolia",
        "label_en": "Magnolia Light",
        "label_zh": "玉蘭初曦",
        "primary": "#facc15",
        "secondary": "#fef9c3",
        "bg_light": "linear-gradient(135deg, #fffbeb 0%, #fef9c3 100%)",
        "bg_dark": "linear-gradient(135deg, #422006 0%, #020617 100%)",
    },
    {
        "id": "plum_blossom",
        "label_en": "Plum Blossom Frost",
        "label_zh": "寒梅映雪",
        "primary": "#f97316",
        "secondary": "#fed7aa",
        "bg_light": "linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%)",
        "bg_dark": "linear-gradient(135deg, #3f1f0a 0%, #020617 100%)",
    },
    {
        "id": "jasmine",
        "label_en": "Jasmine Breeze",
        "label_zh": "茉莉微風",
        "primary": "#10b981",
        "secondary": "#d1fae5",
        "bg_light": "linear-gradient(135deg, #ecfdf5 0%, #bbf7d0 100%)",
        "bg_dark": "linear-gradient(135deg, #064e3b 0%, #020617 100%)",
    },
    {
        "id": "iris",
        "label_en": "Iris Twilight",
        "label_zh": "鳶尾暮光",
        "primary": "#4f46e5",
        "secondary": "#e0e7ff",
        "bg_light": "linear-gradient(135deg, #eef2ff 0%, #e0f2fe 100%)",
        "bg_dark": "linear-gradient(135deg, #111827 0%, #020617 100%)",
    },
    {
        "id": "poppy",
        "label_en": "Poppy Ember",
        "label_zh": "罌粟餘燼",
        "primary": "#f97316",
        "secondary": "#fed7aa",
        "bg_light": "linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%)",
        "bg_dark": "linear-gradient(135deg, #7c2d12 0%, #020617 100%)",
    },
    {
        "id": "azalea",
        "label_en": "Azalea Bloom",
        "label_zh": "杜鵑綻放",
        "primary": "#f43f5e",
        "secondary": "#ffe4e6",
        "bg_light": "linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%)",
        "bg_dark": "linear-gradient(135deg, #4a0410 0%, #020617 100%)",
    },
    {
        "id": "gardenia",
        "label_en": "Gardenia Moon",
        "label_zh": "梔子月白",
        "primary": "#22c55e",
        "secondary": "#dcfce7",
        "bg_light": "linear-gradient(135deg, #f4f4f5 0%, #d4d4d8 100%)",
        "bg_dark": "linear-gradient(135deg, #0f172a 0%, #020617 100%)",
    },
    {
        "id": "bougainvillea",
        "label_en": "Bougainvillea Flame",
        "label_zh": "九重葛之焰",
        "primary": "#ec4899",
        "secondary": "#fce7f3",
        "bg_light": "linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%)",
        "bg_dark": "linear-gradient(135deg, #4a044e 0%, #020617 100%)",
    },
    {
        "id": "lotus_night",
        "label_en": "Lotus Night Lake",
        "label_zh": "夜荷映月",
        "primary": "#22c55e",
        "secondary": "#bbf7d0",
        "bg_light": "linear-gradient(135deg, #ecfeff 0%, #dcfce7 100%)",
        "bg_dark": "linear-gradient(135deg, #022c22 0%, #020617 100%)",
    },
]

PROMPT_TEMPLATES = [
    {
        "id": "general_assistant",
        "label_en": "General Helpful Assistant",
        "label_zh": "通用智慧助理",
        "system_prompt": "You are a helpful and polite AI assistant.",
    },
    {
        "id": "research_agent",
        "label_en": "Research & Analysis",
        "label_zh": "研究與分析代理",
        "system_prompt": "You are a careful research assistant. Always cite sources when possible.",
    },
    {
        "id": "summary_agent",
        "label_en": "Summarization Expert",
        "label_zh": "摘要專家",
        "system_prompt": "You summarize text concisely, preserving key details.",
    },
]

I18N = {
    "en": {
        "title": "Flower Agents Studio",
        "agents_console": "Agents Console",
        "attachment_chat": "Attachment Chat",
        "notes_studio": "Notes Studio",
        "dashboard": "Dashboard",
        "model_settings": "Model Settings",
        "prompt_template": "Prompt Template",
        "theme": "Theme",
        "language": "Language",
        "manage_api_key": "Manage API Keys",
        "provider": "Provider",
        "model": "Model",
        "max_tokens": "Max tokens",
        "run_agents": "Run Agents",
        "system_prompt": "System Prompt",
        "user_prompt": "User Prompt",
        "appearance": "Appearance",
        "dark_mode": "Dark mode",
        "stats_title": "Agent Activity Dashboard",
        "stat_total_runs": "Total Runs",
        "stat_total_tokens": "Total Tokens (approx)",
        "stat_avg_latency": "Avg Latency (s)",
        "stat_last_provider": "Last Provider",
        "attachments_title": "Attachment Chat",
        "notes_title": "Notes Studio",
        "notes_hint": "These notes persist only during this session.",
        "api_env_in_use": "Using environment variable (value is hidden).",
        "api_input_label_openai": "OpenAI API key",
        "api_input_label_gemini": "Gemini API key",
        "api_input_label_anthropic": "Anthropic API key",
    },
    "zhTW": {
        "title": "花語智能工作室",
        "agents_console": "智能代理主控台",
        "attachment_chat": "附件對話",
        "notes_studio": "筆記工作室",
        "dashboard": "互動儀表板",
        "model_settings": "模型設定",
        "prompt_template": "提示模板",
        "theme": "主題風格",
        "language": "介面語言",
        "manage_api_key": "管理 API 金鑰",
        "provider": "模型提供商",
        "model": "模型",
        "max_tokens": "最大 Token 數",
        "run_agents": "執行代理",
        "system_prompt": "系統提示 (System Prompt)",
        "user_prompt": "使用者提示 (User Prompt)",
        "appearance": "外觀設定",
        "dark_mode": "深色模式",
        "stats_title": "智能代理互動儀表板",
        "stat_total_runs": "總執行次數",
        "stat_total_tokens": "Token 使用總量 (約)",
        "stat_avg_latency": "平均延遲 (秒)",
        "stat_last_provider": "最近使用提供商",
        "attachments_title": "附件對話",
        "notes_title": "筆記工作室",
        "notes_hint": "筆記只在本次 Session 期間保留。",
        "api_env_in_use": "正在使用環境變數 (實際值不顯示)。",
        "api_input_label_openai": "OpenAI API 金鑰",
        "api_input_label_gemini": "Gemini API 金鑰",
        "api_input_label_anthropic": "Anthropic API 金鑰",
    },
}

MODEL_OPTIONS = {
    "Gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    "OpenAI": [
        "gpt-5-nano",
        "gpt-4o-mini",
        "gpt-4.1-mini",
    ],
    "Anthropic": [
        "claude-3-5-sonnet-latest",
    ],
}

# ======================
# HELPERS: STATE / THEME / I18N
# ======================

def init_session_state():
    defaults = {
        "page": "agents",
        "provider": "Gemini",
        "model": "gemini-2.5-flash",
        "prompt_template_id": PROMPT_TEMPLATES[0]["id"],
        "language": "en",
        "theme_id": "sakura",
        "dark_mode": False,
        "max_tokens": 2048,
        "temperature": 0.7,
        "system_prompt": PROMPT_TEMPLATES[0]["system_prompt"],
        "user_prompt": "",
        "runs_log": [],
        "notes": "",
        "last_response": "",
        "last_error": "",
        "openai_api_key": "",
        "gemini_api_key": "",
        "anthropic_api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_theme(theme_id: str) -> Dict[str, Any]:
    for t in FLOWER_THEMES:
        if t["id"] == theme_id:
            return t
    return FLOWER_THEMES[0]


def apply_theme_css(theme_id: str, dark_mode: bool):
    theme = get_theme(theme_id)
    bg = theme["bg_dark"] if dark_mode else theme["bg_light"]
    text_color = "#e5e7eb" if dark_mode else "#111827"
    card_bg = "rgba(15,23,42,0.85)" if dark_mode else "rgba(255,255,255,0.9)"
    accent = theme["primary"]
    accent_soft = theme["secondary"]

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg};
            color: {text_color};
        }}
        .flower-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 1rem 1.25rem;
            border: 1px solid rgba(148,163,184,0.3);
            backdrop-filter: blur(18px);
        }}
        .flower-pill {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.1rem 0.55rem;
            font-size: 0.70rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            border: 1px solid rgba(148,163,184,0.6);
            color: {accent};
            background: {accent_soft};
        }}
        .accent-text {{
            color: {accent} !important;
        }}
        .accent-border {{
            border-color: {accent} !important;
        }}
        .accent-bg-soft {{
            background: {accent_soft} !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.85rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def i18n(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return I18N[lang].get(key, key)


def get_prompt_template_by_id(template_id: str) -> Dict[str, Any]:
    for t in PROMPT_TEMPLATES:
        if t["id"] == template_id:
            return t
    return PROMPT_TEMPLATES[0]


# ======================
# HELPERS: YAML / METRICS
# ======================

def load_agents_config(provider: str, model: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    if not AGENTS_YAML_PATH.exists():
        return {}
    text = AGENTS_YAML_PATH.read_text(encoding="utf-8")
    tmpl = Template(text)
    rendered = tmpl.safe_substitute(
        PROVIDER=provider,
        MODEL_NAME=model,
        MAX_TOKENS=max_tokens,
        TEMPERATURE=temperature,
    )
    return yaml.safe_load(rendered)


def record_run(provider: str, model: str, tokens: int, latency: float):
    st.session_state.runs_log.append(
        {
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "tokens": tokens,
            "latency": latency,
        }
    )


def build_dashboard_dataframe() -> pd.DataFrame:
    if not st.session_state.runs_log:
        return pd.DataFrame(columns=["timestamp", "provider", "model", "tokens", "latency"])
    df = pd.DataFrame(st.session_state.runs_log)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # rough heuristic: 4 chars per token
    return max(1, len(text) // 4)


# ======================
# HELPERS: API KEYS & LLM CALLS
# ======================

def get_api_key_for_provider(provider: str) -> Optional[str]:
    if provider == "OpenAI":
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
        return st.session_state.get("openai_api_key") or None

    if provider == "Gemini":
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return env_key
        return st.session_state.get("gemini_api_key") or None

    if provider == "Anthropic":
        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            return env_key
        return st.session_state.get("anthropic_api_key") or None

    return None


def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """
    Call Gemini / OpenAI / Anthropic using their Python SDKs.
    Returns dict: { 'text': str, 'tokens': int, 'latency': float }
    """
    api_key = get_api_key_for_provider(provider)
    if not api_key:
        raise RuntimeError(f"{provider} API key is not set. Provide it via environment or the sidebar.")

    start = time.time()
    text = ""

    if provider == "Gemini":
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model)
        # Combine system + user as context
        prompt_parts = [
            f"System: {system_prompt}",
            f"User: {user_prompt}",
        ]
        resp = gmodel.generate_content(
            prompt_parts,
            generation_config={
                "temperature": float(temperature),
                "max_output_tokens": int(max_tokens),
            },
        )
        text = resp.text or ""

    elif provider == "OpenAI":
        client = openai.OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        text = completion.choices[0].message.content or ""

    elif provider == "Anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Concatenate all text blocks
        chunks = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                chunks.append(block.get("text", ""))
        text = "".join(chunks)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    latency = time.time() - start
    tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + estimate_tokens(text)
    record_run(provider, model, tokens=tokens, latency=latency)

    return {"text": text, "tokens": tokens, "latency": latency}


# ======================
# PAGES
# ======================

def page_agents_console():
    t = i18n
    st.markdown(
        f"""
        <div class="flower-card">
            <div class="flower-pill">{t("agents_console")}</div>
            <h1 style="margin-top:0.3rem;margin-bottom:0.3rem;">{t("title")}</h1>
            <p style="font-size:0.85rem;opacity:0.8;">
                Configure your providers, models, prompts, and run LLM-powered agent workflows (backed by <code>agents.yaml</code>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    col_main, col_side = st.columns([2.2, 1.1])

    with col_main:
        st.subheader(t("system_prompt"))
        st.session_state.system_prompt = st.text_area(
            label="",
            value=st.session_state.system_prompt,
            height=150,
            key="system_prompt_text",
        )

        st.subheader(t("user_prompt"))
        st.session_state.user_prompt = st.text_area(
            label="",
            value=st.session_state.user_prompt,
            height=180,
            key="user_prompt_text",
        )

        if st.session_state.last_error:
            st.error(st.session_state.last_error)
        if st.session_state.last_response:
            st.markdown("#### Response")
            st.markdown(
                f"<div class='flower-card'>{st.session_state.last_response}</div>",
                unsafe_allow_html=True,
            )

    with col_side:
        st.markdown("##### " + t("model_settings"))
        st.markdown(f"**{t('provider')}**")
        st.selectbox(
            label="provider",
            options=list(MODEL_OPTIONS.keys()),
            key="provider",
            label_visibility="collapsed",
        )
        provider = st.session_state.provider

        st.markdown(f"**{t('model')}**")
        st.selectbox(
            label="model",
            options=MODEL_OPTIONS[provider],
            key="model",
            label_visibility="collapsed",
        )

        st.markdown(f"**{t('max_tokens')}**")
        st.session_state.max_tokens = st.slider(
            label=t("max_tokens"),
            min_value=128,
            max_value=8192,
            value=st.session_state.max_tokens,
            step=128,
            label_visibility="collapsed",
        )

        st.markdown("**Temperature**")
        st.session_state.temperature = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            label_visibility="collapsed",
        )

        st.markdown("**" + t("prompt_template") + "**")
        template_labels = [
            f"{pt['label_en']} / {pt['label_zh']}" for pt in PROMPT_TEMPLATES
        ]
        template_ids = [pt["id"] for pt in PROMPT_TEMPLATES]
        idx = template_ids.index(st.session_state.prompt_template_id)
        selected_label = st.selectbox(
            label="Template",
            options=template_labels,
            index=idx,
            label_visibility="collapsed",
        )
        st.session_state.prompt_template_id = template_ids[
            template_labels.index(selected_label)
        ]
        tmpl = get_prompt_template_by_id(st.session_state.prompt_template_id)
        if st.session_state.system_prompt == PROMPT_TEMPLATES[0]["system_prompt"]:
            st.session_state.system_prompt = tmpl["system_prompt"]

        st.write("")
        run_button = st.button("▶ " + t("run_agents"), use_container_width=True)

        st.write("")
        with st.expander("Rendered agents.yaml (read-only)"):
            cfg = load_agents_config(
                st.session_state.provider,
                st.session_state.model,
                st.session_state.max_tokens,
                st.session_state.temperature,
            )
            if cfg:
                st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")
            else:
                st.caption("No agents.yaml found or unable to load.")

    if run_button:
        provider = st.session_state.provider
        model = st.session_state.model
        max_tokens = st.session_state.max_tokens
        temperature = st.session_state.temperature
        system_prompt = st.session_state.system_prompt
        user_prompt = st.session_state.user_prompt

        st.session_state.last_error = ""
        st.session_state.last_response = ""

        try:
            with st.spinner("Calling LLM..."):
                result = call_llm(
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            st.session_state.last_response = result["text"]
            st.success(
                f"LLM call succeeded with {result['tokens']} estimated tokens in {result['latency']:.2f}s."
            )
        except Exception as e:
            st.session_state.last_error = str(e)
            st.error(f"Error during LLM call: {e}")


def page_attachment_chat():
    t = i18n
    st.markdown(
        f"""
        <div class="flower-card">
            <div class="flower-pill">{t("attachment_chat")}</div>
            <h2 style="margin-top:0.3rem;margin-bottom:0.3rem;">{t("attachments_title")}</h2>
            <p style="font-size:0.85rem;opacity:0.8;">
                Upload documents and chat with them using your current provider/model settings.
                Wire this page to your RAG / vector store pipeline as needed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    col_upload, col_chat = st.columns([1.2, 2.0])

    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload one or more files",
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.write("Files uploaded:")
            for f in uploaded_files:
                st.write(f"- {f.name} ({f.size} bytes)")

    with col_chat:
        st.subheader("Chat")
        attachment_prompt = st.text_area(
            label="Ask about your uploaded documents",
            height=160,
            key="attachment_prompt",
        )
        col_left, col_right = st.columns([1, 1])
        with col_left:
            ask_button = st.button("Ask with current model")
        with col_right:
            clear_button = st.button("Clear", type="secondary")

        if clear_button:
            st.session_state.attachment_prompt = ""

        if ask_button:
            provider = st.session_state.provider
            model = st.session_state.model
            system_prompt = (
                "You are a helpful assistant that answers questions about the user's uploaded files. "
                "Use only the information that would reasonably be contained in those files."
            )
            try:
                with st.spinner("Thinking about your documents..."):
                    result = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=attachment_prompt,
                        max_tokens=st.session_state.max_tokens,
                        temperature=st.session_state.temperature,
                    )
                st.markdown("#### Answer")
                st.write(result["text"])
            except Exception as e:
                st.error(f"Error during LLM call: {e}")


def page_notes_studio():
    t = i18n
    st.markdown(
        f"""
        <div class="flower-card">
            <div class="flower-pill">{t("notes_studio")}</div>
            <h2 style="margin-top:0.3rem;margin-bottom:0.3rem;">{t("notes_title")}</h2>
            <p style="font-size:0.85rem;opacity:0.8;">{t("notes_hint")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    st.session_state.notes = st.text_area(
        label="",
        value=st.session_state.notes,
        height=420,
    )


def page_dashboard():
    t = i18n
    st.markdown(
        f"""
        <div class="flower-card">
            <div class="flower-pill">{t("dashboard")}</div>
            <h2 style="margin-top:0.3rem;margin-bottom:0.3rem;">{t("stats_title")}</h2>
            <p style="font-size:0.85rem;opacity:0.8;">
                Track your LLM usage across providers and models: runs, tokens, latency and model mix.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    df = build_dashboard_dataframe()
    total_runs = len(df)
    total_tokens = int(df["tokens"].sum()) if not df.empty else 0
    avg_latency = float(df["latency"].mean()) if not df.empty else 0.0
    last_provider = df.iloc[-1]["provider"] if not df.empty else "-"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("stat_total_runs"), f"{total_runs}")
    col2.metric(t("stat_total_tokens"), f"{total_tokens}")
    col3.metric(t("stat_avg_latency"), f"{avg_latency:.2f}")
    col4.metric(t("stat_last_provider"), last_provider)

    st.write("")

    if df.empty:
        st.info("Run at least one LLM call to populate the dashboard.")
        return

    tab_timeline, tab_models, tab_latency = st.tabs(
        ["Usage Over Time", "Model Mix", "Latency Distribution"]
    )

    with tab_timeline:
        st.markdown("#### Runs & Tokens over Time")
        df_timeline = df.set_index("datetime")[["tokens"]]
        st.line_chart(df_timeline)

    with tab_models:
        st.markdown("#### Provider / Model Usage")
        model_counts = df.groupby(["provider", "model"]).size().reset_index(name="runs")
        pivot = model_counts.pivot_table(
            index="model", columns="provider", values="runs", fill_value=0
        )
        st.bar_chart(pivot)

    with tab_latency:
        st.markdown("#### Latency (seconds)")
        st.bar_chart(df.set_index("datetime")[["latency"]])


# ======================
# SIDEBAR
# ======================

def sidebar():
    t = i18n
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.75rem;">
                <div style="padding:0.4rem 0.6rem;border-radius:0.75rem;
                            background:rgba(15,23,42,0.9);color:white;
                            border:1px solid rgba(148,163,184,0.6);">
                    <span style="font-weight:700;">花</span>
                </div>
                <div>
                    <div style="font-size:0.95rem;font-weight:700;">{t("title")}</div>
                    <div style="font-size:0.75rem;opacity:0.7;">Multi-provider Agents Console</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_labels = {
            "agents": t("agents_console"),
            "attachment": t("attachment_chat"),
            "notes": t("notes_studio"),
            "dashboard": t("dashboard"),
        }
        st.session_state.page = st.radio(
            label="Navigation",
            options=list(nav_labels.keys()),
            format_func=lambda x: nav_labels[x],
            index=list(nav_labels.keys()).index(st.session_state.page),
        )

        st.markdown("---")

        st.markdown(f"**{t('model_settings')}**")
        st.selectbox(
            label=t("provider"),
            options=list(MODEL_OPTIONS.keys()),
            key="provider",
        )
        prov = st.session_state.provider
        st.selectbox(
            label=t("model"),
            options=MODEL_OPTIONS[prov],
            key="model",
        )
        st.slider(
            label=t("max_tokens"),
            min_value=128,
            max_value=8192,
            value=st.session_state.max_tokens,
            step=128,
            key="max_tokens",
        )

        st.markdown("---")
        st.markdown(f"**{t('appearance')}**")

        theme_labels = [
            (ft["label_en"] if st.session_state.language == "en" else ft["label_zh"])
            for ft in FLOWER_THEMES
        ]
        theme_ids = [ft["id"] for ft in FLOWER_THEMES]
        idx = theme_ids.index(st.session_state.theme_id)
        selected_theme = st.selectbox(
            label=t("theme"),
            options=theme_labels,
            index=idx,
        )
        st.session_state.theme_id = theme_ids[theme_labels.index(selected_theme)]

        st.toggle(
            label=t("dark_mode"),
            value=st.session_state.dark_mode,
            key="dark_mode",
        )

        st.markdown("**" + t("language") + "**")
        lang_choice = st.radio(
            "",
            options=["en", "zhTW"],
            index=0 if st.session_state.language == "en" else 1,
            horizontal=True,
            format_func=lambda x: "EN" if x == "en" else "繁體",
        )
        st.session_state.language = lang_choice

        st.markdown("---")
        st.markdown("**" + t("manage_api_key") + "**")

        with st.expander("OpenAI"):
            if os.getenv("OPENAI_API_KEY"):
                st.caption(t("api_env_in_use"))
            else:
                st.text_input(
                    label=t("api_input_label_openai"),
                    type="password",
                    key="openai_api_key",
                )

        with st.expander("Gemini"):
            if os.getenv("GEMINI_API_KEY"):
                st.caption(t("api_env_in_use"))
            else:
                st.text_input(
                    label=t("api_input_label_gemini"),
                    type="password",
                    key="gemini_api_key",
                )

        with st.expander("Anthropic"):
            if os.getenv("ANTHROPIC_API_KEY"):
                st.caption(t("api_env_in_use"))
            else:
                st.text_input(
                    label=t("api_input_label_anthropic"),
                    type="password",
                    key="anthropic_api_key",
                )


# ======================
# MAIN
# ======================

def main():
    init_session_state()
    apply_theme_css(st.session_state.theme_id, st.session_state.dark_mode)
    sidebar()

    page = st.session_state.page
    if page == "agents":
        page_agents_console()
    elif page == "attachment":
        page_attachment_chat()
    elif page == "notes":
        page_notes_studio()
    elif page == "dashboard":
        page_dashboard()
    else:
        page_agents_console()


if __name__ == "__main__":
    main()
