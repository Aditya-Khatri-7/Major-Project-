"""Streamlit front end for the Forensics Agent API.

Run:  streamlit run frontend/app.py        (API address: FORENSICS_API_URL, default http://localhost:8000)
"""
from __future__ import annotations

import io
import os

import requests
import streamlit as st

try:  # optional: clipboard paste (pip install streamlit-paste-button)
    from streamlit_paste_button import paste_image_button
except ImportError:
    paste_image_button = None

API_URL = os.getenv("FORENSICS_API_URL", "http://localhost:8000").rstrip("/")
COLORS = {"synthetic": "#d62728", "authentic": "#2ca02c", "uncertain": "#ff7f0e"}


def _post_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            return f"{exc.response.status_code}: {exc.response.json().get('detail', exc.response.text)}"
        except ValueError:
            return f"{exc.response.status_code}: {exc.response.text[:300]}"
    return str(exc)


def submit_correction(job_id: str, label: str) -> None:
    try:
        r = requests.post(f"{API_URL}/jobs/{job_id}/correction", json={"label": label}, timeout=30)
        r.raise_for_status()
        st.success(f"Recorded human label '{label}' for job {job_id[:8]}.")
    except Exception as exc:
        st.error(f"Could not save correction: {_post_error(exc)}")


def show_result(data: dict) -> None:
    verdict = (data.get("verdict") or "uncertain").lower()
    color = COLORS.get(verdict, "#888888")
    confidence = (data.get("confidence") or 0.0) * 100
    fused = data.get("fused_score")
    review = "  |  Human review recommended" if data.get("escalate_to_human") else ""
    fused_txt = f"  |  fused P(synthetic) = {fused:.2f}" if fused is not None else ""
    st.markdown(
        f"<div style='border-left:6px solid {color};padding:12px 16px;border-radius:6px;"
        f"background:{color}18'><h2 style='margin:0;color:{color}'>{verdict.upper()}</h2>"
        f"<p style='margin:4px 0'>Confidence {confidence:.0f}%{fused_txt}{review}</p></div>",
        unsafe_allow_html=True,
    )
    if data.get("summary"):
        st.write(data["summary"])
    if data.get("verifier_notes"):
        st.info(f"Verifier: {data['verifier_notes']}")
    if data.get("reflexion_used"):
        st.caption("Reflexion: the judge model re-examined this case after the tools disagreed.")

    if data.get("gradcam_url"):
        st.subheader("Where the detector looked (Grad-CAM)")
        try:
            resp = requests.get(f"{API_URL}{data['gradcam_url']}", timeout=30)
            resp.raise_for_status()
            st.image(resp.content, caption="Heatmap shows model attention, not a verified manipulation mask.")
        except Exception as exc:
            st.warning(f"Heatmap unavailable: {_post_error(exc)}")

    with st.expander("Tool-by-tool breakdown", expanded=True):
        for v in data.get("tool_verdicts", []):
            status = " (error, excluded from fusion)" if v.get("error") else ""
            st.markdown(f"**{v['tool']}**{status} - P(synthetic) `{v['score']:.3f}`, confidence `{v['confidence']:.3f}`")
            st.progress(min(max(float(v["score"]), 0.0), 1.0))
            st.caption(v["explanation"])

    with st.expander("Retrieved evidence and citations"):
        if data.get("citations"):
            st.write("Cited sources: " + ", ".join(data["citations"]))
        else:
            st.write("No supporting evidence retrieved from the knowledge base.")
        for e in data.get("evidence", []):
            tag = f"{e['evidence_type']}, similarity {e['similarity_score']:.2f}" + (" , human-verified" if e.get("verified") else "")
            st.markdown(f"**{e['source_id']}** ({tag})")
            st.caption(e["content_snippet"])

    st.markdown("**Was this verdict right?** Your label is stored and improves future retrieval.")
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("It is authentic", key=f"auth_{data['job_id']}"):
        submit_correction(data["job_id"], "authentic")
    if c2.button("It is synthetic", key=f"synth_{data['job_id']}"):
        submit_correction(data["job_id"], "synthetic")
    st.caption(f"Job id: {data['job_id']}")


st.set_page_config(page_title="Forensics Agent", layout="wide")
st.title("Agentic AI + RAG Digital Forensics")
st.caption("Detects AI-generated text and deepfake images. Results are probabilistic, not proof.")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if not health.get("llm_configured"):
        st.warning("The API has no ANTHROPIC_API_KEY: LLM and vision-judge tools will be skipped.")
except Exception:
    st.error(f"Cannot reach the API at {API_URL}. Start it with: uvicorn api.main:app")
    st.stop()

tab_text, tab_image = st.tabs(["Text analysis", "Image analysis"])

with tab_text:
    text_input = st.text_area("Paste text to analyse (at least 20 words)", height=220)
    if st.button("Analyse text", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Running the agent pipeline..."):
                try:
                    r = requests.post(f"{API_URL}/analyze/text", json={"text": text_input}, timeout=180)
                    r.raise_for_status()
                    st.session_state["result"] = r.json()
                except Exception as exc:
                    st.error(f"Analysis failed: {_post_error(exc)}")
    if st.session_state.get("result", {}).get("modality") == "text":
        show_result(st.session_state["result"])

with tab_image:
    uploaded = st.file_uploader("Upload an image (JPEG, PNG or WebP, up to 10 MB)", type=["jpg", "jpeg", "png", "webp"])
    image_name, image_bytes, image_type = None, None, None
    if uploaded is not None:
        image_name, image_bytes, image_type = uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream"
    elif paste_image_button is not None:
        st.caption("Or copy an image (Ctrl+C, or a screenshot) and click the button to paste it.")
        pasted = paste_image_button("Paste image from clipboard", errors="raise")
        if pasted.image_data is not None:
            buf = io.BytesIO()
            pasted.image_data.convert("RGB").save(buf, format="PNG")
            image_name, image_bytes, image_type = "clipboard.png", buf.getvalue(), "image/png"
    if image_bytes is not None:
        st.image(image_bytes, width=320)
    if image_bytes is not None and st.button("Analyse image", type="primary"):
        with st.spinner("Running the image pipeline..."):
            try:
                r = requests.post(f"{API_URL}/analyze/image", timeout=300,
                                  files={"file": (image_name, image_bytes, image_type)})
                r.raise_for_status()
                st.session_state["result"] = r.json()
            except Exception as exc:
                st.error(f"Analysis failed: {_post_error(exc)}")
    if st.session_state.get("result", {}).get("modality") == "image":
        show_result(st.session_state["result"])
