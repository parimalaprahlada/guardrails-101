"""
app.py -- a Streamlit UI over guardrail_pipeline.py

This file adds ZERO guardrail logic of its own. Every check you see run here
is the exact same function from guardrail_pipeline.py -- check_input_shape,
check_and_redact_pii, check_scope, check_safety, generate_answer, check_output.
This file only calls them in order and displays each GuardrailResult as it
comes back, instead of printing it to a terminal.

That's deliberate: the guardrail logic has exactly one home
(guardrail_pipeline.py). If you want to change what a check does, you edit it
there, and both the CLI (`python guardrail_pipeline.py`) and this UI
(`streamlit run app.py`) pick up the change automatically -- there's no
second copy of the logic to keep in sync.

Run it:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import guardrail_pipeline as gp

st.set_page_config(page_title="Guardrails 101", layout="centered")

st.title("Standalone Guardrails")
st.caption(
    "A cooking/recipe assistant, gated by 5 guardrail stages. "
    "Every stage below is the same function from `guardrail_pipeline.py` -- "
    "nothing here is reimplemented."
)

with st.sidebar:
    st.header("The 5 stages")
    st.markdown(
        """
1. **Input shape** — free, plain Python. Empty/too-long messages.
2. **PII redaction** — free, regex. Emails/phones/SSN-like patterns.
3. **Scope check** — 1 LLM call. Is this on-topic (cooking)?
4. **Safety check** — 1 LLM call. Is this dangerous/illegal/harmful?
5. **Output check** — free, regex. Does the *answer* leak PII?

Stages run in this order, cheapest first. Any stage can stop the
pipeline before the next one (and before the real answer is ever
generated) -- so a blocked message never reaches the expensive calls.
        """
    )
    st.divider()
    st.caption("Try the same examples from `--demo` mode, or type your own.")

message = st.text_input(
    "Your message",
    placeholder="e.g. give me a recipe for mysore pak",
)
run = st.button("Run through the pipeline", type="primary")


def _stage_box(result: "gp.GuardrailResult") -> None:
    """Render one GuardrailResult as a pass/fail box. Pure display, no logic."""
    if result.passed:
        st.success(f"**{result.stage}** — PASS\n\n{result.detail}")
    else:
        st.error(f"**{result.stage}** — FAIL\n\n{result.detail}")


if run:
    if not message:
        st.warning("Type a message first.")
        st.stop()

    # This block mirrors run_pipeline() in guardrail_pipeline.py line for
    # line -- same functions, same order, same early-exit logic. Only the
    # *display* (st.success/st.error/st.stop instead of print/return) differs.
    st.divider()

    result = gp.check_input_shape(message)
    _stage_box(result)
    if not result.passed:
        st.stop()

    result = gp.check_and_redact_pii(message)
    _stage_box(result)
    message = result.cleaned_text or message  # continue with the redacted version

    with st.spinner("Checking scope..."):
        result = gp.check_scope(message)
    _stage_box(result)
    if not result.passed:
        st.stop()

    with st.spinner("Checking safety..."):
        result = gp.check_safety(message)
    _stage_box(result)
    if not result.passed:
        st.stop()

    with st.spinner("Generating the answer..."):
        answer = gp.generate_answer(message)
    st.subheader("Assistant answer")
    st.write(answer)

    result = gp.check_output(answer)
    _stage_box(result)
    if not result.passed:
        st.stop()

    st.balloons()
    st.success("✅ Answer Delivered to the user.")
