import streamlit as st

def show_result(prediction):
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])

        with col1:
            st.metric(
                label="Predicted Role Level",
                value=prediction["label"]
            )

        with col2:
            st.info(
                f"""
                **Confidence:** {prediction['confidence']:.2%}

                **Interpretation:**
                The model detected leadership-related language patterns
                consistent with this role level.
                """
            )
