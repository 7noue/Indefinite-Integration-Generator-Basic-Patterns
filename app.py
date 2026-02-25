import streamlit as st
from engine import IntegrationEngine, ComputationResult

class ApplicationUI:
    def __init__(self):
        self.engine = IntegrationEngine()
        self._initialize_state()
        
    def _initialize_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "expr_input" not in st.session_state:
            st.session_state.expr_input = ""

    def _append_to_input(self, val: str):
        st.session_state.expr_input += val

    def render_virtual_keyboard(self):
        """Renders a clean math keyboard."""
        keys = [
            [("x²", "**2"), ("x³", "**3"), ("xⁿ", "**"), ("√", "sqrt(")],
            [("sin", "sin("), ("cos", "cos("), ("tan", "tan("), ("eˣ", "exp(")],
            [("π", "pi"), ("+", " + "), ("-", " - "), ("*", " * ")],
            [("/", " / "), ("(", "("), (")", ")"), ("Clear", "CLEAR")]
        ]
        
        for row in keys:
            cols = st.columns(4)
            for i, (label, val) in enumerate(row):
                with cols[i]:
                    if label == "Clear":
                        if st.button(label, use_container_width=True, key=f"btn_{label}"):
                            st.session_state.expr_input = ""
                    else:
                        st.button(label, on_click=self._append_to_input, args=(val,), use_container_width=True, key=f"btn_{label}")

    def render_trail(self, res: ComputationResult):
        st.divider()
        
        # 1. Given & Method (Clean top section)
        st.markdown("### Solution")
        st.info(f"$$ {res.given} $$")
        st.caption(f"**Method Used:** {res.method}")
        
        # 2. Final Answer (Prominently displayed)
        st.success(f"$$ \\int f(x)dx = {res.final_answer} $$")
        
        # 3. Steps (Tucked inside an expander for a clean look)
        with st.expander("View Step-by-Step Breakdown", expanded=False):
            for i, step in enumerate(res.steps, 1):
                st.markdown(f"**Step {i}:**")
                st.latex(step)
                
        # 4. Verification
        with st.expander("View Verification (Derivative Check)", expanded=False):
            parts = res.verification.split("\n\n")
            if len(parts) >= 2:
                st.latex(parts[0])
                if "Successful" in parts[1]:
                    st.success(parts[1])
                else:
                    st.warning(parts[1])
            else:
                st.warning(res.verification)
                
        # 5. Summary (Using native metrics)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Runtime", res.summary.get("Runtime", "N/A"))
        col2.metric("Iterations", res.summary.get("Iterations", "N/A"))
        col3.metric("Computed At", res.summary.get("Timestamp", "N/A").split(" ")[1])

    def run(self):
        st.set_page_config(page_title="Symbolic Integrator", layout="centered", page_icon="∫")
        
        # Header
        st.title("∫ Indefinite Integration")
        st.markdown("Enter a mathematical expression to compute its indefinite integral.")
        
        # Input Section
        st.text_input("Integrand f(x)", key="expr_input", placeholder="e.g., 3*x**2 or sin(x)")
        
        # Keyboard Expander
        with st.expander("⌨️ Virtual Math Keyboard"):
            self.render_virtual_keyboard()
            
        submit_button = st.button("Compute Integral", type="primary", use_container_width=True)

        # Computation Logic
        if submit_button:
            if not st.session_state.expr_input.strip():
                st.error("Please enter an integrand.")
                return
                
            with st.spinner("Integrating..."):
                current_input = st.session_state.expr_input.strip()
                result = self.engine.compute(current_input)
                
                if result.is_success:
                    self.render_trail(result)
                    
                    if not st.session_state.history or st.session_state.history[-1]["input"] != current_input:
                        st.session_state.history.append({
                            "input": current_input,
                            "result": result
                        })
                else:
                    st.error(f"Computation Failed: {result.error_message}")

if __name__ == "__main__":
    app = ApplicationUI()
    app.run()