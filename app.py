import streamlit as st
import sympy as sp
import time
from datetime import datetime
import importlib.metadata
from typing import Dict, Any, Optional, Tuple

# ==========================================
# Core Domain Classes (OOP)
# ==========================================

class ComputationResult:
    """Data Transfer Object for storing computation trail details."""
    def __init__(self):
        self.given: str = ""
        self.method: str = ""
        self.steps: list[str] = []
        self.final_answer: str = ""
        self.verification: str = ""
        self.summary: Dict[str, Any] = {}
        self.is_success: bool = False
        self.error_message: str = ""

class IntegrationEngine:
    """Handles the symbolic mathematics and verification logic."""
    
    def __init__(self):
        self.x = sp.Symbol('x')
        
    def compute(self, expression_str: str) -> ComputationResult:
        result = ComputationResult()
        start_time = time.perf_counter()
        
        try:
            # 1. Parse Input (GIVEN)
            expr = sp.sympify(expression_str)
            result.given = f"Evaluate the indefinite integral of: {sp.latex(expr)} with respect to x."
            
            # 2. Method Definition (METHOD)
            result.method = "Symbolic Indefinite Integration (Basic Patterns via SymPy)"
            
            # 3. Step Generation (STEPS)
            # For basic patterns, we outline the symbolic workflow
            result.steps.append(f"Identify the integrand: f(x) = {sp.latex(expr)}")
            result.steps.append(f"Set up the integral: \\int ({sp.latex(expr)}) \\, dx")
            
            if expr.is_Add:
                result.steps.append("Apply the Sum Rule (Linearity): The integral of a sum is the sum of the integrals.")
            elif expr.is_Mul and expr.has(sp.Number):
                result.steps.append("Apply the Constant Multiple Rule: Move constants outside the integral.")
                
            # Perform integration
            antiderivative = sp.integrate(expr, self.x)
            result.steps.append(f"Evaluate the antiderivative using standard integration rules: {sp.latex(antiderivative)}")
            result.steps.append("Add the constant of integration, C.")
            
            # 4. Final Answer (FINAL ANSWER)
            result.final_answer = f"{sp.latex(antiderivative)} + C"
            
            # 5. Verification (VERIFICATION)
            derivative = sp.diff(antiderivative, self.x)
            residual = sp.simplify(expr - derivative)
            
            verification_text = (
                f"Back-check by differentiating the result: \\frac{{d}}{{dx}}[{sp.latex(antiderivative)}] = {sp.latex(derivative)}\n\n"
            )
            if residual == 0:
                verification_text += "Verification Successful: The derivative matches the original integrand (Residual = 0)."
            else:
                verification_text += "Verification Warning: Symbolic equivalence could not be trivially established to 0."
                
            result.verification = verification_text
            
            # 6. Summary (SUMMARY)
            end_time = time.perf_counter()
            result.summary = {
                "Runtime": f"{(end_time - start_time) * 1000:.2f} ms",
                "Iterations": "1 (Symbolic Closed-Form Pass)",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "SymPy Version": importlib.metadata.version("sympy"),
                "Streamlit Version": importlib.metadata.version("streamlit")
            }
            
            result.is_success = True
            
        except (sp.SympifyError, TypeError, ValueError) as e:
            result.is_success = False
            result.error_message = f"Syntax Error in Expression: {str(e)}. Please use standard Python math notation (e.g., 'x**2' instead of 'x^2', 'sin(x)', etc.)."
        except Exception as e:
            result.is_success = False
            result.error_message = f"An unexpected error occurred: {str(e)}"
            
        return result

# ==========================================
# UI & Frontend Logic (Streamlit)
# ==========================================

class ApplicationUI:
    """Manages Streamlit UI rendering and state."""
    
    def __init__(self):
        self.engine = IntegrationEngine()
        self._initialize_state()
        
    def _initialize_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
            
    def render_sidebar(self):
        st.sidebar.title("Computation History")
        if not st.session_state.history:
            st.sidebar.info("No computations yet.")
            return
            
        if st.sidebar.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
            
        st.sidebar.markdown("---")
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.sidebar.expander(f"Run {len(st.session_state.history) - idx}: {item['input']}"):
                st.latex(item['result'].final_answer)

    def render_trail(self, res: ComputationResult):
        """Renders the standard trail format inside Streamlit."""
        st.markdown("### Solution Trail")
        
        # GIVEN
        st.markdown("**1. GIVEN**")
        st.info(f"Problem Statement & Inputs: \n\n $$ {res.given} $$")
        
        # METHOD
        st.markdown("**2. METHOD**")
        st.markdown(f"> {res.method}")
        
        # STEPS
        st.markdown("**3. STEPS**")
        for i, step in enumerate(res.steps, 1):
            st.markdown(f"**Step {i}:**")
            st.latex(step)
            
        # FINAL ANSWER
        st.markdown("**4. FINAL ANSWER**")
        st.success(f"Result:\n\n $$ \\int f(x)dx = {res.final_answer} $$")
        
        # VERIFICATION
        st.markdown("**5. VERIFICATION**")
        st.warning(res.verification)
        
        # SUMMARY
        st.markdown("**6. SUMMARY**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- **Runtime:** {res.summary.get('Runtime')}")
            st.markdown(f"- **Iterations:** {res.summary.get('Iterations')}")
            st.markdown(f"- **Timestamp:** {res.summary.get('Timestamp')}")
        with col2:
            st.markdown(f"- **SymPy Ver:** {res.summary.get('SymPy Version')}")
            st.markdown(f"- **Streamlit Ver:** {res.summary.get('Streamlit Version')}")

    def run(self):
        st.set_page_config(page_title="Symbolic Integrator", page_icon="∫", layout="wide")
        st.title("∫ Symbolic Indefinite Integration Generator")
        st.markdown("Enter a mathematical expression in terms of `x` to compute its indefinite integral.")
        
        self.render_sidebar()
        
        # Input Section
        with st.form("compute_form"):
            col1, col2 = st.columns([4, 1])
            with col1:
                expr_input = st.text_input("Integrand f(x)", value="3*x**2 + 2*x + sin(x)", placeholder="e.g., 3*x**2 + sin(x)")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True) # Alignment fix
                submit_button = st.form_submit_button("Compute", type="primary", use_container_width=True)
                
        # Computation Trigger
        if submit_button:
            if not expr_input.strip():
                st.error("Please enter a valid mathematical expression.")
                return
                
            with st.spinner("Computing symbolic integration and verifying..."):
                result = self.engine.compute(expr_input)
                
                if result.is_success:
                    self.render_trail(result)
                    # Save to history
                    st.session_state.history.append({
                        "input": expr_input,
                        "result": result
                    })
                else:
                    st.error("Computation Failed")
                    st.error(result.error_message)

if __name__ == "__main__":
    app = ApplicationUI()
    app.run()