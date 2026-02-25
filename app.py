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
    def __init__(self):
        self.x = sp.Symbol('x')
        self.u = sp.Symbol('u') # Add a symbol for U-Substitution
        
    def _detect_u_substitution(self, expr) -> Optional[Tuple[sp.Expr, sp.Expr, sp.Expr]]:
        """
        Scans the expression tree for a valid basic U-Substitution pattern.
        Returns a tuple of (u_expr, du_expr, integrand_in_terms_of_u) if found.
        """
        if not expr.is_Mul:
            return None
            
        u_candidates = []
        
        # 1. Identify potential 'u' candidates by looking inside composite functions
        for arg in expr.args:
            if arg.is_Pow:
                # Check base (e.g., (x^2 + 1)^3 -> u = x^2 + 1)
                if arg.base.has(self.x) and arg.base != self.x: 
                    u_candidates.append(arg.base)
                # Check exponent (e.g., e^(x^2) -> u = x^2)
                if arg.exp.has(self.x) and arg.exp != self.x: 
                    u_candidates.append(arg.exp)
            elif isinstance(arg, (sp.sin, sp.cos, sp.tan, sp.exp, sp.log)):
                # Check function arguments (e.g., sin(x^3) -> u = x^3)
                inner_arg = arg.args[0]
                if inner_arg.has(self.x) and inner_arg != self.x: 
                    u_candidates.append(inner_arg)
                    
        # 2. Test candidates to see if their derivative exists in the expression
        for u_expr in u_candidates:
            du_expr = sp.diff(u_expr, self.x)
            if du_expr == 0: 
                continue
            
            # Divide original expression by the derivative of our candidate
            ratio = sp.simplify(expr / du_expr)
            
            # For basic u-sub, if the remaining ratio has no 'x' outside of our 'u', it's a match.
            # The simplest case: the ratio is just a constant (e.g., x * sin(x^2) / 2x = 1/2 * sin(x^2))
            if not ratio.has(self.x):
                # Map the expression strictly to the new 'u' variable
                # We extract the constant, and apply the function to 'u'
                constant_multiplier = ratio
                
                # Reconstruct the u-integrand
                # This is a basic heuristic for powers and trig functions
                u_integrand = sp.simplify(expr.subs(u_expr, self.u) / du_expr)
                
                # Ensure no 'x' was left behind in the transformation
                if not u_integrand.has(self.x):
                    return u_expr, du_expr, u_integrand
                    
        return None

    def compute(self, integrand_str: str) -> ComputationResult:
        result = ComputationResult()
        start_time = time.perf_counter()
        
        try:
            expr = sp.sympify(integrand_str)
            result.given = rf"\int \left( {sp.latex(expr)} \right) \, dx"
            
            result.steps.append(rf"\text{{Identify the integrand: }} f(x) = {sp.latex(expr)}")
            
            # Check for U-Substitution first
            u_sub_data = self._detect_u_substitution(expr)
            
            if u_sub_data:
                u_expr, du_expr, u_integrand = u_sub_data
                result.method = "Symbolic Indefinite Integration (Integration by Substitution)"
                
                # Generate specific U-Sub steps using LaTeX
                result.steps.append(rf"\text{{Let }} u = {sp.latex(u_expr)}")
                result.steps.append(rf"\text{{Then differentiate }} u \text{{ with respect to }} x \text{{: }} \frac{{du}}{{dx}} = {sp.latex(du_expr)}")
                result.steps.append(rf"\text{{Rearrange to isolate }} dx \text{{: }} dx = \frac{{du}}{{{sp.latex(du_expr)}}}")
                result.steps.append(rf"\text{{Substitute }} u \text{{ and }} dx \text{{ back into the original integral:}}")
                result.steps.append(rf"\int \left( {sp.latex(u_integrand)} \right) \, du")
                
                # Integrate with respect to u
                antiderivative_u = sp.integrate(u_integrand, self.u)
                result.steps.append(rf"\text{{Evaluate the integral with respect to }} u \text{{: }} {sp.latex(antiderivative_u)}")
                
                # Back-substitute x
                antiderivative = antiderivative_u.subs(self.u, u_expr)
                result.steps.append(rf"\text{{Substitute }} {sp.latex(u_expr)} \text{{ back in for }} u \text{{: }} {sp.latex(antiderivative)}")
                result.steps.append(rf"\text{{Add the constant of integration, }} C.")
                
            else:
                result.method = "Symbolic Indefinite Integration (Basic Standard Patterns)"
                antiderivative = sp.integrate(expr, self.x)
                
                if isinstance(antiderivative, sp.Integral):
                    raise ValueError("This integral requires advanced techniques beyond Basic Patterns and U-Substitution, or has no closed-form solution.")
                    
                result.steps.append(rf"\text{{Evaluate using basic standard rules: }} {sp.latex(antiderivative)}")
                result.steps.append(rf"\text{{Add the constant of integration, }} C.")
            
            result.final_answer = rf"{sp.latex(antiderivative)} + C"
            
            # Verification Phase
            derivative = sp.diff(antiderivative, self.x)
            residual = sp.simplify(expr - derivative)
            
            verification_text = rf"\text{{Back-check by differentiating: }} \frac{{d}}{{dx}}\left[{sp.latex(antiderivative)}\right] = {sp.latex(derivative)}"
            
            if residual == 0:
                result.verification = verification_text + "\n\n**Verification Successful:** Derivative matches the integrand (Residual = 0)."
            else:
                result.verification = verification_text + "\n\n**Verification Warning:** Symbolic equivalence to 0 not trivially established."
                
            result.summary = {
                "Runtime": f"{(time.perf_counter() - start_time) * 1000:.2f} ms",
                "Iterations": "1 (Symbolic Pass with Substitution Check)",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            result.is_success = True
            
        except Exception as e:
            result.is_success = False
            result.error_message = f"Error: {str(e)}"
            
        return result

# ==========================================
# UI & Frontend Logic (Streamlit)
# ==========================================

class ApplicationUI:
    def __init__(self):
        self.engine = IntegrationEngine()
        self._initialize_state()
        
    def _initialize_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "expr_input" not in st.session_state:
            st.session_state.expr_input = ""
        if "show_keyboard" not in st.session_state:
            st.session_state.show_keyboard = False

    def _append_to_input(self, val: str):
        """Callback to add symbols from keyboard to input field."""
        st.session_state.expr_input += val

    def render_virtual_keyboard(self):
        """Renders a toggleable math keyboard."""
        st.write("### Math Keyboard")
        
        # Define layout
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
        st.markdown("---")
        st.markdown("### 🧩 Solution Trail")
        
        # 1. GIVEN
        st.markdown("**1. GIVEN:**")
        st.info(f"$$ {res.given} $$")
        
        # 2. METHOD
        st.markdown("**2. METHOD:**")
        st.markdown(f"> {res.method}")
        
        # 3. STEPS
        st.markdown("**3. STEPS:**")
        # Use a container to keep steps visually grouped
        with st.container():
            for i, step in enumerate(res.steps, 1):
                st.markdown(f"**Step {i}:**")
                # st.latex automatically wraps the string in the correct math delimiters
                st.latex(step) 
            
        # 4. FINAL ANSWER
        st.markdown("**4. FINAL ANSWER:**")
        st.success(f"$$ \\int f(x)dx = {res.final_answer} $$")
        
        # 5. VERIFICATION
        st.markdown("**5. VERIFICATION:**")
        # Split the verification string to separate the LaTeX math from the Markdown text
        parts = res.verification.split("\n\n")
        
        if len(parts) >= 2:
            latex_part = parts[0]
            markdown_part = parts[1]
            
            # Render the derivative math
            st.latex(latex_part)
            
            # Render the success/warning message with proper UI colors
            if "Successful" in markdown_part:
                st.success(markdown_part)
            else:
                st.warning(markdown_part)
        else:
            # Fallback if the string wasn't split properly
            st.warning(res.verification)
        
        # 6. SUMMARY
        st.markdown("**6. SUMMARY:**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"**Runtime:** {res.summary.get('Runtime', 'N/A')}")
            st.caption(f"**Iterations:** {res.summary.get('Iterations', 'N/A')}")
        with col2:
            st.caption(f"**Timestamp:** {res.summary.get('Timestamp', 'N/A')}")

    def run(self):
        st.set_page_config(page_title="Symbolic Integrator", layout="centered")
        st.title("∫ Indefinite Integration Generator")
        
        # Keyboard Toggle
        if st.toggle("⌨️ Show Virtual Keyboard", value=st.session_state.show_keyboard):
            self.render_virtual_keyboard()
            st.markdown("---")

        # Visual Input Bar (Framed with Integral signs)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.markdown("<h2 style='text-align: right;'>∫</h2>", unsafe_allow_html=True)
        with col2:
            st.text_input("Integrand", key="expr_input", label_visibility="collapsed", placeholder="Enter expression, e.g., 3*x**2")
        with col3:
            st.markdown("<h2>dx</h2>", unsafe_allow_html=True)
            
        submit_button = st.button("Compute", type="primary", use_container_width=True)

        if submit_button:
            if not st.session_state.expr_input.strip():
                st.error("Please enter an integrand.")
                return
                
            with st.spinner("Integrating..."):
                current_input = st.session_state.expr_input.strip()
                result = self.engine.compute(current_input)
                
                if result.is_success:
                    self.render_trail(result)
                    
                    # --- NEW LOGIC: Prevent duplicate consecutive history entries ---
                    # Only append if history is empty OR the last input is different
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