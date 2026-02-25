import sympy as sp
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

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
        self.u = sp.Symbol('u')
        
    def _detect_u_substitution(self, expr) -> Optional[Tuple[sp.Expr, sp.Expr, sp.Expr]]:
        """Scans the expression tree for a valid basic U-Substitution pattern."""
        if not expr.is_Mul:
            return None
            
        u_candidates = []
        
        for arg in expr.args:
            if arg.is_Pow:
                if arg.base.has(self.x) and arg.base != self.x: 
                    u_candidates.append(arg.base)
                if arg.exp.has(self.x) and arg.exp != self.x: 
                    u_candidates.append(arg.exp)
            elif isinstance(arg, (sp.sin, sp.cos, sp.tan, sp.exp, sp.log)):
                inner_arg = arg.args[0]
                if inner_arg.has(self.x) and inner_arg != self.x: 
                    u_candidates.append(inner_arg)
                    
        for u_expr in u_candidates:
            du_expr = sp.diff(u_expr, self.x)
            if du_expr == 0: 
                continue
            
            ratio = sp.simplify(expr / du_expr)
            
            if not ratio.has(self.x):
                constant_multiplier = ratio
                u_integrand = sp.simplify(expr.subs(u_expr, self.u) / du_expr)
                
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
            
            u_sub_data = self._detect_u_substitution(expr)
            
            if u_sub_data:
                u_expr, du_expr, u_integrand = u_sub_data
                result.method = "Integration by Substitution"
                
                result.steps.append(rf"\text{{Let }} u = {sp.latex(u_expr)}")
                result.steps.append(rf"\text{{Differentiate }} u \text{{: }} \frac{{du}}{{dx}} = {sp.latex(du_expr)}")
                result.steps.append(rf"\text{{Isolate }} dx \text{{: }} dx = \frac{{du}}{{{sp.latex(du_expr)}}}")
                result.steps.append(rf"\text{{Substitute into original integral:}}")
                result.steps.append(rf"\int \left( {sp.latex(u_integrand)} \right) \, du")
                
                antiderivative_u = sp.integrate(u_integrand, self.u)
                result.steps.append(rf"\text{{Evaluate integral: }} {sp.latex(antiderivative_u)}")
                
                antiderivative = antiderivative_u.subs(self.u, u_expr)
                result.steps.append(rf"\text{{Substitute }} u \text{{ back: }} {sp.latex(antiderivative)}")
                result.steps.append(rf"\text{{Add the constant of integration, }} C.")
                
            else:
                result.method = "Basic Standard Patterns"
                antiderivative = sp.integrate(expr, self.x)
                
                if isinstance(antiderivative, sp.Integral):
                    raise ValueError("Requires advanced techniques beyond Basic Patterns/U-Sub, or has no closed-form solution.")
                    
                result.steps.append(rf"\text{{Evaluate using basic rules: }} {sp.latex(antiderivative)}")
                result.steps.append(rf"\text{{Add the constant of integration, }} C.")
            
            result.final_answer = rf"{sp.latex(antiderivative)} + C"
            
            # Verification Phase
            derivative = sp.diff(antiderivative, self.x)
            residual = sp.simplify(expr - derivative)
            
            verification_text = rf"\text{{Back-check by differentiating: }} \frac{{d}}{{dx}}\left[{sp.latex(antiderivative)}\right] = {sp.latex(derivative)}"
            
            if residual == 0:
                result.verification = verification_text + "\n\n**Verification Successful:** Derivative matches the integrand."
            else:
                result.verification = verification_text + "\n\n**Verification Warning:** Symbolic equivalence to 0 not trivially established."
                
            result.summary = {
                "Runtime": f"{(time.perf_counter() - start_time) * 1000:.2f} ms",
                "Iterations": "1",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            result.is_success = True
            
        except Exception as e:
            result.is_success = False
            result.error_message = f"Error: {str(e)}"
            
        return result