import os
import json
import numpy as np

try:
    from google import genai
except ImportError:
    genai = None

# Load Gemini API key from environment to prevent GitHub secret leaks
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def generate_solve_function(examples: list) -> str:
    """
    Passes the input/output grids to the Gemini LLM and asks for a Python function
    that maps the input to the output.
    """
    if not genai or not GEMINI_API_KEY:
        return ""
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = "You are an expert Python programmer solving the ARC Challenge.\n"
        prompt += "Write a Python function named `solve(input_grid)` that takes a 2D numpy array and returns the transformed 2D numpy array.\n"
        prompt += "You may use numpy (imported as np).\n\n"
        prompt += "Here are the training examples demonstrating the hidden rule:\n"
        
        for i, ex in enumerate(examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input:\n{np.array(ex['input']).tolist()}\n"
            prompt += f"Output:\n{np.array(ex['output']).tolist()}\n\n"
            
        prompt += "Think about the geometric, topological, or color mapping rule. Then write the code.\n"
        prompt += "Return ONLY valid Python code starting with `def solve(input_grid):`. Do not include markdown blocks like ```python."
        
        # Use the newer google-genai client formatting
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        code = response.text.strip()
        # Clean up markdown if the LLM ignores instructions
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""
