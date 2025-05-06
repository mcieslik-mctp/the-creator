import argparse
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part

PROJECT_ID = "mctp00004"  # Your Google Cloud project ID
LOCATION = "us-central1"    # Your Google Cloud project location

def generate_script_from_instructions(instruction_text):
    """
    Sends instructions to Vertex AI and gets a Python script.
    """
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.5-pro-preview-03-25") # Using user suggested preview model

    prompt = f"""Generate a Python script based on the following instructions.
The script should be complete and executable.
Only output the Python code itself, without any surrounding text or explanations.

Instructions:
{instruction_text}
"""

    try:
        response = model.generate_content(
            [Part.from_text(prompt)],
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.4, # Adjust for creativity vs. determinism
                "top_p": 1
            }
        )
        if response.candidates and response.candidates[0].content.parts:
            generated_script = response.candidates[0].content.parts[0].text
            # Clean up potential markdown code block formatting
            if generated_script.startswith("```python"):
                generated_script = generated_script[len("```python"):]
            if generated_script.startswith("```"):
                generated_script = generated_script[len("```"):]
            if generated_script.endswith("```"):
                generated_script = generated_script[:-len("```")]
            return generated_script.strip()
        else:
            return "# Error: Could not generate script. No content in response."
    except Exception as e:
        return f"# Error generating script: {e}"

def main():
    parser = argparse.ArgumentParser(description="Generates a Python script using Vertex AI based on instructions from a TXT file and saves it to an output file.")
    parser.add_argument("instruction_file", type=str, help="Path to the TXT file containing instructions.")
    parser.add_argument("output_file", type=str, help="Path to the output Python file where the generated script will be saved.")

    args = parser.parse_args()

    instruction_file_path = args.instruction_file
    output_file_path = args.output_file

    if not os.path.exists(instruction_file_path):
        print(f"Error: Instruction file not found at {instruction_file_path}")
        return

    try:
        with open(instruction_file_path, 'r') as f:
            instructions = f.read()
    except Exception as e:
        print(f"Error reading instruction file: {e}")
        return

    if not instructions.strip():
        print("Error: Instruction file is empty.")
        return

    print(f"Generating Python script from '{instruction_file_path}'...")
    python_script = generate_script_from_instructions(instructions)

    if python_script.startswith("# Error"):
        print(python_script) # Print error from generation
        return

    try:
        with open(output_file_path, 'w') as f:
            f.write(python_script)
        print(f"\nSuccessfully generated and saved Python script to '{output_file_path}'")
    except Exception as e:
        print(f"Error writing generated script to file '{output_file_path}': {e}")

if __name__ == "__main__":
    main()
