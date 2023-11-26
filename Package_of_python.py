import astor
import streamlit as st
import openai
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Python AST (Abstract Syntax Trees) module for code analysis

# Set your OpenAI API key
openai.api_key = "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"

# Function to generate text using the gpt-3.5-turbo engine
def generate_text(task_description, code_input, keywords_input):
    # Split keywords by comma and create sections for each keyword
    keywords = keywords_input.split(',')
    sections = []

    # Add task description and code
    sections.append(f"Task Description: {task_description}\n Improvements Code:\n{code_input}")

    # Add sections for each keyword
    for keyword in keywords:
        sections.append(f"Improvements for {keyword.strip()}:")

    # Combine sections into a prompt
    prompt = "\n\n".join(sections)

    # Define the parameters for the chat completion
    params = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    }

    # Make the API call
    response = openai.ChatCompletion.create(**params)

    # Extract and return the generated text
    return response['choices'][0]['message']['content']


def check_main_concepts(code):
    required_functions = ["preprocess_data", "split_data", "train_model", "test_model"]
    missing_concepts = []

    for func_name in required_functions:
        if func_name not in code:
            missing_concepts.append(func_name)

    return not missing_concepts, missing_concepts


# Function to check if the code has been modularized
def check_modularization(code):
    try:
        # Parse the code to an AST
        tree = ast.parse(code)

        # Initialize a counter for function definitions
        function_count = 0

        # Iterate through the nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1

        # Check if there are multiple functions
        return function_count > 1

    except SyntaxError:
        # Handle syntax errors in code
        return False


# Streamlit app
def main():
    st.title("Code Review Assistant")

    # Input fields
    task_description = st.text_area("Task Description", "")
    code_input = st.text_area("Code Input", "")
    keywords_input = st.text_input("Keywords", "Performance, Readability, Speed, Maintainability")

    if st.button("Generate Code Review"):
        # Call OpenAI's ChatGPT 3.5 Turbo to generate code review
        result = generate_code_review(task_description, code_input, keywords_input)

        # Display results
        st.subheader("Code Review Result:")
        st.write(result['output'])

        # Display improved code
        st.subheader("Improved Code:")
        st.code(result['improved_code'], language="python")

        # Display similarity score
        st.subheader("Similarity Score:")
        st.write(str(result['similarity_score'])+"%")

        # Check if the code meets the task requirements
        if float(result['similarity_score']) > 50:
            # Check if the code contains main concepts
            contains_main_concepts, missing_concepts = check_main_concepts(result['improved_code'])

            # Check if the code has been modularized
            is_modularized = check_modularization(result['improved_code'])

            # Check if the code is AI-generated
            is_ai_generated = check_task_completion(result['improved_code'], task_description)

            if contains_main_concepts and is_modularized and is_ai_generated:
                st.success("Task Completed Successfully!")
            else:
                missing_concepts_text = ', '.join(missing_concepts)
                st.warning(f"Task Not Completed. The generated code is missing: {missing_concepts_text}")

                if not is_modularized:
                    st.warning("The generated code is missing modularization.")
                else:
                    st.warning("The generated code is modularized.")
                if not is_ai_generated:
                    st.warning("The generated code is not AI-generated.")
                else:
                    st.warning("The generated code is AI-generated.")
        else:
            st.warning("Task Not Completed. The generated code does not meet the requirements.")


# Function to generate code review using ChatGPT 3 Turbo
def generate_code_review(task_description, code_input, keywords_input):
    try:
        # Use the generate_text function with gpt-3.5-turbo engine to get suggestions
        suggestions_text = generate_text(task_description, code_input, keywords_input)

        # Parse the suggestions from the generated text
        suggestions = extract_suggestions(suggestions_text)

        # Generate improved code
        improved_code = generate_improved_code(code_input, suggestions)

        # Optimize the user's input code for performance
        optimized_code = optimize_code(improved_code)

        # Calculate similarity score between the optimized code and task description
        similarity_score = calculate_similarity_score(optimized_code, task_description)

        return {'output': suggestions_text, 'suggestions': suggestions, 'improved_code': improved_code,
                'similarity_score': similarity_score}

    except SyntaxError:
        # Handle syntax errors in the provided code
        st.error("The code has a syntax error. Please check and fix the code.")
        return {'output': '', 'suggestions': {}, 'improved_code': '', 'similarity_score': '0%'}


# Function to calculate similarity score using cosine similarity
def calculate_similarity_score(code, task_description):
    # Use TF-IDF to vectorize the code and task description
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([code, task_description])

    # Calculate cosine similarity between the vectors
    similarity_matrix = cosine_similarity(vectors)
    similarity_score = similarity_matrix[0, 1]

    # Round the similarity score to two decimal places
    similarity_percentage = round(similarity_score * 100, 2)

    # Convert to string and extract the first two characters
    similarity_percentage_str = str(similarity_percentage)[:2]

    return similarity_percentage_str


# Function to extract suggestions from ChatGPT response
def extract_suggestions(output):
    suggestions = {}
    lines = output.split('\n')

    for line in lines:
        if ':' in line:
            point, suggestion = map(str.strip, line.split(':', 1))
            suggestions[point] = suggestion

    return suggestions


# Function to generate improved code based on suggestions
def generate_improved_code(original_code, suggestions):
    # Split the original code into lines
    original_lines = original_code.split('\n')

    # Apply the suggested modifications
    for point, suggestion in suggestions.items():
        # Split the suggestion into lines and apply each line modification
        suggestion_lines = suggestion.split('\n')

        for line_modification in suggestion_lines:
            if line_modification.startswith('+'):
                # Extract the line number where the modification is suggested
                line_number = int(point)
                
                # Add a comment to explain the added line
                comment = f"# Suggested improvement: {line_modification[1:]}"
                
                # Insert the comment above the added line
                original_lines.insert(line_number, comment)

                # Insert the added line below the comment
                original_lines.insert(line_number + 1, line_modification[1:])
            elif line_modification.startswith('-'):
                # Optionally, you can choose to ignore or handle deletions differently
                pass
            else:
                # Optionally, you can choose to handle other modifications differently
                pass

    # Combine the modified lines back into code
    improved_code = '\n'.join(original_lines)

    # Optimize the code
    optimized_code = optimize_code(improved_code)

    return optimized_code


# Function to optimize code
def optimize_code(code):
    # Parse the code to an AST
    tree = ast.parse(code)

    # Example: Removing redundant nodes (you can add more optimization techniques)
    optimized_tree = remove_redundant_nodes(tree)

    # Convert the optimized AST back to code
    optimized_code = astor.to_source(optimized_tree)

    return optimized_code


# Function to remove redundant nodes from the AST (example optimization)
def remove_redundant_nodes(tree):
    # Example: Remove empty print statements
    new_body = [node for node in tree.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Str) and node.value.s == "")]
    tree.body = new_body
    return tree


# Function to check if the code meets the requirements of the task
def check_task_completion(generated_code, task_description):
    # Extract functions from task description and generated code
    task_functions = extract_functions(task_description)
    code_functions = extract_functions(generated_code)

    # Check if all task functions are present in the code
    contains_all_functions = all(task_function in code_functions for task_function in task_functions)

    # Check if the code is AI-generated (check for patterns related to GPT-3.5 Turbo)
    is_ai_generated = "gpt-3.5-turbo" in generated_code.lower() or "openai" in generated_code.lower()

    return contains_all_functions and is_ai_generated


# Function to extract functions from code using AST
def extract_functions(code):
    functions = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.add(node.name)
    except SyntaxError:
        # Handle syntax errors in code
        pass
    return functions


if __name__ == "__main__":
    main()
