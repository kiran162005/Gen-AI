# from openai import OpenAI

# # Create client
# client = OpenAI(api_key=" ")

# # Chain of thought prompt
# prompt = """
# Solve the following problem step by Step"

# Question:
# If a train travels 60km in 1 hr , how far will it travel in 3.5 hours?

# Think through the problem step-by-step and then give the final answer.
# """

# response = client.response.create(
#     model="gpt-4.1",
#     input=prompt
# )

# print(response.output_text)


from google import genai

# Create client
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# Chain of thought prompt
prompt = """
Solve the following problem step by Step

Question:
If a train travels 60km in 1 hr , how far will it travel in 3.5 hours?

Think through the problem step-by-step and then give the final answer.
"""

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

print(response.text)