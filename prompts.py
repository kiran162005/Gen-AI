# Python Program: Demonstration of Prompt Engineering Techniques
# Techniques Covered: Zero-shot, One-shot, Few-shot Prompting
# Application: Text classification using Gemini LLM

from google import genai

# Create client using API key
client = genai.Client(api_key="API_Key")


# Function for Zero-Shot Prompting
def zero_shot_prompt():
    print("\nZERO SHOT PROMPTING")

    prompt = "Classify the sentiment of the following sentence as Positive or Negative: 'The product quality is amazing and I love it.'"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    print("Prompt:", prompt)
    print("Response:", response.text)


# Function for One-Shot Prompting
def one_shot_prompt():
    print("\nONE SHOT PROMPTING")

    prompt = """
Example:
Sentence: I love this phone
Sentiment: Positive

Now classify:
Sentence: This laptop is very slow
Sentiment:
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    print("Prompt:", prompt)
    print("Response:", response.text)


# Function for Few-Shot Prompting
def few_shot_prompt():
    print("\nFEW SHOT PROMPTING")

    prompt = """
Sentence: I love this product
Sentiment: Positive

Sentence: This service is terrible
Sentiment: Negative

Sentence: The experience was wonderful
Sentiment: Positive

Now classify:
Sentence: The food was awful
Sentiment:
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    print("Prompt:", prompt)
    print("Response:", response.text)


# Main Program
def main():
    print("PROMPT ENGINEERING APPLICATION USING GEMINI LLM")

    zero_shot_prompt()
    one_shot_prompt()
    few_shot_prompt()


# Execute program
main()