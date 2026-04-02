from transformers import pipeline

def summarize_text(text, max_length=150, min_length=50):
    # Load the summarization pipeline with a pre-trained model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Generate summarized text
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    
    return summary[0]['summary_text']


if __name__ == "__main__":
    # Example long passage
    passage = """
    Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data, 
    adapt to new inputs, and perform human-like tasks. AI applications include natural language processing, 
    computer vision, and robotics. Companies are increasingly using AI to improve efficiency, reduce costs, 
    and create innovative products. However, concerns around ethics, bias, and job displacement continue 
    to be topics of discussion as AI technology advances.
    """
    
    summary = summarize_text(passage)
    
    print("Summarized Text:")
    print(summary)