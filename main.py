from bs4 import BeautifulSoup
import re
import requests
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the bart tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Define a function to summarize the text using the Bart model
def summarize(text, num_sentences, max_length=1024):
    # Use the tqdm library to create a progress bar

    # Encode the text using the Bart tokenizer
    input_ids = tokenizer.encode(text, return_tensors='pt')
    # Truncate the input text to the specified maximum length
    input_ids = input_ids[:, :max_length]
    # Generate summary using the Bart model
    summary_ids = model.generate(input_ids,
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=max_length,
                                 min_length=num_sentences * 2,
                                 early_stopping=True)

    # Decode the summary ids to get the summary text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Return the summary text
    return summary_text


# Specify the URL of the Wikipedia page
url = "https://simple.wikipedia.org/wiki/Common_Era"

# Send a request to the URL and store the response
response = requests.get(url)

# Parse the HTML content of the response
soup = BeautifulSoup(response.content, "html.parser")

# Extract the main content element
main_content = soup.find("div", {"id": "mw-content-text"})

# Extract the text from the main content element
main_text = main_content.text

# Use regex to remove the "f=ma" string
main_text = re.sub(r"displaystyle", "", main_text)
# Use regex to remove multiple new lines and replace them with a single space
main_text = re.sub(r"\n+", " ", main_text)

# # Create a PlaintextParser object to parse the main_text
# parser = PlaintextParser.from_string(main_text, Tokenizer("english"))

# Use the summarize function to generate a summary of the text
summary = summarize(main_text, 3)
print(summary)
