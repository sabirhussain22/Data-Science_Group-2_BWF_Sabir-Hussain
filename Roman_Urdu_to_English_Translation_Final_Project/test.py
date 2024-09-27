import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the saved model and tokenizer from the Kaggle output directory
model_dir = 'D:/Downloads/Translation/fine_tuned_model'  # Adjust this path as necessary
tokenizer = MarianTokenizer.from_pretrained(model_dir)
model = MarianMTModel.from_pretrained(model_dir)

# Function to translate text using the loaded model and tokenizer
def translate_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    # Generate translation using the model
    translated_tokens = model.generate(**inputs)
    # Decode the translated tokens
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Streamlit interface
st.title('Roman Urdu to English Translator')

# Text input for Roman Urdu text
roman_text = st.text_area('Enter Roman Urdu text here:', '')

# Button to perform translation
if st.button('Translate'):
    if roman_text:
        with st.spinner('Translating...'):
            translation = translate_text(roman_text)
        st.success('Translation completed!')
        st.write('**Roman Urdu:**', roman_text)
        st.write('**English Translation:**', translation)
    else:
        st.error('Please enter some text to translate.')
