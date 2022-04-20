import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pickle import load

@st.cache(allow_output_mutation=True)
def get_trained_model():
    # load the model
    model = load_model('model.h5')
    model.summary()  # included to make it visible when model is reloaded
    # load the tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    return model, tokenizer

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = np.argmax(model.predict(encoded), axis=-1)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)
    
def get_demo(model, tokenizer):
    placeholder = st.empty()
    seed_text = placeholder.text_input('Please input the seeds word below and press ENTER', value='')
    
    if (seed_text != ''):
        # generate new text
        generated = generate_seq(model, tokenizer, 50, seed_text, 30)
            
        col1, col2 = st.columns(2)

        with col1:
            st.write('Input: ')
            st.info(seed_text)
        with col2:
            st.write('Caption: ')
            st.success(seed_text + generated)

if __name__ == "__main__":
    # load model
    model, tokenizer = get_trained_model()
    
    st.title("""Caption Generator *Bahasa* 🇮🇩""")
    st.write("""This app generates caption for social media users (such as Instagram and Twitter) in *Bahasa* 🇮🇩 using a deep learning model.""")
    
    with st.expander("ℹ️ - About this app"):     
        st.write("""
        The model employs an LSTM (Long Short-Term Memory) architecture trained by thousands of captions acquired from multiple online sources.
        The average accuracy of the trained model is 81.48.
        
        Note: *The trained model will be improved as the present generated sentence sometimes is a bit weird and cheesy*.

        """)
        
        st.metric(label="Avg. Accuracy", value="81.48", delta="0.0")
    
    get_demo(model, tokenizer)
        
