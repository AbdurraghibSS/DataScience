import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
import os
import speech_recognition as sr
from streamlit_bokeh_events import streamlit_bokeh_events

st.title("Text to Speech with Bokeh and Streamlit")




st.header("**_ ### Speech to Text With Streamlit Bokeh Events ### _**")
st.text("Click 'Speak' button below to talk. Must connect to Internet")
stt_button = Button(label="Speak", width=100)

stt_button.js_on_event("button_click", CustomJS(code="""
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
 
    recognition.onresult = function (e) {
        var value = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
            }
        }
        if ( value != "") {
            document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
        }
    }
    recognition.start();
    """))

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)


if result:
    if "GET_TEXT" in result:
        st.write(result.get("GET_TEXT"))


st.header("**_### Text to Speech With Streamlit Bokeh Events ###_**")

text = st.text_input("Type what you want to say on input box below and Click 'Listen' button to listen")

tts_button = Button(label="Listen", width=100)

tts_button.js_on_event("button_click", CustomJS(code=f"""
    var u = new SpeechSynthesisUtterance();
    u.text = "{text}";
    u.lang = 'en-US';

    speechSynthesis.speak(u);
    """))

st.bokeh_chart(tts_button)

st.header("**_### Speech to Text With Google Speech Recognition from an Audio File ###_**")
uploaded_file = st.file_uploader("Click 'Browse File' to Upload Your Audio File. Accepted Audio File Only .wav", type=['wav'])
st.audio(uploaded_file)

if uploaded_file is not None:
        # Initialize recognizer class (for recognizing the speech)
 r = sr.Recognizer()

        # Reading Audio file as source
        # listening the audio file and store in audio_text variable

 with sr.AudioFile(uploaded_file) as source:
     
  audio_text = r.listen(source)
    
        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
  try:
        
        # using google speech recognition
   text = r.recognize_google(audio_text)
   st.write('Converting audio transcripts into text ...')
   st.write(text)
     
  except:
   st.write('Sorry.. run again...')






