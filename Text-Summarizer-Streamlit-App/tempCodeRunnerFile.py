
@st.cache_resource
def text_summary(text, maxlength=None):
    #create summary instance
    summary = Summary()
    text = (text)
    result = summary(text)