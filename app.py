import zipfile
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from io import BytesIO
import streamlit as st

nltk.download("stopwords")
from nltk.corpus import stopwords

stp_wds = set(stopwords.words("hinglish"))
print(stp_wds)

def get_text_file(f):
    with zipfile.ZipFile(f,"r") as f:
        list_of__txt_files = [f for f in f.namelist() if f.endswith(".txt")]
        if not list_of__txt_files:
            return None
        with f.open(list_of__txt_files[0]) as f:
            return f.read().decode("UTF-8")

chat_txt = get_text_file("WhatsApp Chat with Shruti.zip")

def preprocessing(chat_txt):
    messages = []
    for line in chat_txt.split('\n'):
        match = re.match(r"(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}) - (.*?): (.*)", line)
        if match:
            date, time, sender, message = match.groups()
            messages.append({"date": date, "time": time, "sender": sender, "message": message})
    return pd.DataFrame(messages)

def filters(df):
    st.sidebar.header("Filters")
    yrs=df["new_date"].dt.year.unique()
    months=["Jan","Feb","March","April","May","June","July","Aug","Sept","Oct","Nov","Dec"]
    year = st.sidebar.selectbox("Select Year",["All"]+sorted(yrs),index=0)
    month = st.sidebar.selectbox("Select Month", ["All"] + list(months), index=0)
    st.write(f"Displaying data for {year}-{month}")
    filtered_df = df.copy()
    if year != "All":
        filtered_df = filtered_df[filtered_df['new_date'].dt.year == year]
    if month != "All":
        filtered_df = filtered_df[filtered_df["month"] == month]
    return year,month,filtered_df

def clean_dataframe(df):
    def cat(row):
        if "<Media omitted>" in row:
            return "Media"
        return "Text"
    def deleted(row):
        if row=="This message was deleted" or row=="You deleted this message":
            return "Deleted"
        return "Not Deleted"
    def month(row):
        d={1:"Jan",2:"Feb",3:"March",4:"April",5:"May",6:"June",7:"July",8:"Aug",9:"Sept",10:"Oct",11:"Nov",12:"Dec"}
        return d[row]
    def cat_time(row):
        if 6<=int(row.split(":")[0])<12:
            return "Morning"
        elif 12<=int(row.split(":")[0])<18:
            return "Afternoon"
        elif 18<=int(row.split(":")[0])<21:
            return "Evening"
        else:
            return "Night"
    def hr(row):
        return row.split(":")[0]
    new_df = df.copy()
    new_df["img or text"] = new_df["message"].apply(cat)
    new_df["message"] = [row.replace("<Media omitted>", "Media") for row in new_df["message"]]
    new_df["is_deleted"] = new_df["message"].apply(deleted)
    new_df["new_date"] = pd.to_datetime(new_df["date"])
    new_df["month"] = new_df["new_date"].dt.month.apply(month)
    new_df["Interval"] = new_df["time"].apply(cat_time)
    new_df["message length"] = new_df["message"].replace(["Media","You deleted this message","This message was deleted"],'').apply(len)
    new_df["day"] = [{0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}[row] for row in new_df["new_date"].dt.weekday]
    new_df["hour"] = new_df["time"].apply(hr)
    new_df["datetime"] = pd.to_datetime(new_df["date"]+" "+new_df["time"])
    time_gap = pd.Timedelta(hours=11)
    new_df["initiated"] = new_df["datetime"].diff() > time_gap
    return new_df

def sentiments(df):
     def clean(text):
        text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', '', text)
        if "http" in text:
            text=""
        return text.lower().strip()
     def get_emojis(text):
          emoji = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
          return emoji.findall(text)
     def gn_gm(text):
          text=''.join(text.lower().split())
          if "goodnight" in text or "goodmoring" in text or 'shubhratri' in text or 'suprabhat' in text:
               return 1
          return 0
     new_df = df.copy()
     new_df["emojis"] = new_df["message"].apply(get_emojis)
     new_df["cleaned_msg"] = new_df["message"].apply(clean)
     new_df["emoji_count"] = [len(row) for row in new_df["emojis"]]
     new_df["gn_gm"] = new_df["message"].apply(gn_gm)
     return new_df

st.title("WhatsApp Chat and Sentiment Analysis")

uploaded_file = st.file_uploader("Upload a WhatsApp Chat ZIP file", type=["zip"])
if uploaded_file is not None:
    chat_txt = get_text_file(uploaded_file)
else:
    st.warning("Please upload a WhatsApp chat ZIP file.")
    st.stop()

df = preprocessing(chat_txt)

st.write("Provided Chat:")
st.dataframe(clean_dataframe(df).head(50))
tab1, tab2, tab3 = st.tabs(["📊 General Stats", "😊 Sentiment Analysis", "💬 Word Trends"])
year,month,df = filters(clean_dataframe(df))
with tab1:
    option = st.radio("Select a Question", ["Who talks the most?","Which day of the week has the most messages?","What are the most active hours for the chat?","Who initiates the conversations more?","Most common time interval?","How many messages were deleted?","Longest Streak!","Response Time!"])
    st.write(f"Total Messages: {len(df)}")
    if option=="Who talks the most?":
            st.write(f"{df['sender'].value_counts().idxmax()}")
            st.bar_chart(df["sender"].value_counts(),horizontal=True)
            st.write(df["sender"].value_counts())
    elif option=="Which day of the week has the most messages?":
            st.write(f"{df['day'].value_counts().idxmax()}")
            st.bar_chart(df["day"].value_counts(),horizontal=True)
    elif option=="What are the most active hours for the chat?":
            hour_counts=df["hour"].value_counts().sort_index()
            heatmap_data = pd.DataFrame(hour_counts).reset_index()
            heatmap_data.columns = ["Hour", "Message Count"]
            heatmap_data = heatmap_data.pivot(index="Message Count", columns="Hour", values="Message Count")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(heatmap_data, cmap="Blues", ax=ax)
            plt.title("WhatsApp Activity Heatmap (Hourly)")
            plt.xlabel("Hour of the Day")
            plt.ylabel("Message Count")
            st.pyplot(fig)
    elif option=="Who initiates the conversations more?":
            st.write(df[df["initiated"]]["sender"].value_counts().idxmax())
            st.bar_chart(df[df["initiated"]]["sender"].value_counts(),horizontal=True)
    elif option=="Most common time interval?":
            st.write(f"{df['Interval'].value_counts().idxmax()}")
            st.bar_chart(df["Interval"].value_counts(),horizontal=True)
            st.write(df["Interval"].value_counts())
    elif option=="How many messages were deleted?":
            st.write(f"{df['is_deleted'].value_counts().idxmax()}")
            st.bar_chart(df["is_deleted"].value_counts(),horizontal=True)
            st.write(df["is_deleted"].value_counts())
    elif option=="Longest Streak!":
            st.write(f"{df['date'].value_counts().idxmax()}")
            st.bar_chart(df["date"].value_counts().head(10),horizontal=True)
            st.write(df["date"].value_counts().head(10))
    elif option=="Response Time!":
            new = df.sort_values(["new_date"])
            new["prev_sender"] = new["sender"].shift(1)
            new["prev_timestamp"] = new["new_date"].shift(1)
            new["response_time"] = (new["new_date"] - new["prev_timestamp"]).dt.total_seconds()/3600
            new.loc[new["sender"] == new["prev_sender"], "response_time"] = None
            disp = new.groupby("sender")["response_time"].mean().sort_values()
            st.write(disp.idxmin())
            st.bar_chart(disp,horizontal=True)
            st.write(disp)

with tab2:
    df = sentiments(df)
    option = st.radio("Select a Question",["Who use the most emojis?","What are the most used emojis?","Most Goodnight or Goodmoring?"])
    if option=="Who use the most emojis?":
         st.write(f"Total Messages: {len(df)}")
         st.write(f"{df.groupby('sender')['emoji_count'].sum().idxmax()}")
         st.bar_chart(df.groupby('sender')['emoji_count'].sum(),horizontal=True)
    elif option=="What are the most used emojis?":
         option = st.radio(label="For whom?",options=["All"] + list(df["sender"].unique()))
         if option=="All":
            st.write(f"Total Messages: {len(df)}")
            all_emojis = [emoji for sublist in df['emojis'] for emoji in sublist]
            emoji_counts = pd.Series(all_emojis).value_counts()
            st.write(emoji_counts.idxmax())
            st.bar_chart(emoji_counts.head(10),horizontal=True)
            st.write(emoji_counts)
         else:
            new_df = df[df["sender"]==option]
            st.write(f"Total Messages: {len(new_df)}")
            all_emojis = [emoji for sublist in new_df['emojis'] for emoji in sublist]
            emoji_counts = pd.Series(all_emojis).value_counts()
            st.write(emoji_counts.idxmax())
            st.bar_chart(emoji_counts.head(10),horizontal=True)
            st.write(emoji_counts)
    elif option=="Most Goodnight or Goodmoring?":
        st.write(f"Total Messages: {len(df)}")
        gn_gm_count = df.groupby("sender")['gn_gm'].sum()
        st.write(gn_gm_count.idxmax())
        st.bar_chart(gn_gm_count.head(10),horizontal=True)
        st.write(gn_gm_count)

with tab3:
     option = st.radio("Select a Question",["What are the most used words?","Who sends the longest messages on average?"])
     if option=="What are the most used words?":
        def remove_punc(text):
            return re.sub(r'[^\w\s]', '', text)
        words = df["message"].apply(remove_punc)
        words = " ".join(words).lower().split()
        words = [word for word in words if word not in stp_wds]
        word_counts = pd.Series(words).value_counts().head(10)
        if "media" in word_counts:
             word_counts = word_counts.drop("media")
        st.write(word_counts.idxmax())
        st.bar_chart(word_counts.head(10),horizontal=True)
        st.write(word_counts)
     elif option=="Who sends the longest messages on average?":
        st.write(f"{df.groupby('sender')['message length'].mean().idxmax()}")
        st.bar_chart(df.groupby("sender")["message length"].mean(),horizontal=True)
        st.write(df.groupby("sender")["message length"].mean())