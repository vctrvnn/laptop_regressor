import streamlit as st
import pickle
import numpy as np

#import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Predictor")

#brand
company = st.selectbox('Brand', df['Company'].unique())

#type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

#ram
ram = st.selectbox('RAM', df['Ram'].unique())

#Weight
weight = st.number_input('Weight')

#Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

#IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

#screen size
screen_size = st.number_input('Screen size')

#screen resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU Brand', df['cpu_brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU Brand', df['gpu_brand'].unique())

os = st.selectbox('OpSys', df['os'].unique())

if st.button('Predict Price'):
    #query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    
    x, y = map(int, resolution.split('x'))
    ppi = (x ** 2 + y ** 2) ** 0.5 / screen_size
    
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.title(int(np.exp(pipe.predict(query))[0]))

