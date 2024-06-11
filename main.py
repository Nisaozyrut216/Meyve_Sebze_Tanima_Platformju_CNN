from altair import themes
import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    print(model.summary())
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    print(predictions.shape)
    print(np.argmax(predictions))
    return np.argmax(predictions) #return index of max element

#Yan Menü
st.sidebar.title("Gösterge Paneli")
app_mode = st.sidebar.selectbox("Sayfa Seçiniz",["Anasayfa","Proje Hakkinda","Tahminleme"])

#Ana Sayfa
if(app_mode=="Anasayfa"):
    st.header("MEYVE SEBZE TANIMA SİSTEMİ")
    image_path = "home_img.jpg"
    st.image(image_path)


#Proje Hakkında Bilgiler
elif(app_mode=="Proje Hakkinda"):
    st.header("Proje Hakkında Bilgi")
    st.subheader("Veri Seti Hakkında")
    st.text("Bu veri kümesi aşağıdaki gıda maddelerinin resimlerini içerir:")
    st.code("meyveler- muz, elma, armut, üzüm, portakal, kivi, karpuz, nar, ananas, mango.")
    st.code("sebzeler – salatalık, havuç, kırmızı biber, soğan, patates, limon, domates, turp, pancar, lahana, marul, ıspanak, soya fasulyesi, karnabahar, dolmalık biber, kırmızı biber, şalgam, mısır, mısır, tatlı patates, kırmızı biber, jalepeño, zencefil, sarımsak, bezelye, patlıcan.")
    st.subheader("İçerik")
    st.text("Bu veri kümesi üç klasör içeriyor:")
    st.text("1. Eğitim (her biri 100 resim)")
    st.text("2. test (her biri 10 resim)")
    st.text("3. doğrulama (her biri 10 resim)")

#Tahmin Sayfası 
elif(app_mode=="Tahminleme"):
    st.header("Model Tahmini")
    test_image = st.file_uploader("Lütfen Resim Seçin:")
    if(st.button("Resmi Göster")):
        st.image(test_image,width=4,use_column_width=True)
    #Tahmin Butonu
    if(st.button("Tahmin Et")):
        st.snow()
        st.write("Tahminimiz")
        result_index = model_prediction(test_image)
        #Okuma Kısmı
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        print(result_index, len(label), label)
        st.success("Modelin Tahmin Ettiği {}".format(label[result_index]))



