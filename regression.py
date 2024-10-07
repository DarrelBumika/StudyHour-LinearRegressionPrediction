import base64
import streamlit as st
import sklearn
import joblib, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def main():
    st.set_page_config(initial_sidebar_state="collapsed")

    with open("src/style/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    file_ = open("src/img/giphy.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(f"""
    <div style="display:flex;flex-dir:row">
        <h1>Hasil Skor Ujian Berdasarkan Jam Belajar Siswa</h1>
        <img style="width:300px;height:auto;object-fit:contain;" src="data:image/gif;base64,{data_url}" alt="cat gif">
    </div>
    """, unsafe_allow_html=True)

    html_templ = """
        <div style="background-color:#b17457;padding:0px 20px;border-radius:10px">
            <h3 style="color:white;font-size:25px">Memprediksi Skor Ujian Siswa Menggunakan Regresi Linier</h3>
        </div>
    """

    st.markdown(html_templ, unsafe_allow_html=True)

    with st.sidebar:
        activity = ["Skor Ujian Berdasarkan Jam Belajar", "Apa itu Regresi?"]
        choice = st.sidebar.selectbox("Menu", activity)

    if choice == "Skor Ujian Berdasarkan Jam Belajar":
        st.subheader("Skor Ujian Berdasarkan Jam Belajar")
        study_hour = st.slider("Berapa total jam belajar: ", 0.0, 10.0)

        if st.button("PROSES"):
            st.subheader("Hasil Prediksi Skor Ujian: ")
            reg = load_prediction_models("models/linear_regression_student.pkl")
            study_hour_reshaped = np.array(study_hour).reshape(-1, 1)
            predicted_score = reg.predict(study_hour_reshaped)
            predicted_score = predicted_score[0][0] if predicted_score[0][0] <= 100 else float(100)
            st.markdown(f"""
                <div style="display:flex;justify-content:center;align-items:center;padding:20px;border-radius:10px;background-color:#b17457">
                    <p style="font-size:4rem;color:#faf7f0">{predicted_score:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("Detail Prediksi"):
                df = pd.read_csv("data/score_updated.csv")
                x_range = np.linspace(df["Hours"].min(), df["Hours"].max(), 100)
                y_range = reg.predict(x_range.reshape(-1, 1)).flatten()
                fig = go.Figure([
                    go.Scatter(
                        x=df["Hours"],
                        y=df["Scores"],
                        mode="markers",
                        marker=dict(color="#d8d2c2", size=10),
                        name="Data"
                    ),
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode="lines",
                        line=dict(color="#b17457", width=3),
                        name="Regression Line"
                    ),
                    go.Scatter(
                        x=[study_hour],
                        y=[predicted_score],
                        mode="markers",
                        marker=dict(color="#4a4947", size=10),
                        name="Prediction",
                    ),
                ])
                event = st.plotly_chart(fig, on_select="rerun")\

    else:
        st.divider()
        st.markdown("""
        <div style="text-align: center;">
            Linear regresi adalah metode statistik yang digunakan untuk memodelkan hubungan antara satu variabel 
            independen (prediktor) dengan variabel dependen (hasil) dengan menggunakan persamaan garis lurus. Tujuannya 
            adalah untuk menemukan garis terbaik yang memprediksi nilai variabel dependen berdasarkan nilai variabel 
            independen.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
