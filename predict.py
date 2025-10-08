import streamlit as st
import pandas as pd
import pickle

with open('best_model.pkl', 'rb') as file:
    model_inference = pickle.load(file)

def run():
    st.title('Prediksi Churn Nasabah Bank')
    st.write('Masukkan data calon nasabah di bawah ini:')

    with st.form(key='form_churn'):
        col1, col2 = st.columns(2)

        with col1:
            row_number = st.number_input('Row Number', min_value=0, value=0)
            customer_id = st.number_input('Customer ID', min_value=0, value=0)
            surname = st.text_input('Surname', placeholder='Contoh: Albert')
            credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=0)
            geography = st.selectbox('Geography', ['Pilih', 'France', 'Germany', 'Spain'])
            gender = st.selectbox('Gender', ['Pilih', 'Male', 'Female'])
            age = st.number_input('Age', min_value=0, max_value=120, value=0)

        with col2:
            tenure = st.number_input('Tenure', min_value=0, max_value=10, value=0)
            balance = st.number_input('Balance', min_value=0.0, value=0.0, format="%.2f")
            num_products = st.selectbox('Number of Products', ['Pilih', 1, 2, 3, 4])
            has_cr_card = st.selectbox('Has Credit Card?', ['Pilih', 0, 1])
            is_active = st.selectbox('Is Active Member?', ['Pilih', 0, 1])
            est_salary = st.number_input('Estimated Salary', min_value=0.0, value=0.0, format="%.2f")

        submit = st.form_submit_button('Predict')

    if submit:
        # Cek validasi pilihan agar tidak 'Pilih'
        if 'Pilih' in [geography, gender, num_products, has_cr_card, is_active]:
            st.warning('Silakan lengkapi semua input terlebih dahulu.')
        else:
            data_inf = pd.DataFrame([{
                'RowNumber': row_number,
                'CustomerId': customer_id,
                'Surname': surname,
                'CreditScore': credit_score,
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': int(num_products),
                'HasCrCard': int(has_cr_card),
                'IsActiveMember': int(is_active),
                'EstimatedSalary': est_salary
            }])

            st.subheader('Data yang Dimasukkan:')
            st.dataframe(data_inf)

            pred = model_inference.predict(data_inf)

            st.subheader('Hasil Prediksi:')
            if pred[0] == 1:
                st.error('Nasabah ini berpotensi untuk churn (berhenti menggunakan layanan).')
            else:
                st.success('Nasabah ini cenderung loyal (tidak churn).')

if __name__ == '__main__':
    run()
