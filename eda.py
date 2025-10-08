import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    # Title
    st.title("Prediksi Nasabah Churn - Bank XYZ")

    # Image
    st.image('https://siplawfirm.id/wp-content/uploads/2019/04/pengaturan-perbankan.jpg',caption='Source : google.com')

    # Latar Belakang
    st.header("Latar Belakang")
    st.markdown('''
    Sebagai data scientist di perusahaan perbankan, saya mendapat tugas untuk membuat model yang bisa membantu tim marketing mengetahui lebih awal siapa saja nasabah yang berisiko tinggi akan berhenti menjadi pelanggan. Hal ini penting karena mempertahankan nasabah lama jauh lebih hemat dan menguntungkan daripada harus mencari nasabah baru. Model ini nantinya akan digunakan sebagai alat bantu supaya tim bisa mengambil tindakan yang cepat, seperti menghubungi nasabah, memberikan penawaran khusus, atau cara lainnya sebelum mereka benar benar pergi.
    ''')

    # Dataset
    st.header("Dataset yang Digunakan")
    st.markdown('''
    Dataset yang digunakan adalah data nasabah dari Bank XYZ yang terdiri dari informasi demografi,
    aktivitas perbankan, dan atribut lainnya. Dataset ini terdiri dari 10.000 nasabah dan telah melalui proses pembersihan data,
    pengecekan missing values, duplikasi, dan outlier.

    Fitur fitur penting dalam dataset ini antara lain:
    - **Row Number** : Nomor Baris
    - **CustomerId** : ID unik nasabah
    - **Surname** : Nama belakang nasabah
    - **Credit Score** : Skor kredit nasabah
    - **Geography** : Asal negara nasabar
    - **Gender** : Jenis kelamin nasabah
    - **Age** : Umur nasabah
    - **Tenure** : Lama menjadi nasabah (dalam tahun)
    - **Balance** : Jumlah saldo nasabah di rekening bank
    - **NumOfProducts** : Jumlah produk yang digunakan nasabah (tabungan, kartu kredit, pinjaman, dll)
    - **HasCrCard** : Apakah nasabah memiliki kartu kredit?
    - **IsActiveMember** : 
    - **EstimatedSalary** : Estimasi gaji tahunan nasabah
    - **Exited** : Status churn
    ''')

    # Load Dataset
    df = pd.read_csv("Churn_Modelling.csv")

    # Dataframe
    st.dataframe(df)

    # EDA
    st.header("Exploratory Data Analysis (EDA)")

    # Distribusi Target
    st.subheader("Distribusi Target (Exited)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    df['Exited'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0],
                                         colors=['#66b3ff', '#ff9999'],
                                         labels=['Tidak Churn', 'Churn'])
    sns.countplot(data=df, x='Exited', ax=ax[1], palette='Set2')
    ax[0].set_title("Pie Chart")
    ax[1].set_title("Barchart")
    ax[1].set_xticklabels(['Tidak Churn', 'Churn'])
    st.pyplot(fig)
    st.markdown("""
    **Pie Chart**

    - Dari distribusi visualisasi pie chart, jumlah orang yang tidak churn atau tetap menjadi nasabah itu lebih banyak daripada yang churn. Persentase dari yang tidak churn itu sebanyak 79.6%, sedangkan yang churn itu 20.4%.

    **Barchart**

    - Grafik barchart menunjukkan hal yang sama dengan piechart, yang dimana yang tidak churn itu lebih banyak daripada yang churn. Kalau di piechart kita bisa melihat persentasenya, di barchart kita bisa melihat jumlah dari yang tidak churn (0) maupun yang churn (1). Jumlah data yang churn (0) itu sebanyak kurang lebih 8000, sedangkan yang churn itu sebanyak 2000.

    - Dari 2 grafik ini bisa disimpulkan bahwa data target tidak imbalance dan imbalancenyapun sangat jauh perbedaannya, sehingga ini nanti perlu dilakukan balancing data agar data tidak bias.
    """)

    # Distribusi Umur
    st.subheader("Distribusi Umur Nasabah")
    fig = plt.figure(figsize=(10, 5))
    sns.histplot(df['Age'], kde=True, bins=30, color='skyblue')
    plt.title("Distribusi Usia Nasabah")
    st.pyplot(fig)
    st.markdown("""
    - Berdasarkan grafik distribusi umur, mayoritas nasabah berada di rentang usia 30 hingga 40 tahun, yang merupakan kelompok usia produktif dan aktif secara finansial. Setelah usia 40 tahun, jumlah nasabah mulai menurun, dan sangat sedikit yang berusia di atas 60 tahun. Hal ini menunjukkan bahwa nasabah bank didominasi oleh orang orang yang sedang dalam masa kerja aktif, sehingga strategi pemasaran atau produk perbankan sebaiknya difokuskan pada kelompok usia ini.
    """)

    # Saldo vs Churn
    st.subheader("Distribusi Saldo Berdasarkan Status Churn")
    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(x='Exited', y='Balance', data=df, palette='coolwarm')
    plt.title("Perbandingan Saldo antara Churn dan Tidak Churn")
    st.pyplot(fig)
    st.markdown("""
    - Dari grafik di atas, terlihat bahwa nasabah yang churn (exited = 1) justru memiliki saldo yang sedikit lebih tinggi dibandingkan nasabah yang tetap menjadi nasabah pelanggan (exited = 0). Rata rata saldo mereka lebih tinggi, dan rentang saldo mereka juga lebih luas. Ini menunjukkan bahwa nasabah dengan saldo besar pun belum tentu loyal, sehingga bank perlu memahami lebih dalam kenapa nasabah bernilai tinggi bisa memutuskan untuk keluar.
    """)

    # Aktivitas vs Churn
    st.subheader("Aktivitas Nasabah vs Churn")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(ax=axes[0], x='IsActiveMember', hue='Exited', data=df, palette='Set2')
    axes[0].set_title('Jumlah Nasabah berdasarkan Aktivitas dan Churn')
    axes[0].set_xlabel('IsActiveMember (1 = Aktif, 0 = Tidak Aktif)')
    axes[0].legend(title='Exited', labels=['Tidak Churn', 'Churn'])

    active_prop = df.groupby('IsActiveMember')['Exited'].value_counts(normalize=True).unstack()
    active_prop.plot(kind='bar', stacked=True, color=['#A1D99B','#FC9272'], ax=axes[1])
    axes[1].set_title('Proporsi Churn berdasarkan Aktivitas Nasabah')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['0', '1'], rotation=0)
    axes[1].legend(['Tidak Churn','Churn'], bbox_to_anchor=(1.05, 1), loc='upper left')
    for i, val in enumerate(active_prop[1]):
        axes[1].text(i, active_prop[0][i] + val / 2, f'{val:.1%}', ha='center')
    st.pyplot(fig)
    st.markdown("""
    - Dari grafik tersebut, terlihat bahwa nasabah yang tidak aktif (IsActiveMember = 0) memiliki jumlah churn lebih tinggi dibandingkan nasabah aktif. Sebaliknya, nasabah yang aktif cenderung lebih setia atau tidak churn. Artinya, aktivitas nasabah berhubungan dengan loyalitas mereka, semakin aktif mereka menggunakan layanan bank, semakin kecil kemungkinan mereka berhenti.

    - Nasabah yang tidak aktif (IsActiveMember = 0) memiliki jumlah churn lebih tinggi secara absolut dibandingkan yang aktif, dan jika dilihat dari proporsinya, sekitar 27% nasabah tidak aktif melakukan churn, hampir dua kali lipat dibandingkan nasabah aktif yang hanya sekitar 14%.
    """)

    # Negara vs Churn
    st.subheader("Negara vs Churn")
    geo_prop = df.groupby('Geography')['Exited'].value_counts(normalize=True).unstack()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(ax=axes[0], x='Geography', hue='Exited', data=df, palette='husl')
    axes[0].set_title('Jumlah Nasabah per Negara berdasarkan Churn')
    geo_prop.plot(kind='bar', stacked=True, color=['#FFB6C1','#20B2AA'], ax=axes[1])
    axes[1].set_title('Proporsi Churn per Negara')
    for i in range(len(geo_prop)):
        val = geo_prop.iloc[i, 1]
        base = geo_prop.iloc[i, 0]
        axes[1].text(i, base + val / 2, f'{val:.0%}', ha='center')
    st.pyplot(fig)
    st.markdown("""
    - Berdasarkan grafik di atas, terlihat bahwa nasabah dari France merupakan yang terbanyak secara keseluruhan, dan mayoritas dari mereka tidak churn. Sebaliknya, nasabah dari Germany memiliki jumlah churn yang hampir sama banyaknya dengan yang tidak churn, sehingga tingkat churn di Germany terlihat paling tinggi dibanding negara lain. Sementara itu, nasabah dari Spain sebagian besar juga tetap menjadi pelanggan. Ini menunjukkan bahwa nasabah dari Germany lebih berisiko churn, sehingga bank perlu memperhatikan strategi khusus untuk pelanggan di negara tersebut.

    - Berdasarkan grafik di atas, nasabah dari France merupakan yang terbanyak secara keseluruhan, dan mayoritas dari mereka tidak churn. Namun jika dilihat dari proporsinya, nasabah dari Germany memiliki tingkat churn tertinggi, yaitu sekitar 32%, jauh lebih besar dibanding France (16%) dan Spain (17%). Meskipun jumlah churn di Germany tidak paling banyak, tetapi secara persentase terhadap total nasabah di negara tersebut, risiko churn di Germany jauh lebih besar. Hal ini menunjukkan bahwa bank perlu memberikan perhatian khusus terhadap nasabah di Germany karena mereka cenderung lebih mudah berhenti menggunakan layanan dibanding nasabah dari negara lain.
    """)

    # Gender vs Churn
    st.subheader("Gender vs Churn")
    gender_group = df.groupby('Gender')['Exited'].value_counts(normalize=True).unstack()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.countplot(ax=axes[0], x='Gender', hue='Exited', data=df, palette='pastel')
    axes[0].set_title('Jumlah Nasabah per Gender dan Churn')
    axes[1].pie(gender_group.loc['Female'], labels=['Tidak Churn','Churn'],
                autopct='%1.1f%%', colors=['#ADD8E6','#F4A261'])
    axes[1].set_title('Proporsi Churn - Female')
    axes[2].pie(gender_group.loc['Male'], labels=['Tidak Churn','Churn'],
                autopct='%1.1f%%', colors=['#ADD8E6','#F4A261'])
    axes[2].set_title('Proporsi Churn - Male')
    st.pyplot(fig)
    st.markdown("""
    - Berdasarkan grafik, jumlah nasabah pria memang sedikit lebih banyak dibanding wanita, dan secara jumlah, nasabah wanita memiliki churn yang lebih tinggi. Namun jika dilihat dari persentasenya, sekitar 25% wanita churn, sedangkan pada pria hanya sekitar 16%. Artinya, nasabah wanita memiliki risiko churn yang lebih tinggi dibanding pria, sehingga bank perlu mempertimbangkan pendekatan layanan yang lebih personal atau sesuai kebutuhan wanita untuk meningkatkan loyalitas mereka sebagai pelanggan.
    """)

    # Jumlah Produk vs Churn
    st.subheader("Jumlah Produk vs Churn")
    product_prop = df.groupby('NumOfProducts')['Exited'].value_counts(normalize=True).unstack()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(ax=axes[0], x='NumOfProducts', hue='Exited', data=df, palette='cool')
    product_prop.plot(kind='bar', stacked=True, color=['#A1D99B','#FC9272'], ax=axes[1])
    axes[1].set_title('Proporsi Churn berdasarkan Jumlah Produk')
    for i, val in enumerate(product_prop.iloc[:, 1]):
        base = product_prop.iloc[i, 0]
        if pd.notnull(val): axes[1].text(i, base + val/2, f'{val:.1%}', ha='center')
    st.pyplot(fig)
    st.markdown("""
    - Dari visualisasi di atas, nasabah yang memiliki 1 produk memang menyumbang jumlah churn paling banyak, namun jika dilihat dari persentasenya, justru nasabah dengan 3 dan 4 produk memiliki tingkat churn yang jauh lebih tinggi, yaitu 83% dan bahkan 100%. Sementara itu, nasabah dengan 2 produk merupakan kelompok terbesar dan paling stabil, karena hanya sekitar 8% yang churn. Hal ini menunjukkan bahwa memiliki terlalu banyak produk justru bisa meningkatkan risiko churn, kemungkinan karena beban, kebingungan, atau ketidakpuasan. Bank perlu lebih berhati hati dalam menawarkan banyak produk sekaligus, dan memastikan bahwa setiap penawaran benar-benar relevan dan sesuai kebutuhan nasabah.
    """)

    st.markdown('---')
    st.caption('Dashboard by Gede Davon Ananda Putra | Hacktiv8 HCK-032')


if __name__ == '__main__':
    run()
