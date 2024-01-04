###############################################################
# RFM Analizi ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

# Veri Seti Hikayesi
# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) olarak
# yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# Değişkenler

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date:  Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline:  Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import numpy as np
import pandas as pd
import datetime as dt

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%2f.' % x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

df.head(10)
df.columns
df.shape
df["master_id"].nunique()
df.describe().T
df.isnull().sum()
df.info()

df["order_channel"].value_counts()
df["last_order_channel"].value_counts()

def ilk_izlenim(dataframe):
    print(dataframe.head(10))
    print("--------------------------------------------------------------------")
    print(f"Column Names: \n{dataframe.columns}")
    print("--------------------------------------------------------------------")
    print(f"Shape: \n{dataframe.shape}")
    print("--------------------------------------------------------------------")
    print(f"Describe: \n{dataframe.describe().T}")
    print("--------------------------------------------------------------------")
    print(f"Null Check: \n{dataframe.isnull().sum()}")
    print("--------------------------------------------------------------------")
    print(f"Info: \n{dataframe.info()}")
    print("--------------------------------------------------------------------")
    print("Tekil PK: ")
    print(dataframe["master_id"].nunique())
    print("--------------------------------------------------------------------")
    print(dataframe["order_channel"].value_counts())
    print("--------------------------------------------------------------------")
    print(dataframe["last_order_channel"].value_counts())


ilk_izlenim(df)

###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################

df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

dt_columns = df.columns[df.columns.str.contains("date")]
df[dt_columns] = df[dt_columns].apply(pd.to_datetime)

df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total_ever": "sum",
                                 "customer_value_total_ever": "sum"})

df.sort_values(by="customer_value_total_ever", ascending=False).head(10)
df.sort_values(by="order_num_total_ever", ascending=False).head(10)

def data_prep(dataframe):
    # Toplam Harcama ve Toplam Sipariş Sayısı Hesabı
    dataframe["order_num_total_ever"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total_ever"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    # Tarih İçeren Değişkenlerin Transformasyonu
    dt_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[dt_columns] = dataframe[dt_columns].apply(pd.to_datetime)
    print("Veri analize hazır hale getirildi.")
    print("--------------------------------------------------------------------")
    print(dataframe.head())
    print("--------------------------------------------------------------------")
    print(dataframe.info())

    return dataframe


data_prep(df)


###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)

rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["order_num_total_ever"]
rfm["monetary"] = df["customer_value_total_ever"]


###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

rfm["RF_SCORE"] =rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)


###############################################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

def create_rfm(dataframe):
    # Veriyi Hazırlma
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)


    # RFM METRIKLERININ HESAPLANMASI
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm["customer_id"] = dataframe["master_id"]
    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).astype('timedelta64[D]')
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]

    # RF ve RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))


    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    return rfm[["customer_id", "recency","frequency","monetary","RF_SCORE","RFM_SCORE","segment"]]

rfm_df = create_rfm(df)


###############################################################
# 8. Kampanya Hazırlanması
###############################################################

# Kampanya 1:

# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde.
# Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçmek isteniliyor.
# Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

rfm_f = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]
rfm_f = rfm_f.merge(df[["master_id","interested_in_categories_12"]],
                    how="left",
                    left_on="customer_id",
                    right_on="master_id")
rfm_f.drop(columns="master_id", inplace=True)
camp_1 = rfm_f[rfm_f["interested_in_categories_12"].str.contains("KADIN")]["customer_id"]

camp_1.to_csv("kampanya_1.csv", index=False)


# Kampanya 2:

# Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır.
# Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama
# uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler
# özel olarak hedef alınmak isteniyor.
# Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.

rfm_f = rfm[rfm["segment"].isin(["cant_loose", "at_risk","hibernating","new_customers"])]
rfm_f = rfm_f.merge(df[["master_id","interested_in_categories_12"]],
                    how="left",
                    left_on="customer_id",
                    right_on="master_id")
rfm_f.drop(columns="master_id", inplace=True)
camp_2 = rfm_f[(rfm_f["interested_in_categories_12"].str.contains("ERKEK")) | (rfm_f["interested_in_categories_12"].str.contains("COCUK"))]["customer_id"]

camp_2.to_csv("kampanya_2.csv", index=False)











