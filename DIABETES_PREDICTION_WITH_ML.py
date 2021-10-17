# İş Problemi: Özellikleri belirtildiğinde bir kişinin diyabetli olup olmadığını
# Tahmin edecek bir mak. ögr. modeli geliştirebilir misiniz?
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

#Diabetes veri setini tanıyalım.
#21 yaş üstündeki Pima India kadınları üzerinde yapılan testlerden gelen verilerdir.
df_= pd.read_csv("Hafta7/diabetes.csv")
df=df_.copy()
df.head()
# Hedef (bağımlı) değişken "Outcome" değişkenidir. Diyabetli olmayı 1, olmamayı 0 ifade ediyor.
# 8 tane de bağımsız değişken var.
# Pregnancies: Hamilelik Sayısı
# Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# BloodPressure: Kan Basıncı (Küçük Tansiyon)
# SkinThickness: Cilt kalınlığı
# Insulin: 2 saatlik serum insülini
# BMI(Body mass index): Vücut kitle indeksi. Kilo/boy^2-->(kg/m²)
# DiabetesPedigreeFunction: Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# Age: Yaş


#Gerekli olabilecek fonksiyonları tanımlayalım:
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#Veri setini anlamaya çalışalım:
check_df(df)
# Hedef değişkenin sınıfları ve frekansları:
df["Outcome"].value_counts() #0'lardan 500 tane var. 1'lerden 268 tane var.
# Hedef değişkenin sınıf oranları:
100 * df["Outcome"].value_counts() / len(df) #0'lar %65.104 oranında, 1'ler %34.896

# Targetın diğer (sayısal) değişkenlerle ilişkisi:target'ın yanına numeriklerin ort. al yaz.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
cols = [col for col in df.columns if "Outcome" not in col] #outcome dışındakileri al.
for col in cols:
    target_summary_with_num(df, "Outcome", col)  # Outcome iile birlikte numerikleri gez fonk'u uygula.

#Veri Ön İşleme
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Aykırı değer var mı?
for col in num_cols:
    print(col, check_outlier(df, col))  # Hepsi False geldi. Aykırı değer yok.

# Veri setinde bilimsel olarak sıfır olamayacak değerler var. Bunları Na ile değiştirmeliyiz.
# İlk önce na_cols ismini verelim sonra Na ile dolduralım.
na_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for col in na_cols:
    df[col].replace(0, np.NaN, inplace=True)
df.head()
df.isnull().sum()  #Na degerler kaç tane baktık.
#Şimdi de Na değerleri median ile dolduralım.
for i in na_cols:
    df.loc[df[i].isnull(), i] = df[i].fillna(df[i].median())

df.head()

corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns, annot=True, cmap='RdYlGn')
plt.show()

#Değiken Müh. (Yeni Değişkenler türetelim.)

#Age
df.loc[df["Age"] < 24, "NEW_AGE"] = "Young" #24 yaşından küçüklere "Young" de. Column ismi NEW_AGE.
df.loc[(df["Age"] >= 24) & (df['Age'] < 41), 'NEW_AGE'] = "Mature"
df.loc[(df["Age"] >= 41) & (df['Age'] < 55), 'NEW_AGE'] = "Senior"
df.loc[df["Age"] >= 55, "NEW_AGE"] = "Elder"

# BMI: Vücut kitle endeksine göre kilonun etkisini yorumlayabiliriz.
df.loc[df["BMI"] < 18.5, "NEW_BMI"] = "Underweight"
df.loc[(df["BMI"] >= 18.5) & (df['BMI'] < 25.0), 'NEW_BMI'] = "Normal"
df.loc[(df["BMI"] >= 25.0) & (df['BMI'] < 30.0), 'NEW_BMI'] = "Overweight"
df.loc[df["BMI"] >= 30.0, 'NEW_BMI'] = "Obese"

# Blood Pressure: Kan basıncı hangi aralıkta olmalı ona göre araştırıp kategorilere ayırdık.
df.loc[df["BloodPressure"] < 80, "NEW_BLOOD_PRE"] = "Optimal"
df.loc[(df["BloodPressure"] >= 80) & (df["BloodPressure"] <= 84), "NEW_BLOOD_PRE"] = "Normal"
df.loc[(df["BloodPressure"] >= 85) & (df["BloodPressure"] <= 89), "NEW_BLOOD_PRE"] = "High_normal"
df.loc[(df["BloodPressure"] >= 90) & (df["BloodPressure"] <= 99), "NEW_BLOOD_PRE"] = "Grade_1_hypertension"
df.loc[(df["BloodPressure"] >= 100) & (df["BloodPressure"] <= 109), "NEW_BLOOD_PRE"] = "Grade_2_hypertension"
df.loc[df["BloodPressure"] >= 110, "NEW_BLOOD_PRE"] = "Grade_3_hypertension"

# Glucose: Kandaki glikoz değerlerine göre kategorilere ayıralım.
df.loc[df["Glucose"] < 140, "NEW_GLUCOSE"] = "Normal"
df.loc[(df["Glucose"] >= 140) & (df['Glucose'] <= 199), "NEW_GLUCOSE"] = "Prediabetes"

# Age and Glucose: Yaş ve glikoz oranının yüksek olması etkili olabilir. İkisini birlikte gözlemleyelim.
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

# Insulin seviyesi değişkeni türetelim: 166dan büyük olanlar Abnormal.
df.loc[((df["Insulin"] >= 16) & (df["Insulin"] <=166)), "NEW_INSULIN_SCORE"] ="Normal"
df.loc[((df["Insulin"] >166) ), "NEW_INSULIN_SCORE"] ="Abnormal"

df.head()
numerical= [col for col in df.columns if df[col].dtypes != "O"]
cols = [col for col in numerical if "Outcome" not in col]

# Numerik değikenlerde standartlaştırma işlemi yapalım:
for col in numerical:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()
# Column isimlerin hepsi büyük yazsın:
df.columns = [col.upper() for col in df.columns]
df.head()

# Encoding İşlemleri: Mak. ögr. modeli bizden mumerik deger ister.
# 1)Label Encoder:
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# 2)One-HOt Encoding: Birb. farklı 2'den fazla sınıf sayısına sahip olanları kapsayacak şekilde.
df = pd.get_dummies(df, drop_first=True)
df.head()

# Model & Prediction: Model kurma. Logistic Regression yapacağız.
y = df["OUTCOME"] #target değişken.
X = df.drop(["OUTCOME"], axis=1) # target hariç diğerleri bağımsız değişken.x'e atadık.

# Train-Test olarak ayıralım. Holdout Yöntemi:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

log = LogisticRegression(solver='liblinear')
log_model = log.fit(X_train, y_train)

log_model.intercept_   # array([-1.27459214])
log_model.coef_

# Tahmin
y_pred = log_model.predict(X)
y_pred[0:10] # array([1., 0., 1., 0., 1., 0., 0., 0., 1., 1.])
y[0:10]

y_pred = log_model.predict(X_train)
y_pred[0:10] # array([1., 1., 0., 0., 0., 0., 1., 1., 0., 0.])

y_prob = log_model.predict_proba(X_train)[:, 1]

print(classification_report(y_train, y_pred)) #precision,recall veya f1 scoreuna bakabiliriz.

# test
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob) #f1-score: 0.84

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()  #AUC: 0.87





