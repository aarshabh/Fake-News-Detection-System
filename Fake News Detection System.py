import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.tree import DecisionTreeClassifier

df_fake = pd.read_csv("D:/Python/Project Fake News/Fake_news/Fake.csv")
df_true = pd.read_csv("D:/Python/Project Fake News/Fake_news/True.csv")

df_fake["class"] = 0
df_true["class"] = 1

df_fake_manual = df_fake.tail(10)
for i in range(23480, 23470 - 1):
    df_fake.drop([i], axis=0, inplace=True)

df_true_manual = df_fake.tail(10)
for i in range(21416, 21406 - 1):
    df_true.drop([i], axis=0, inplace=True)

# df_manual_test = pd.concat([df_fake_manual, df_true_manual], axis=0)
# df_manual_test.to_csv('D:/Python/Project Fake News/Fake_news/manual_testing.csv')
df_merge = pd.concat([df_fake, df_true], axis=0)
df = df_merge.drop(['title', 'subject', 'date'], axis=1)
df = df.sample(frac=1)


def word_drop(text):
    text = text.lower()
    text = re.sub("\[.*?", " ", text)
    text = re.sub('\\W', "", text)
    text = re.sub("https?://\S+|www\.\S+", '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


df["text"] = df["text"].apply(word_drop)
x = df["text"]
y = df["class"]
y = y.to_numpy()
y1 = y.flatten()
x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.25, random_state=0)
print(x_train)
print("-"*100)
print(y_train)

vector = TfidfVectorizer()
xv_train = vector.fit_transform(x_train)
xv_test = vector.transform(x_test)
log_reg = LogisticRegression()
log_reg.fit(xv_train, y_train)
# print(log_reg.score(xv_test, y_test))
x_pred = log_reg.predict(xv_test)
# print(classification_report(y_test, x_pred))
dt = DecisionTreeClassifier()
dt.fit(xv_train, y_train)
# print(dt.score(xv_test, y_test))
dt_pred = dt.predict(xv_test)
# print(classification_report(y_test, dt_pred))
gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(xv_train, y_train)
# print(gbc.score(xv_test, y_test))
gbc_pred = gbc.predict(xv_test)
rfc = RandomForestClassifier()
rfc.fit(xv_train, y_train)


def output_label(n):
    if n == 0:
        return "Fake News"
    else:
        return "Not A Fake News"


def manual_testing(text):
    testing_news = {"text": [text]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop)
    new_x_test = new_def_test["text"]
    new_xv_test = vector.transform(new_x_test)
    lr_pred = log_reg.predict(new_xv_test)
    pred_dt = dt.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    s1 = "LR Prediction: {}".format(output_label(lr_pred))
    s2 = "DT Prediction: {}".format(output_label(pred_dt))
    s3 = "GBC Prediction: {}".format(output_label(pred_gbc))
    s4 = "RFC Prediction: {}".format(output_label(pred_rfc))
    return print(s1 + "\n" + s2 + "\n" + s3 + "\n" + s4)


news = str(input("enter news:- "))
manual_testing(news)
