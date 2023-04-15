from tkinter import *
import tkinter
import tkinter.messagebox
import gensim
import pickle
from pyvi import ViTokenizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import os

MODEL_PATH='./model/'
DATA_PATH='./data/'
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
svd = TruncatedSVD(n_components=300, random_state=42)
danh_sach_model = []
val_model = ["naive_bayes_MultinomialNB_tfidf", 
            "naive_bayes_MultinomialNB_tfidf_ngram", 
            "naive_bayes_MultinomialNB_tfidf_ngram_char",
            "naive_bayes_BernoulliNB_tfidf", 
            "naive_bayes_BernoulliNB_tfidf_svd", 
            "linear_LogisticRegression_tfidf", 
            "linear_LogisticRegression_tfidf_svd",
            "svm_svc_tfidf_svd", 
            "ensemble_RandomForestClassifier_tfidf_svd", 
            "xgboost_XGBClassifier_tfidf_svd"]
print("============ danh sách model ====================")
for i in os.listdir(MODEL_PATH):
    danh_sach_model.append(i[0:-6])
    if val_model.__contains__(i[0:-6]):
        print("Yes " + i)
    else:
        print("No " + i)
print("=================================================")

class Article_Predict(Tk):
    def __init__(self):
        super(Article_Predict,self).__init__()
        event = None
        self.title('Nhận diện thể loại báo')
# Generate data
        self.val_stat = ""
        self.model=""
        self.danh_sach_label = ""
        self.X_data = pickle.load(open('./data/X_data.pkl', 'rb'))
        self.y_data = pickle.load(open('./data/y_data.pkl', 'rb'))
# first line
        self.label_category = Label(self, text="Danh mục")
        self.label_input = Label(self, text="Nhập bài báo")
        self.label_stat = Label(self, text = self.val_stat)
        self.label_model = Label(self, text = "Chọn model")

        self.label_category.grid(row=0, column=0, columnspan = 1)
        self.label_input.grid(row=0, column=1, columnspan = 1)
        self.label_stat.grid(row = 0, column = 2)
        self.label_model. grid(row = 0, column = 3, columnspan = 2)
# second line
        self.ta_input = Text(self, height=20, width=100)

        self.ta_input.grid(row = 1, column = 1, rowspan = 10, columnspan= 2)
# Predict area
        self.label_label = Label(self, font=(15))
        self.label_mod_sel = Label(self, font=(20))
        self.btn_reload = Button(self, text = "reload", command=self.load_data, width=30)
        self.btn_predict = Button(self, text = "predict", command=self.predict, width=30)
        self.btn_cancel = Button(self, text = "cancel", command = lambda: self.exit(), width=30)
        self.label_result = Label(self, text = "result", font=(30))
        
        self.label_label.grid(row = 11, column = 0, columnspan= 5)
        self.label_mod_sel.grid(row = 12, column = 0, columnspan= 5)
        self.btn_reload.grid(row = 13, column = 0, columnspan=2)
        self.btn_predict.grid(row = 13, column = 2)
        self.btn_cancel.grid(row=13, column = 3)
        self.label_result.grid(row = 14, column = 0, rowspan = 2 , columnspan = 5)
# Init function
        self.load_category()
        self.load_button_model()
        self.load_data()
        self.select_model(0)
        # self.test()

    def load_category(self):
        encoder = preprocessing.LabelEncoder()
        encoder.fit_transform(self.y_data)
        labels = []
        count = 0
        for i in encoder.classes_:
            labels.append(Label(self,text=i))
            labels[count].grid(row = count + 1, column = 0)
            count += 1
    
    def load_button_model(self):
        self.btn_mod1 = Button(self, text = "MulNB TFIDF", command = lambda: self.select_model(0), width=20)
        self.btn_mod2 = Button(self, text = "MulNB NGram", command = lambda: self.select_model(1), state=DISABLED, width=20)
        self.btn_mod3 = Button(self, text = "MulNB Ngram Char", command = lambda: self.select_model(2), state=DISABLED, width=20)
        self.btn_mod4 = Button(self, text = "BernoulliNB", command = lambda: self.select_model(3), width=20)
        self.btn_mod5 = Button(self, text = "BernoulliNB SVD", command = lambda: self.select_model(4), width=20)
        self.btn_mod6 = Button(self, text = "linear LoRe", command = lambda: self.select_model(5), width=20)
        self.btn_mod7 = Button(self, text = "linear LoRe SVD", command = lambda: self.select_model(6), width=20)
        self.btn_mod8 = Button(self, text = "svm svc SVD", command = lambda: self.select_model(7), width=20)
        self.btn_mod9 = Button(self, text = "ensemble RFC SVD", command = lambda: self.select_model(8), width=20)
        self.btn_mod10 = Button(self, text = "xgboost XGBC SVD", command = lambda: self.select_model(9), width=20)

        self.btn_mod1.grid(row = 1, column = 3, columnspan = 2)
        self.btn_mod2.grid(row = 2, column = 3, columnspan = 2)
        self.btn_mod3.grid(row = 3, column = 3, columnspan = 2)
        self.btn_mod4.grid(row = 4, column = 3, columnspan = 2)
        self.btn_mod5.grid(row = 5, column = 3, columnspan = 2)
        self.btn_mod6.grid(row = 6, column = 3, columnspan = 2)
        self.btn_mod7.grid(row = 7, column = 3, columnspan = 2)
        self.btn_mod8.grid(row = 8, column = 3, columnspan = 2)
        self.btn_mod9.grid(row = 9, column = 3, columnspan = 2)
        self.btn_mod10.grid(row = 10, column = 3, columnspan = 2)
    
    def load_data(self):
        print('0')
        tfidf_vect.fit(self.X_data) # learn vocabulary and idf from training set
        data_tfidf = tfidf_vect.transform(self.X_data)
        svd.fit(data_tfidf)
        # print("1")
        # tfidf_vect_ngram.fit(self.X_data)
        # print("2")
        # tfidf_vect_ngram_char.fit(self.X_data)
        print("sẵn sàng")

    def exit(self):
        self.destroy()

    def select_model(self, number):
        self.model = val_model[number]
        self.label_mod_sel.config(text = "Model được lựa chọn: " + self.model)
        self.label_result.config(text="Kết quả dự đoán: ")
        self.show_log("select model " + self.model)

    def preprocessing_doc(self, doc):
        lines = gensim.utils.simple_preprocess(doc)
        lines = ' '.join(lines)
        lines = ViTokenizer.tokenize(lines)
        return lines

    def data_handle(self):
        input = self.ta_input.get(1.0, END)
        # lines = gensim.utils.simple_preprocess(input)
        # lines = ' '.join(lines)
        # data = ViTokenizer.tokenize(lines)
        data = self.preprocessing_doc(input)
        data_tfidf = tfidf_vect.transform([data])
        return data_tfidf

    def data_handle_svd(self):
        input = self.ta_input.get(1.0, END)
        data = self.preprocessing_doc(input)
        data_tfidf = tfidf_vect.transform([data])
        # svd.fit(data_tfidf)
        data_svd = svd.transform(data_tfidf)
        return data_svd

    def label_load(self):
        encoder = preprocessing.LabelEncoder()
        y_data_n = encoder.fit_transform(self.y_data)
        print(encoder.classes_)
    
    def predict(self):
        file_path = os.path.join(MODEL_PATH, self.model + ".model")
        if not os.path.isfile(file_path):
            self.show_log("not have file " + file_path)
            self.show_error_message_box(text="Không tìm thấy model " + self.model)
        else:
            if self.model[-3:] != 'svd':
                print(self.model[-3:])
                data = self.data_handle()
            else:
                data = self.data_handle_svd()
            loaded_model = pickle.load(open(file_path, 'rb'))
            result=loaded_model.predict(data)
            self.label_result.config(text="Kết quả dự đoán là: " + result[0])
            self.show_info_message_box(title="Kết quả", text="Đây là bài báo thuộc thể loại " + result[0])

    def show_log(self, text):
        print("-----==========----- " + text)

    def show_info_message_box(self, title=None, text=None):
        tkinter.messagebox.showinfo(title=title, message=text)

    def show_error_message_box(self, title=None, text=None):
        tkinter.messagebox.showerror(title=title, message=text)

    # def test(self):



# window = Article_Predict()
# window.mainloop()
def main():
    window = Article_Predict()
    window.mainloop()
    
if __name__ == "__main__":  
    main()