from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------


df = pd.read_csv("Training.csv")
tr = pd.read_csv("Testing.csv")

l1 = [c for c in df.columns if c != 'prognosis']  # all 132 symptom columns

le = LabelEncoder()
le.fit(df['prognosis'])

X = df[l1]
y = le.transform(df['prognosis'])

X_test = tr[l1]
y_test = le.transform(tr['prognosis'])


# ---------------------------------------------------------------------------
print("Training models, please wait...")

clf_tree = tree.DecisionTreeClassifier().fit(X, y)
print("DecisionTree accuracy:", accuracy_score(y_test, clf_tree.predict(X_test)))

clf_rf = RandomForestClassifier().fit(X, y)
print("RandomForest accuracy:", accuracy_score(y_test, clf_rf.predict(X_test)))

clf_nb = GaussianNB().fit(X, y)
print("NaiveBayes accuracy:", accuracy_score(y_test, clf_nb.predict(X_test)))

print("Training complete.")


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def get_selected_symptoms():
    """Read the 5 dropdowns and build a fresh symptom vector (fixes the
    original bug where the vector was never reset between predictions)."""
    picked = [Symptom1.get(), Symptom2.get(), Symptom3.get(),
              Symptom4.get(), Symptom5.get()]
    picked = [p for p in picked if p in l1]  # ignore "Select Here" placeholders

    if not picked:
        messagebox.showwarning("No symptoms selected",
                                "Please select at least one symptom.")
        return None

    l2 = [0] * len(l1)
    for k in range(len(l1)):
        if l1[k] in picked:
            l2[k] = 1
    return pd.DataFrame([l2], columns=l1)


def predict_with(clf, output_box):
    inputtest = get_selected_symptoms()
    if inputtest is None:
        return
    predicted_label = clf.predict(inputtest)[0]
    disease_name = le.inverse_transform([predicted_label])[0]
    output_box.delete("1.0", END)
    output_box.insert(END, disease_name)


def DecisionTree():
    predict_with(clf_tree, t1)


def randomforest():
    predict_with(clf_rf, t2)


def NaiveBayes():
    predict_with(clf_nb, t3)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
root = Tk()
root.configure(background='black')
root.title("Disease Predictor using Machine Learning")

Symptom1 = StringVar(); Symptom1.set("Select Here")
Symptom2 = StringVar(); Symptom2.set("Select Here")
Symptom3 = StringVar(); Symptom3.set("Select Here")
Symptom4 = StringVar(); Symptom4.set("Select Here")
Symptom5 = StringVar(); Symptom5.set("Select Here")
Name = StringVar()

w2 = Label(root, justify=LEFT, text="Disease Predictor using Machine Learning", fg="Red", bg="White")
w2.config(font=("Times", 30, "bold italic"))
w2.grid(row=1, column=0, columnspan=2, padx=100)

w2 = Label(root, justify=LEFT, text="This Project by  Roshan Ali and Prince Raj", fg="Pink", bg="Blue")
w2.config(font=("Times", 30, "bold italic"))
w2.grid(row=2, column=0, columnspan=2, padx=100)

NameLb = Label(root, text="Name of the Patient", fg="Red", bg="Sky Blue")
NameLb.config(font=("Times", 15, "bold italic"))
NameLb.grid(row=6, column=0, pady=15, sticky=W)

S1Lb = Label(root, text="Symptom 1", fg="Blue", bg="Pink")
S1Lb.config(font=("Times", 15, "bold italic")); S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="White", bg="Purple")
S2Lb.config(font=("Times", 15, "bold italic")); S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Green", bg="white")
S3Lb.config(font=("Times", 15, "bold italic")); S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="blue", bg="Yellow")
S4Lb.config(font=("Times", 15, "bold italic")); S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="purple", bg="light green")
S5Lb.config(font=("Times", 15, "bold italic")); S5Lb.grid(row=11, column=0, pady=10, sticky=W)

lrLb = Label(root, text="DecisionTree", fg="white", bg="red")
lrLb.config(font=("Times", 15, "bold italic")); lrLb.grid(row=15, column=0, pady=10, sticky=W)

destreeLb = Label(root, text="RandomForest", fg="Red", bg="Orange")
destreeLb.config(font=("Times", 15, "bold italic")); destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="White", bg="green")
ranfLb.config(font=("Times", 15, "bold italic")); ranfLb.grid(row=19, column=0, pady=10, sticky=W)

OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1 = OptionMenu(root, Symptom1, *OPTIONS); S1.grid(row=7, column=1)
S2 = OptionMenu(root, Symptom2, *OPTIONS); S2.grid(row=8, column=1)
S3 = OptionMenu(root, Symptom3, *OPTIONS); S3.grid(row=9, column=1)
S4 = OptionMenu(root, Symptom4, *OPTIONS); S4.grid(row=10, column=1)
S5 = OptionMenu(root, Symptom5, *OPTIONS); S5.grid(row=11, column=1)

dst = Button(root, text="Prediction 1", command=DecisionTree, bg="Red", fg="yellow")
dst.config(font=("Times", 15, "bold italic")); dst.grid(row=8, column=3, padx=10)

rnf = Button(root, text="Prediction 2", command=randomforest, bg="White", fg="green")
rnf.config(font=("Times", 15, "bold italic")); rnf.grid(row=9, column=3, padx=10)

lr = Button(root, text="Prediction 3", command=NaiveBayes, bg="Blue", fg="white")
lr.config(font=("Times", 15, "bold italic")); lr.grid(row=10, column=3, padx=10)

t1 = Text(root, height=1, width=40, bg="Light green", fg="red")
t1.config(font=("Times", 15, "bold italic")); t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40, bg="White", fg="Blue")
t2.config(font=("Times", 15, "bold italic")); t2.grid(row=17, column=1, padx=10)

t3 = Text(root, height=1, width=40, bg="red", fg="white")
t3.config(font=("Times", 15, "bold italic")); t3.grid(row=19, column=1, padx=10)

root.mainloop()
