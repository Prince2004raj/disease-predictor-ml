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
# Data loading — symptom list & disease encoding derived directly from the
# CSVs, so they can never go out of sync with the data.
# ---------------------------------------------------------------------------
df = pd.read_csv("Training.csv")
tr = pd.read_csv("Testing.csv")

l1 = [c for c in df.columns if c != 'prognosis']

le = LabelEncoder()
le.fit(df['prognosis'])

X = df[l1]
y = le.transform(df['prognosis'])
X_test = tr[l1]
y_test = le.transform(tr['prognosis'])

print("Training models, please wait...")
clf_tree = tree.DecisionTreeClassifier().fit(X, y)
print("DecisionTree accuracy:", accuracy_score(y_test, clf_tree.predict(X_test)))
clf_rf = RandomForestClassifier().fit(X, y)
print("RandomForest accuracy:", accuracy_score(y_test, clf_rf.predict(X_test)))
clf_nb = GaussianNB().fit(X, y)
print("NaiveBayes accuracy:", accuracy_score(y_test, clf_nb.predict(X_test)))
print("Training complete.")

# ---------------------------------------------------------------------------
# Precautions — loaded from symptom_precaution.csv (must be in same folder).
# Keys are stripped so 'Diabetes ' / 'Diabetes' both match safely.
# ---------------------------------------------------------------------------
prec_df = pd.read_csv("symptom_precaution.csv")
PRECAUTIONS = {}
for _, row in prec_df.iterrows():
    name = str(row["Disease"]).strip()
    precs = [str(row[c]).strip() for c in
             ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
             if str(row[c]).strip() and str(row[c]).strip().lower() != "nan"]
    PRECAUTIONS[name] = precs

# Life-threatening / emergency-level diseases that trigger the red alert.
SEVERE_DISEASES = {
    "Heart attack", "AIDS", "Paralysis (brain hemorrhage)", "Tuberculosis",
    "Dengue", "Pneumonia", "Hepatitis B", "Hepatitis C", "Hepatitis D",
    "Hepatitis E", "hepatitis A", "Alcoholic hepatitis", "Typhoid", "Malaria",
}

_blink_jobs = {}  # widget -> after() job id, so we can cancel old blinks


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def get_selected_symptoms():
    picked = [Symptom1.get(), Symptom2.get(), Symptom3.get(),
              Symptom4.get(), Symptom5.get()]
    picked = [p for p in picked if p in l1]

    if not picked:
        messagebox.showwarning("No symptoms selected",
                                "Please select at least one symptom.")
        return None
    if len(picked) < 2:
        messagebox.showinfo("Tip",
                             "For better accuracy, try selecting 2 or more symptoms.")

    l2 = [0] * len(l1)
    for k in range(len(l1)):
        if l1[k] in picked:
            l2[k] = 1
    return pd.DataFrame([l2], columns=l1)


def stop_blink(widget, normal_bg):
    job = _blink_jobs.pop(widget, None)
    if job is not None:
        root.after_cancel(job)
    widget.config(bg=normal_bg)


def start_blink(widget, normal_bg, times_left=8):
    """Alternate the widget's background between red and black a few times,
    then settle on solid red so the alert stays visible."""
    job = _blink_jobs.pop(widget, None)
    if job is not None:
        root.after_cancel(job)

    if times_left <= 0:
        widget.config(bg="red", fg="white")
        return

    current = widget.cget("bg")
    next_color = "black" if current == "red" else "red"
    widget.config(bg=next_color, fg="white")
    job = root.after(400, lambda: start_blink(widget, normal_bg, times_left - 1))
    _blink_jobs[widget] = job


def show_result(output_box, normal_bg, disease_name):
    output_box.delete("1.0", END)
    output_box.insert(END, disease_name)

    if disease_name in SEVERE_DISEASES:
        start_blink(output_box, normal_bg)
        precaution_label.config(
            text="⚠️ This may be a serious condition — please seek medical help immediately.",
            fg="red"
        )
    else:
        stop_blink(output_box, normal_bg)
        precaution_label.config(text="", fg="black")

    precs = PRECAUTIONS.get(disease_name, [])
    precaution_box.delete("1.0", END)
    if precs:
        precaution_box.insert(END, "Precautions:\n" + "\n".join(f"• {p}" for p in precs))
    else:
        precaution_box.insert(END, "No precaution data available.")
    precaution_box.insert(END, "\n\nNote: This is not a medical diagnosis. Consult a doctor.")


def predict_with(clf, output_box, normal_bg):
    inputtest = get_selected_symptoms()
    if inputtest is None:
        return
    predicted_label = clf.predict(inputtest)[0]
    disease_name = le.inverse_transform([predicted_label])[0]
    show_result(output_box, normal_bg, disease_name)


def DecisionTree():
    predict_with(clf_tree, t1, "Light green")


def randomforest():
    predict_with(clf_rf, t2, "White")


def NaiveBayes():
    predict_with(clf_nb, t3, "red")


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

# --- Fixed SOS / emergency number, top-right corner ---
sos_label = Label(root, text="🚨 Emergency: 108 / 112", fg="white", bg="red")
sos_label.config(font=("Times", 14, "bold"))
sos_label.grid(row=1, column=3, rowspan=2, padx=20, sticky=N)

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

# --- Shared precautions panel (updates after any of the 3 predictions) ---
precaution_title = Label(root, text="Precautions", fg="white", bg="black")
precaution_title.config(font=("Times", 15, "bold italic"))
precaution_title.grid(row=21, column=0, pady=(20, 0), sticky=W)

precaution_label = Label(root, text="", bg="black")
precaution_label.config(font=("Times", 12, "bold"))
precaution_label.grid(row=21, column=1, pady=(20, 0), sticky=W)

precaution_box = Text(root, height=6, width=60, bg="white", fg="black")
precaution_box.config(font=("Times", 12))
precaution_box.grid(row=22, column=0, columnspan=2, padx=10, pady=10, sticky=W)

root.mainloop()
