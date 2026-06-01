import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import datetime
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from include.text_processing import clean_text
from include.database import DatabaseLayer
from src.ml_pipeline import generate_synthetic_dataset, build_new_pipeline

BG_DARK, BG_CARD, BG_INPUT = "#0f172a", "#1e293b", "#0f2035"
TXT_MAIN, TXT_DIM, BORDER = "#e2e8f0", "#94a3b8", "#334155"
TEAL, BLUE, RED, GREEN, YELLOW = "#14b8a6", "#3b82f6", "#ef4444", "#22c55e", "#f59e0b"
FONT_H2, FONT_BODY, FONT_SM, FONT_MONO = ("Consolas", 12, "bold"), ("Consolas", 10), ("Consolas", 9), ("Courier New", 10)

class AppState:
    def __init__(self):
        self.db = DatabaseLayer()
        self.model = None          
        self.run_metrics = {}            
        self.session_logs = []            

class FakeNewsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.state_obj = AppState()
        self.title("Fake News Detection System")
        self.configure(bg=BG_DARK)
        self.geometry("1100x800")
        self._build_ui()
        self._refresh_db_status()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG_DARK)
        tk.Label(hdr, text="⬡ FAKE NEWS DETECTION SYSTEM", font=("Consolas", 20, "bold"), fg=TEAL, bg=BG_DARK).pack(side="left")
        self.lbl_db_status = tk.Label(hdr, text="● DB: checking…", font=FONT_SM, fg=YELLOW, bg=BG_DARK)
        self.lbl_db_status.pack(side="right")
        hdr.pack(fill="x", padx=20, pady=10)

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=6)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)

        left = self._build_left_panel(body)
        right = self._build_right_panel(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right.grid(row=0, column=1, sticky="nsew")

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status_var, font=FONT_SM, fg=TXT_DIM, bg=BG_CARD, anchor="w", padx=10).pack(fill="x", side="bottom")

    def _card(self, parent, title: str):
        w = tk.Frame(parent, bg=BG_DARK)
        tk.Label(w, text=title, font=FONT_H2, fg=TEAL, bg=BG_DARK).pack(anchor="w", pady=2)
        c = tk.Frame(w, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        c.pack(fill="both", expand=True)
        return w, c

    def _build_left_panel(self, parent):
        f = tk.Frame(parent, bg=BG_DARK)
        w_s, c_s = self._card(f, "▸ CONFIG & TRAINING")
        w_s.pack(fill="x", pady=4)
        
        btn_train = tk.Button(c_s, text="⚡  TRAIN MODEL", font=FONT_H2, fg=BG_DARK, bg=TEAL, command=self._start_training_thread, relief="flat", cursor="hand2")
        btn_train.pack(fill="x", padx=10, pady=10)

        w_m, c_m = self._card(f, "▸ RUN METRICS")
        w_m.pack(fill="both", expand=True, pady=4)
        self.metrics_text = scrolledtext.ScrolledText(c_m, font=FONT_MONO, bg=BG_INPUT, fg=GREEN, state="disabled")
        self.metrics_text.pack(fill="both", expand=True, padx=8, pady=8)

        w_a, c_a = self._card(f, "▸ CONTROLS")
        w_a.pack(fill="x", pady=4)
        tk.Button(c_a, text="🔻 Drop Weights", fg=RED, bg=BG_CARD, command=self._drop_model, relief="flat").pack(fill="x", padx=8, pady=2)
        tk.Button(c_a, text="📊 Show Analytics Chart", fg=TEAL, bg=BG_CARD, command=self._show_chart, relief="flat").pack(fill="x", padx=8, pady=2)

        return f

    def _build_right_panel(self, parent):
        f = tk.Frame(parent, bg=BG_DARK)
        w_i, c_i = self._card(f, "▸ ARTICLE TEXT INPUT")
        w_i.pack(fill="both", expand=True, pady=4)

        self.article_input = scrolledtext.ScrolledText(c_i, font=FONT_MONO, bg=BG_INPUT, fg=TXT_MAIN, wrap="word")
        self.article_input.pack(fill="both", expand=True, padx=8, pady=8)

        tk.Button(c_i, text="🔍  ANALYZE ARTICLE", font=FONT_H2, fg=BG_DARK, bg=BLUE, command=self._analyze_article, relief="flat").pack(fill="x", padx=8, pady=8)

        w_r, c_r = self._card(f, "▸ PREDICTION RESULT")
        w_r.pack(fill="x", pady=4)
        self.lbl_verdict = tk.Label(c_r, text="—", font=("Consolas", 28, "bold"), fg=TXT_DIM, bg=BG_CARD)
        self.lbl_verdict.pack(anchor="w", padx=10)
        self.lbl_confidence = tk.Label(c_r, text="Confidence: —", font=FONT_H2, fg=TXT_DIM, bg=BG_CARD)
        self.lbl_confidence.pack(anchor="w", padx=10, pady=4)

        return f

    def _refresh_db_status(self):
        if self.state_obj.db.online: self.lbl_db_status.config(text="● DB: Connected", fg=GREEN)
        else: self.lbl_db_status.config(text="● DB: Offline", fg=YELLOW)

    def _start_training_thread(self):
        self.status_var.set("⏳ Training model background thread…")
        threading.Thread(target=self._train_model, daemon=True).start()

    def _train_model(self):
        try:
            df = generate_synthetic_dataset(n_per_class=800)
            X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
            
            X_train_c = X_train.apply(clean_text)
            X_test_c  = X_test.apply(clean_text)

            pipeline = build_new_pipeline()
            pipeline.fit(X_train_c, y_train)
            acc = accuracy_score(y_test, pipeline.predict(X_test_c))

            self.state_obj.model = pipeline
            self.state_obj.run_metrics = {"total": len(df), "acc": acc}
            self.after(0, self._on_training_complete)
        except Exception as exc:
            self.after(0, lambda: self.status_var.set(f"❌ Error: {exc}"))

    def _on_training_complete(self):
        m = self.state_obj.run_metrics
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("end", f"TRAINING DONE\nTotal Rows: {m['total']}\nTest Accuracy: {m['acc']*100:.2f}%\n")
        self.metrics_text.config(state="disabled")
        self.status_var.set("✅ Model Trained successfully!")

    def _analyze_article(self):
        if self.state_obj.model is None:
            messagebox.showwarning("No Model", "Train model first.")
            return
        txt = self.article_input.get("1.0", "end-1c").strip()
        cleaned = clean_text(txt)
        
        proba = self.state_obj.model.predict_proba([cleaned])[0]
        prob_dict = dict(zip(self.state_obj.model.classes_, proba))
        
        prediction = "REAL" if prob_dict.get("REAL", 0) > prob_dict.get("FAKE", 0) else "FAKE"
        confidence = prob_dict[prediction] * 100

        color = RED if prediction == "FAKE" else GREEN
        self.lbl_verdict.config(text=f"🚨 {prediction}" if prediction == "FAKE" else f"✅ {prediction}", fg=color)
        self.lbl_confidence.config(text=f"Confidence: {confidence:.1f}%", fg=color)
        
        self.state_obj.session_logs.append({"prediction": prediction, "confidence": confidence})
        self.state_obj.db.insert_prediction(txt, prediction, confidence / 100)

    def _drop_model(self):
        self.state_obj.model = None
        self.lbl_verdict.config(text="—", fg=TXT_DIM)
        self.status_var.set("Weights dropped.")

    def _show_chart(self):
        if not self.state_obj.session_logs: return
        df = pd.DataFrame(self.state_obj.session_logs)
        df["prediction"].value_counts().plot(kind="bar", color=[GREEN, RED])
        plt.title("Session Predictions")
        plt.show()