from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import matplotlib.pyplot as plt
import qrcode
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

app = Flask(__name__)

food_cache = {}

# -----------------------------
# LOAD DATASET
# -----------------------------
files = [
    "FOOD-DATA-GROUP1.csv",
    "FOOD-DATA-GROUP2.csv",
    "FOOD-DATA-GROUP3.csv",
    "FOOD-DATA-GROUP4.csv",
    "FOOD-DATA-GROUP5.csv"
]

dfs = [pd.read_csv(f) for f in files]
nutrition_df = pd.concat(dfs, ignore_index=True)
nutrition_df.columns = (
    nutrition_df.columns.astype(str)
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

FOOD_COL = "food"
CAL_COL = "caloric_value"
PRO_COL = "protein"
CARB_COL = "carbohydrates"
FAT_COL = "fat"

# -----------------------------
# ViT MODEL
# -----------------------------
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

def predict_food(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = vit_model(**inputs)

    idx = outputs.logits.argmax(-1).item()
    return vit_model.config.id2label[idx]

# -----------------------------
# NUTRITION CALCULATION
# -----------------------------
def calculate_nutrition(ingredients, serving):
    cal = pro = carb = fat = 0
    weight_each = serving / max(len(ingredients), 1)

    for ing in ingredients:
        ing = ing.strip().lower()
        row = nutrition_df[nutrition_df[FOOD_COL].str.lower().str.contains(ing, regex=False)]

        if not row.empty:
            cal += row[CAL_COL].values[0] * weight_each / 100
            pro += row[PRO_COL].values[0] * weight_each / 100
            carb += row[CARB_COL].values[0] * weight_each / 100
            fat += row[FAT_COL].values[0] * weight_each / 100

    return cal, pro, carb, fat

# -----------------------------
# SUMMARY
# -----------------------------
def generate_summary(food, ingredients, cal, p, c, f):
    return (
        "This dish provides balanced nutrition with moderate calories. "
        "Based on the ingredients, it includes healthy levels of protein, "
        "carbohydrates, and fats required for daily metabolic needs."
    )

# -----------------------------
# SAVE MACRO CHART (PNG)
# -----------------------------
def save_macro_chart(protein, carbs, fats):
    labels = ['Protein', 'Carbs', 'Fats']
    values = [protein, carbs, fats]
    colors = ['#00ff90', '#4da6ff', '#ffcc00']

    fig, ax = plt.subplots(figsize=(4,4))
    wedges, _ = ax.pie(values, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.35))
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set(aspect="equal")

    chart_path = "static/macro_chart.png"
    plt.savefig(chart_path, transparent=True)
    plt.close()

    return chart_path

# -----------------------------
# QR CODE GENERATOR
# -----------------------------
def generate_qr(url):
    qr_img = qrcode.make(url)
    qr_path = "static/qr.png"
    qr_img.save(qr_path)
    return qr_path

# -----------------------------
# MAIN PAGE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        img = request.files["image"]
        os.makedirs("static", exist_ok=True)
        image_path = "static/uploaded.jpg"
        img.save(image_path)

        title = request.form["title"]
        ingredients = request.form["ingredients"].split(",")
        serving = int(request.form["serving"])

        detected = predict_food(image_path)
        calories, p_g, c_g, f_g = calculate_nutrition(ingredients, serving)

        total = p_g + c_g + f_g
        p = (p_g / total) * 100 if total else 0
        c = (c_g / total) * 100 if total else 0
        f = (f_g / total) * 100 if total else 0

        summary = generate_summary(title, ingredients, calories, p_g, c_g, f_g)

        chart_path = save_macro_chart(p, c, f)

        # QR Code for PDF link
        qr_path = generate_qr("http://localhost:5000/download_pdf")

        # CACHE
        food_cache.update({
            "food": title,
            "recognized": detected,
            "calories": round(calories, 2),
            "protein": round(p, 2),
            "carbs": round(c, 2),
            "fats": round(f, 2),
            "summary": summary,
            "image": image_path,
            "chart": chart_path,
            "qr": qr_path
        })

        return render_template(
            "result.html",
            food=title,
            recognized=detected,
            calories=round(calories, 2),
            protein=round(p, 2),
            carbs=round(c, 2),
            fats=round(f, 2),
            summary=summary,
            image_path=image_path,
            qr_path=qr_path
        )

    return render_template("index.html")

# -----------------------------
# PDF GENERATOR
# -----------------------------
@app.route("/download_pdf")
def download_pdf():

    pdf_path = "static/report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, 750, f"Nutrition Report – {food_cache['food']}")

    # Food Image
    try:
        img = ImageReader(food_cache["image"])
        c.drawImage(img, 40, 520, width=200, height=200)
    except:
        c.drawString(40, 720, "[Image Error]")

    # Recognition
    c.setFont("Helvetica-Bold", 14)
    c.drawString(270, 700, "AI Recognition:")
    c.setFont("Helvetica", 12)
    c.drawString(270, 680, f"Detected: {food_cache['recognized']}")

    # Nutrition
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 470, "Nutrition Values:")
    c.setFont("Helvetica", 12)
    c.drawString(40, 450, f"Calories: {food_cache['calories']} kcal")
    c.drawString(40, 430, f"Protein: {food_cache['protein']}%")
    c.drawString(40, 410, f"Carbs: {food_cache['carbs']}%")
    c.drawString(40, 390, f"Fats: {food_cache['fats']}%")

    # Chart
    try:
        chart_img = ImageReader(food_cache["chart"])
        c.drawImage(chart_img, 300, 430, width=250, height=250)
    except:
        c.drawString(300, 600, "[Chart error]")

    # Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 350, "Summary:")
    text = c.beginText(40, 330)
    text.setFont("Helvetica", 11)
    for line in food_cache["summary"].split("\n"):
        text.textLine(line)
    c.drawText(text)

    c.save()
    return send_file(pdf_path, as_attachment=True)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)