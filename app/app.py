import sys
import os
import re
from markupsafe import Markup
from flask import Flask, render_template, request, jsonify

# 🔗 Ajout du chemin vers le dossier rag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_chat import get_answer

app = Flask(__name__)

# 🔍 Filtre personnalisé pour rendre les liens cliquables
@app.template_filter('autolink')
def autolink(text):
    url_pattern = r'(https?://\S+)'
    return Markup(re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text))

# 🆕 Nouveau filtre pour convertir \n en <br>
@app.template_filter('nl2br')
def nl2br(text):
    if text:
        return Markup(text.replace("\n", "<br>"))
    return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    question = request.json.get("question")
    print("❓ Question reçue :", question)

    try:
        answer = get_answer(question)
        print("💬 Réponse générée :", answer)
        
        # 🆕 On renvoie directement l'answer (traitée dans le template)
        return jsonify({"response": answer})
    except Exception as e:
        print("⚠️ Erreur :", e)
        return jsonify({"response": "Désolé, une erreur est survenue. 😓"})

if __name__ == "__main__":
    app.run(debug=True)
