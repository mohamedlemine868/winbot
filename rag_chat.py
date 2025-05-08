from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
import google.generativeai as genai
import os

# Charger la clé API Gemini
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Créer le modèle Gemini
model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

def get_answer(question):
    # 1. Gestion des salutations
    greetings = ["bonjour", "salut", "hello", "bonsoir", "hi"]
    if question.lower().strip() in greetings:
        return "Bonjour 👋 ! Je suis WinBot. Pose-moi une question sur WinCompta ou WinHub."

    print("🔍 Chargement des embeddings et de l'index FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("rag/faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS index chargé")

    # 2. Chercher les documents pertinents
    retriever = db.as_retriever()
    docs = retriever.invoke(question)
    print(f"📄 Documents pertinents trouvés : {len(docs)}")

    if not docs:
        return "Je n’ai rien trouvé dans mes documents..."

    print("🧠 Extrait du premier document :", docs[0].page_content[:200])

    # 3. Construire la question enrichie pour Gemini
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""
Tu es un assistant expert des logiciels WinCompta et WinHub.

Ta mission :
- Utilise uniquement les informations du contexte ci-dessous pour répondre.
- Si le contexte mentionne un tutoriel vidéo ou un lien, affiche-le clairement à la fin sous la forme :
  🎥 Tutoriel : [titre ou description] - [lien]
- Si aucun tutoriel n'est mentionné, n'invente rien.
- Sois clair, structuré, et précis.

Contexte :
{context}

Question :
{question}

Réponds maintenant :
"""

    # 4. Envoyer la requête à Gemini
    response = model.generate_content(prompt)

    if response and response.candidates:
        final_answer = response.candidates[0].content.parts[0].text
        return final_answer.strip()
    else:
        return "Je n'ai pas pu obtenir de réponse de Gemini..."

if __name__ == "__main__":
    print("🤖 Bienvenue sur WinBot RAG ! Pose ta question ou tape 'exit' pour quitter.\n")
    while True:
        query = input("❓ Vous: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Au revoir !")
            break
        try:
            answer = get_answer(query)
            print(f"💬 WinBot: {answer}\n")
        except Exception as e:
            print(f"⚠️ Erreur: {e}\n")















           






