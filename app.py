# Banoqabil Assistant - Assignment 02 - RAG App
# Knowledgebase: banoqabil.pk website
# Embedding: all-MiniLM-L6-v2, Vector DB: Pinecone

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import requests
from bs4 import BeautifulSoup

PINECONE_KEY = "APNI_PINECONE_KEY_YAHAN"
GROQ_KEY = "APNI_GROQ_KEY_YAHAN"

urls = ["https://banoqabil.pk/", "https://banoqabil.pk/about-us", "https://banoqabil.pk/courses"]
all_text = ""
for url in urls:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        all_text += soup.get_text() + "\n"
    except:
        pass

chunks = [all_text[i:i+1000] for i in range(0, len(all_text), 1000)]

model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=PINECONE_KEY)
index_name = "banoqabil-assistant"
if index_name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})
index = pc.Index(index_name)

vectors = []
for i, chunk in enumerate(chunks[:50]):
    emb = model.encode(chunk).tolist()
    vectors.append((f"chunk-{i}", emb, {"text": chunk}))
index.upsert(vectors=vectors)

client = Groq(api_key=GROQ_KEY)
print("Banoqabil Assistant Ready!")

while True:
    question = input("Tum: ")
    if question.lower() == "bye":
        break
    q_emb = model.encode(question).tolist()
    results = index.query(vector=q_emb, top_k=3, include_metadata=True)
    context = "\n".join([m['metadata']['text'] for m in results['matches']])
    prompt = f"Sirf is context se jawab do, agar jawab context me na ho to bolo 'Mujhe Banoqabil ki website me ye maloomat nahi mili'. Context: {context} Sawal: {question}"
    res = client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role": "user", "content": prompt}])
    print("Assistant:", res.choices[0].message.content)
