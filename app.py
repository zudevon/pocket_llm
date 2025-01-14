from flask import Flask, request, render_template, session, redirect, url_for, flash, jsonify
import os
import fitz  # PyMuPDF for PDF text extraction
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
import re
import string

from utils.clean_text import extract_sentences_spacy, clean_text_chunks, extract_clean_sentences

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config["UPLOAD_FOLDER"] = "uploads/"

# Initialize embedding model and storage
model = SentenceTransformer('all-MiniLM-L6-v2')
pdf_text_data = []  # To store text chunks
embeddings = []  # To store embeddings for similarity search
index = None  # NearestNeighbors model for searching embeddings

open_ai_key = "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"

# chat_gpt_client = OpenAI(api_key=open_ai_key)
openai.api_key = open_ai_key


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text())
    return " ".join(text)


# Define your 6-digit password
PASSWORD = "691691"

# Landing Page Route
@app.route('/', methods=['GET', 'POST'])
def landing():
    if request.method == 'POST':
        entered_password = request.form['password']
        if entered_password == PASSWORD:
            return redirect(url_for('welcome'))
        else:
            flash('Invalid password. Please try again.')
            return redirect(url_for('landing'))
    return render_template('landing.html')


@app.route("/main", methods=["GET"])
def welcome():
    session.clear()  # Clear session data
    return render_template("welcome.html")

# Route for handling PDF upload
@app.route("/upload", methods=["POST"])
def upload():
    pdf_file = request.files["pdf"]
    if pdf_file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
        pdf_file.save(file_path)

        # Extract text and update embeddings
        text = extract_text_from_pdf(file_path)
        # text_chunks = clean_text_chunks(text)
        # text_chunks = extract_sentences_spacy(text)
        text_chunks = extract_clean_sentences(text)

        global embeddings, index
        for chunk in text_chunks:
            pdf_text_data.append(chunk)
            embedding = model.encode(chunk)
            embeddings.append(embedding)

        # Update NearestNeighbors index with appropriate `n_neighbors`
        n_neighbors = min(5, len(embeddings))
        index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        index.fit(np.array(embeddings))

        return redirect(url_for("interact"))
    return redirect(url_for("welcome"))

# Route for the interaction page to ask questions or upload more PDFs
@app.route("/interact", methods=["GET", "POST"])
def interact():
    answer = None
    if request.method == "POST":
        if 'pdf' in request.files:
            # Handle new PDF upload
            pdf_file = request.files["pdf"]
            if pdf_file:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
                pdf_file.save(file_path)

                # Process new PDF text and update embeddings
                text = extract_text_from_pdf(file_path)
                # text_chunks = clean_text_chunks(text)
                # text_chunks = extract_sentences_spacy(text)
                text_chunks = extract_clean_sentences(text)

                global embeddings, index
                for chunk in text_chunks:
                    pdf_text_data.append(chunk)
                    embedding = model.encode(chunk)
                    embeddings.append(embedding)

                # Reinitialize NearestNeighbors with updated n_neighbors
                n_neighbors = min(5, len(embeddings))
                index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
                index.fit(np.array(embeddings))

                return redirect(url_for("interact"))

        elif 'question' in request.form:
            question = request.form.get("question")
            question_embedding = model.encode(question).reshape(1, -1)

            # Retrieve the top matches for the question
            distances, indices = index.kneighbors(question_embedding)
            top_chunks = [pdf_text_data[idx] for idx in indices[0]]

            # Prepare context for LLM to generate an answer
            context = " ".join(top_chunks)
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

            """TEST QUESTION : what is a hierarchical recurrent network"""
            # shitty job of processing chunks on a large pdf
            response = openai.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" depending on the model you want to use
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].to_dict()['message']['content']

    return render_template("interact.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
