import os
import aiofiles
import asyncpg
import fitz  
import pandas as pd
import numpy as np
import asyncio
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine
from sentence_transformers import SentenceTransformer, util
from langchain_community.llms import HuggingFaceHub
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from werkzeug.utils import secure_filename
from pgvector.asyncpg import register_vector
import logging
import Document_Q_A_Application.config as config
from Document_Q_A_Application import app

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "api_key"

embedding_model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-v2')

llm_model = HuggingFaceHub(repo_id="Qwen/Qwen1.5-32B-Chat")

executor = ThreadPoolExecutor(max_workers=5)

engine = create_engine(config.SQLALCHEMY_DATABASE_URI)
metadata = MetaData()
metadata.reflect(bind=engine)

selected_documents = []
available_documents = []

async def get_user_db_connection(user_id):
    user_db_name = f"user_{user_id}_db"
    return await asyncpg.connect(
        database=user_db_name,
        user=config.DB_USERNAME,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT
    )

async def create_user_database(user_id):
    conn = await asyncpg.connect(
        database="postgres",
        user=config.DB_USERNAME,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT
    )
    db_name = f"user_{user_id}_db"

    existing_db = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
    if not existing_db:
        await conn.execute(f'CREATE DATABASE "{db_name}"')

    await conn.close()

    user_conn = await asyncpg.connect(
        database=db_name,
        user=config.DB_USERNAME,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT 
    )

    await user_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    await user_conn.close()

async def store_document_in_db(user_id, filename, content, embeddings):
    try:
        conn = await get_user_db_connection(user_id)
        await register_vector(conn)

        table_name = filename.replace(".", "_")

        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(768)
            )
        """)

        await conn.execute(f"""
            INSERT INTO {table_name} (content, embedding)
            VALUES ($1, $2)
        """, content, embeddings.tolist())

        await conn.close()
    except Exception as e:
        return str(e)

def generate_embeddings(content):
    return embedding_model.encode(content)

def extract_text(filepath):
    if filepath.lower().endswith(".pdf"):
        text = ""
        pdf = fitz.open(filepath)
        for page in pdf:
            text += page.get_text("text")
        return text
    elif filepath.lower().endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None

async def get_db_connection():
    return await asyncpg.connect(
        database="postgres",
        user=config.DB_USERNAME,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT
    )


async def retrieve_relevant_tables(query_embedding, top_percent=0.4):
    conn = await get_db_connection()
    await register_vector(conn)

    table_names = list(metadata.tables.keys())
    table_similarities = []

    for table_name in table_names:
        query = f"SELECT embedding FROM {table_name} LIMIT 1"
        doc_embedding = await conn.fetchval(query)

        if doc_embedding is None:
            continue

        doc_embedding = np.array(doc_embedding, dtype=np.float32)
        similarity = util.pytorch_cos_sim(query_embedding, doc_embedding)[0].item()

        table_similarities.append((similarity, table_name))

    await conn.close()

    if not table_similarities:
        return []

    table_similarities.sort(reverse=True, key=lambda x: x[0])

    k = max(1, int(len(table_similarities) * top_percent))
    return [table[1] for table in table_similarities[:k]]


async def retrieve_table_content(tables):
    conn = await get_db_connection()
    doc_contents = []

    for table in tables:
        query = f"SELECT content FROM {table} LIMIT 1"
        result = await conn.fetchval(query)
        if result:
            doc_contents.append(result)

    await conn.close()
    return doc_contents


def generate_answer(retrieved_docs_content, user_query):
    context = "\n\n".join(retrieved_docs_content)

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        RunnablePassthrough.assign(context=context, question=user_query)
        | prompt
        | llm_model
    )

    return rag_chain.invoke({})

@app.route('/')
@app.route('/upload', methods=['POST'])
def upload_document():
    if not request.content_type or 'multipart/form-data' not in request.content_type:
        return jsonify({'error': 'Content-Type must be multipart/form-data'}), 400

    if 'file' not in request.files or 'user_id' not in request.form:
        return jsonify({'error': 'Missing file or user_id'}), 400
    
    file = request.files['file']
    user_id = request.form['user_id']

    conn = asyncpg.connect(
        database="postgres",
        user=config.DB_USERNAME,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT
    )

    user_db = f"user_{user_id}_db"

    existing_user = conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", user_db)

    conn.close()

    if existing_user:
        return jsonify({'error': 'User already exists. Pick a different user name.'})

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_contents = file.read()
    file.seek(0)

    if not file_contents:
        return jsonify({'error': 'Uploaded file is empty'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    content = extract_text(filepath)
    if content is None:
        return jsonify({'error': 'Unsupported file format. Only PDF and TXT are supported.'}), 400

    embeddings = executor.submit(generate_embeddings, content).result()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(create_user_database(user_id))

    error = loop.run_until_complete(store_document_in_db(user_id, filename, content, embeddings))
    if error:
        return jsonify({'error': error}), 500

    return jsonify({'message': 'Document uploaded and stored successfully', 'table': filename.replace(".", "_")})

@app.route('/query', methods=['POST'])
def retrieve_documents():
    logging.info(f"Request headers: {request.headers}")     
    logging.info(f"Request data: {request.data}")           

    global available_documents

    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json()
    if 'query' not in data:
        return jsonify({'error': 'Please provide the question'}), 400

    user_query = request.json['query']

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    query_embedding = loop.run_in_executor(None, embedding_model.encode, user_query)

    available_documents = loop.run_until_complete(retrieve_relevant_tables(query_embedding))

    if not available_documents:
        return jsonify({'message': 'No relevant documents found.'})

    return jsonify({
        'query': user_query,
        'available_documents': available_documents
    })

@app.route('/select_documents', methods=['POST'])
def select_documents():
    global selected_documents
    data = request.json

    is_subset = set(selected_documents).issubset(set(available_documents))

    if ('tables' not in data) or (not isinstance(data['tables'], list)) or (available_documents == []) or (not is_subset):
        return jsonify({'error': 'Invalid input, expected list of table names'}), 400

    selected_documents = data['tables']

    return jsonify({'message': 'Documents selected successfully', 'selected_documents': selected_documents})


@app.route('/selected_documents', methods=['GET'])
def get_selected_documents():
    return jsonify({'selected_documents': selected_documents})


@app.route('/selected_documents', methods=['DELETE'])
def clear_selected_documents():
    global selected_documents
    selected_documents = []
    return jsonify({'message': 'All selected documents have been cleared'})


@app.route('/generate_answer', methods=['POST'])
def rag_question_answering():
    global selected_documents, available_documents

    if 'query' not in request.json:
        return jsonify({'error': 'Missing query'}), 400

    user_query = request.json['query']

    if not selected_documents:
        selected_documents = available_documents

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    retrieved_docs_content = loop.run_until_complete(retrieve_table_content(selected_documents))

    if not retrieved_docs_content:
        return jsonify({'message': 'No relevant content found in selected documents.'})

    answer = generate_answer(retrieved_docs_content, user_query)

    return jsonify({
        'query': user_query,
        'answer': answer,
        'documents_used': selected_documents
    })      
