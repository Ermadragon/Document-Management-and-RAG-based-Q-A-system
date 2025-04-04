Document Management and RAG-based Q&A Application

This is a Flask backend application which accepts documents from users and generates answers to questions through Retrieval Augmented Generation. The application uses PostgreSQL to store vector embeddings of uploaded documents. The vector embeddings are used to retrieve relevant documents through embedding similarity with the user question. Answers are generated based on the retrieved documents using LLM models such as Qwen1.5-32B-Chat.  

Required libraries
The libraries and models used in the application can be given as follows:
1. Sentence Transformers (for embedding generation)
     Model msmarco-roberta-base-v2 was used
2. pgvector extension of PostgreSQL is required for the application
3. asyncpg
4. asyncio
5. SQLAlchemy
6. PyMuPDF (for reading uploaded documents)
7. Langchain and HuggingFaceHub (for LLM models and for implementing RAG)
     Hugging Face model Qwen1.5-32B-Chat was used for generation

The different routes in the application are as follows: - 
1. GET method
     i. /selected_documents
2. POST methods
     i. /upload
     ii. /query
     iii. /select_documents
     iv. /generate_answer
3. DELETE method
     i. /selected_documents

The above mentioned routes can be tested by passing API requests through the requests library.

     import requests

url = "http://127.0.0.1:5555/upload"

The 5555 in the url provides the port on which the application runs. 
/upload represents the respective route. 

The different API requests for the different endpoints can be passed as follows: -

/upload:

For PDFs

      with open(r"E:\A1.pdf", 'rb') as f:
      files = {'file': ('A1.pdf', f, 'application/pdf')}                     
      data = {'user_id': 'A100'}                     
      response = requests.post(url, files=files, data=data)
                        
For .txt files

     with open(r"C:\Users\hp\Downloads\S1359836815003595.txt", 'rb') as f:
     files = {'file': ('S1359836815003595.txt', f, 'text/plain')}
     data = {'user_id': 'A100'}     
     response = requests.post(url, files=files, data=data)

/query:

     headers = {"Content-Type": "application/json"}
     data = {"query": "What is the meaning of life?"}

     response = requests.post(url, json=data, headers=headers)

Similarly API requests for the other endpoints can be given with the required data.     

Note: Replace the HuggingFaceHub token and PostgreSQL login password in config.py with the required values.
