import numpy as np
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA

# DB_FAISS_PATH = 'vectorstore/db_faiss'

# custom_prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# def set_custom_prompt():
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt

# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=db.as_retriever(search_kwargs={'k': 2}),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={'prompt': prompt}
#                                        )
#     return qa_chain

# def load_llm():
#     llm = CTransformers(
#         model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
#         model_type="llama",
#         max_new_tokens = 512,
#         temperature = 0.8
#     )
#     return llm

# def qa_bot(query):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                        model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)
#     response = qa({'query': query})
#     return response['result']  # Return only the result text without metadata


@csrf_exempt
def llm_model(request):
    if request.method == 'POST':
        print("aa")
        print(request.POST.get('query'))
        # response = qa_bot(query)
    return JsonResponse({"message": "NO"})

# def validate(request):
#     print(request.GET)
#     print(request.POST)
#     print(request.body)
#     loaded_model = load_model("app1/validate.h5")
#     # # img_path = 'app1/uploaded_image.jpg'
#     # img_path = 'app1/image.jpg'
#     # img = image.load_img(img_path, target_size=(256, 256))
#     # img_array = image.img_to_array(img)
#     # img_array = np.expand_dims(img_array, axis=0)
#     # img_array /= 255.0
#     # predictions = loaded_model.predict(img_array)
#     # if predictions[0] < 0.5:
#     #     return JsonResponse({"ML": "YES"})
#     # print("ML NO")
#     return JsonResponse({"ML": "NO"})

