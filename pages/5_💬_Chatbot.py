
import time
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import os
import uuid
import time
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="centered"
)
key = os.getenv('PRIVATEKEY2')
key = key.replace('\\n', '\n')
if not firebase_admin._apps:
    jeyson = {
    "type": "service_account",
    "project_id": "inter-viewer",
    "private_key_id": os.getenv('PRIVATEKEY1'),
    "private_key": key,
    "client_email": "firebase-adminsdk-61ffn@inter-viewer.iam.gserviceaccount.com",
    "client_id": "112443602599283296327",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-61ffn%40inter-viewer.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
    }
    cred = credentials.Certificate(jeyson)
    firebase_admin.initialize_app(cred)

# Inicializar Firestore
db = firestore.client()
def generar_id_documento():
    id_aleatorio = uuid.uuid4()
    return str(id_aleatorio)
def leer_respuesta_de_firestore(doc_id):
    while True:
        doc_ref = db.collection('conversations').document(doc_id)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            respuesta = data.get('response')
            if respuesta:
                return respuesta
        else:
            print(f'El documento {doc_id} no existe.')
            return None
        
        time.sleep(0.5)
def crear_prompt_en_firestore(prompt):
    # Generar un ID único
    id_documento = generar_id_documento()
    
    # Referencia al documento utilizando el ID generado
    doc_ref = db.collection('conversations').document(id_documento)
    
    # Agregar el documento a la colección con el ID generado
    doc_ref.set({
        'prompt': prompt,
        'response': None,  # Inicialmente no hay respuesta
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    
    return id_documento
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is Storymodelers?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        id = crear_prompt_en_firestore(prompt)
        response = leer_respuesta_de_firestore(id)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
        