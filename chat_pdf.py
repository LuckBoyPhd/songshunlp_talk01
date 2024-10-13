

def bot(context,input_text):

    from langchain_openai import ChatOpenAI
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    
    llm = ChatOpenAI(
        base_url = BASE_URL,
        api_key= API_KEY,
        model="glm-4",temperature=0.0)

    

    messages = [
                (
                    "assistant",
                    "你是一个中文和英文的专家，下面问题请用中文回答",
                ),
                ("human", f"请参考一下资料,{context},请用中文回答问题{input_text}"),
            ]
    ai_msg = llm.invoke(messages)

    return  ai_msg.content
    

def init_embedding():
    # init embedding model
    
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    embedding_model_name ='bce-embedding-base_v1'
    embedding_model_kwargs  = {'device': 'cpu'}
    embedding_encode_kwargs = {'normalize_embeddings': False}



    embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,model_kwargs=embedding_model_kwargs,encode_kwargs=embedding_encode_kwargs)
    
    return embed_model



def load_documents(pdfname):    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    raw_documents = PyPDFLoader(pdfname).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)
    
    return documents


#index = faiss.IndexFlatL2(len(embed_model.embed_query("hello world")))



def build_faiss_db(documents, embed_model):

    from langchain_community.vectorstores import FAISS   
    db = FAISS.from_documents(documents, embed_model)
    db.save_local("faiss_index_constitution")
    
    return db 


def query_from_db(db,query,k):
    results = db.similarity_search(
        db = db,
        query=query,
        k=k,
    )
    
    out = [res.page_content for  res in results]
    
    return out



if __name__ == "__main__":
    
    
    pdfname =  "llama2.pdf"
    embed_model = init_embedding()
    
    documents = load_documents(pdfname)
    db = build_faiss_db(documents, embed_model)
    
    query = "What is llama2?"

    k = 3
    
    out = query_from_db(db,query,k)
    
    context = '\n'.join( o for o in out)

    print("上下文：",context)
    response = bot(context,query)
    print('回答结果')
    print(response)