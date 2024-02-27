import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import (
                                                        AzureSearch,
                                                        AzureSearchVectorStoreRetriever
                                                        )
from ragas.llms import LangchainLLM
from ragas import evaluate
from ragas.metrics.critique import harmfulness
from datasets import load_dataset


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["azure_endpoint"] = "https://ai-dev-openai-fts.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "7439c61b0745446dbbb029521f498265"

azure_model = AzureChatOpenAI(
    openai_api_key = "7439c61b0745446dbbb029521f498265",
    azure_endpoint = "https://ai-dev-openai-fts.openai.azure.com/",
    openai_api_version = "2023-07-01-preview",
    openai_organization = "azure",
    temperature=0,    
    deployment_name="gpt35")

azure_embeddings = AzureOpenAIEmbeddings(
            openai_api_key = "7439c61b0745446dbbb029521f498265",
            azure_endpoint = "https://ai-dev-openai-fts.openai.azure.com/",
            openai_api_type = "azure",
            openai_api_version = "2023-07-01-preview",
            deployment="text-embedding-ada-002", 
            chunk_size=1
            )


acs = AzureSearch(azure_search_endpoint="https://ai-dev-ss-fts.search.windows.net",
                 azure_search_key="ICyDmM0S3sWaLpYwNGPfUOz7jblRyOHnE6p7uDHPRRAzSeAaLBFM",
                 index_name="reg-data-ankur",
                 embedding_function= azure_embeddings.embed_query
                 )


retriever = acs.as_retriever(search_kwargs={"k": 3})

from langchain import PromptTemplate

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = PromptTemplate(
    template=template, 
    input_variables=["context","question"]
  )


questions = ["What did the president say about Justice Breyer?", 
                 "What did the president say about Intel's CEO?",
                 "What did the president say about gun violence?",
                ]


ground_truths = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                    ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
                    ["The president asked Congress to pass proven measures to reduce gun violence."]]
   



def rag_evaluation(retriever, prompt, azure_model, questions, ground_truths):
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt 
        | azure_model
        | StrOutputParser() 
    )

    from datasets import Dataset

    questions = questions
    ground_truths = ground_truths


    answers = []
    contexts = []

    # Inference
    for query in questions:
        answers.append(rag_chain.invoke(query))
        contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

    # To dict
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    from ragas.metrics import (
        context_precision,
        answer_relevancy,
        faithfulness,
        context_recall,
        )
    from ragas.metrics.critique import harmfulness

    # list of metrics we're going to use
    metrics = [
         faithfulness,
         answer_relevancy,
         context_recall,
         context_precision,
         harmfulness,
         ]

    # wrapper around azure_model
    ragas_azure_model = LangchainLLM(azure_model)
    # patch the new RagasLLM instance
    answer_relevancy.llm = ragas_azure_model

    # embeddings can be used as it is
    answer_relevancy.embeddings = azure_embeddings

    for m in metrics:
        m.__setattr__("llm", ragas_azure_model)


    result = evaluate(dataset, metrics=metrics)
    
    df = result.to_pandas()

    return df

