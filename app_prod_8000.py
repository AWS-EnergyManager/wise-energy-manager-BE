import logging
import traceback

import os
import boto3
from langchain.chains import ConversationChain, RetrievalQA
from langchain.llms.bedrock import Bedrock
# from langchain_aws import BedrockLLM
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory

from langchain.retrievers.bedrock import AmazonKnowledgeBasesRetriever

from assistant.config import AgenticAssistantConfig
from assistant.prompts import CLAUDE_PROMPT, DOC_PROMPT
## placeholder for lab 3, step 4.2, replace this with imports as instructed


from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin

logger = logging.getLogger()
logger.setLevel(logging.INFO)

config = AgenticAssistantConfig()


app = Flask(__name__)
CORS(app)


def home():
    sample_data = {
        'message': 'Congratuation'
    }
    return jsonify(sample_data)


def get_basic_chatbot_conversation_chain(
    user_input, session_id, k, clean_history, verbose=True
):

    bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.bedrock_region)

    claude_llm = Bedrock(
        model_id=config.llm_model_id,
        client=bedrock_runtime,
        model_kwargs={"temperature": 0.0, "maxTokenCount": 500},
    )

    message_history = DynamoDBChatMessageHistory(
        table_name=config.chat_message_history_table_name, session_id=session_id
    )

    if clean_history:
        print("Cleaning history")
        message_history.clear()

    memory = ConversationBufferWindowMemory(
        memory_key="history",
        # Change the human_prefix from Human to something else
        # to not conflict with Human keyword in Anthropic Claude model.
        human_prefix="Hu",
        k=k,
        chat_memory=message_history,
        return_messages=False,
    )

    conversation_chain = ConversationChain(
        prompt=CLAUDE_PROMPT, llm=claude_llm, verbose=verbose, memory=memory
    )

    return conversation_chain


## placeholder for lab 3, step 4.3, replace this with the get_agentic_chatbot_conversation_chain helper.
def get_doc_chatbot_conversation_chain(
    user_input, session_id, k, clean_history, verbose=True
):

    bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.bedrock_region)

    claude_llm = Bedrock(
        model_id=config.llm_model_id,
        client=bedrock_runtime,
        model_kwargs={"temperature": 0.0, "maxTokenCount": 500},
    )


    KNOWLEDGEBASE_ID = config.kb_id
    if KNOWLEDGEBASE_ID is None:
        raise ValueError("KNOWLEDGEBASE_ID is not set in the environment")
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=KNOWLEDGEBASE_ID,
        region_name = "us-west-2",
        retrieval_config={"vectorSearchConfiguration": 
                            {"numberOfResults": 3,
                                'overrideSearchType': "SEMANTIC", # optional
                            }
                        },
    )

    qa = RetrievalQA.from_chain_type(
        llm=claude_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # chain_type_kwargs={"prompt": user_input},
    )

    result = qa.invoke(user_input)

    return result['result']

@app.route('/', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin()
def lambda_handler():

    # eample_event = {
    #     "user_input": "Hello",
    #     "session_id": "test",
    #     "chatbot_type": "basic",
    #     "clean_history": False
    # }clean_history
    user_input = request.json.get("user_input")
    session_id = request.json.get("session_id")
    chatbot_type = request.json.get("chatbot_type", "basic")
    clean_history = request.json.get("clean_history", False)
    chatbot_types = ["basic", "agentic"]
    k = request.json.get("k", 3)
    power_usage = request.json.get("power_usage", [])

    if chatbot_type == "basic":
        conversation_chain = get_basic_chatbot_conversation_chain(
            user_input, session_id, k, clean_history
        ).predict
    elif chatbot_type == "doc":
        result = get_doc_chatbot_conversation_chain(
            user_input, session_id, k, clean_history
        )
        return jsonify({
            "statusCode": 200,
            "response": result
        })
    elif chatbot_type == "agentic":
        return {
            "statusCode": 200,
            "response": (
                f"The agentic mode is not supported yet. Extend the code as instructed"
                " in lab 3 to add it."
            ),
        }
    else:
        return {
            "statusCode": 200,
            "response": (
                f"The chatbot_type {chatbot_type} is not supported."
                f" Please use one of the following types: {chatbot_types}"
            ),
        }

    try:
        response = conversation_chain(input=user_input)
    except Exception:
        response = (
            "Unable to respond due to an internal issue." " Please try again later"
        )
        print(traceback.format_exc())

    return {"statusCode": 200, "response": response}

def main():
    sample_event = {
        "user_input": "Hello",
        "session_id": "test",
        "chatbot_type": "basic",
        "clean_history": False
    }
    ret = lambda_handler(sample_event,{})
    print("ret =", ret)
    
if __name__ == '__main__':
    # main()
    app.run(debug=False, port=8000, host='0.0.0.0')