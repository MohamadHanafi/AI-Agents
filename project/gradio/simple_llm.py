# Import the necessary packages
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM

# Gradio
import gradio as gr

# env
import os
from pathlib import Path
from dotenv import load_dotenv


env_file_path = Path.cwd().parent.parent / '.env'

# Load environment variables from .env file
load_dotenv(env_file_path)

api_key = os.getenv("IBM_WATSON_API_KEY")
projectId = os.getenv("IBM_WATSON_PROJECT_ID")
url = os.getenv("IBM_WATSON_URL")

# Specify the model and project settings 
# (make sure the model you wish to use is commented out, and other models are commented)
model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503' # Specify the Mixtral model
# model_id = 'meta-llama/llama-3-2-11b-vision-instruct' # Specify llama model

# Set the necessary parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # Specify the max tokens you want to generate
    GenParams.TEMPERATURE: 0.5, # This randomness or creativity of the model's responses
}

# Wrap up the model into WatsonxLLM inference
watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url=url,
    project_id=projectId,
    params=parameters,
    apikey=api_key
)

# Function to generate a response from the model
def generate_response(prompt_txt):
    generated_response = watsonx_llm.invoke(prompt_txt)
    return generated_response

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
	allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860)

