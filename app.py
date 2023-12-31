# Import the Flask module and other necessary libraries
from flask import Flask, request, jsonify
from transformers import AutoTokenizer,GenerationConfig, AutoModelForSeq2SeqLM,Seq2SeqTrainer, Seq2SeqTrainingArguments,DataCollatorForSeq2Seq
from huggingface_hub import login
import re
import torch
from peft import PeftModel, PeftConfig
import openai
from dotenv import load_dotenv
import os
import time
from flask_cors import CORS, cross_origin

load_dotenv()
# Initialize Flask app
app = Flask(__name__)
CORS(app,supports_credentials=True)

# Perform Hugging Face Hub login
with open ('huggingface.txt','r') as file:
  huggingface_token=file.read().strip()
login(token=huggingface_token)

# Load peft config for pre-trained checkpoint etc.
peft_model_health = "nitsw/results"
config = PeftConfig.from_pretrained(peft_model_health)


# # load LLM model and tokenizer
model_health = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_health)
model.eval()

# Load peft config for pre-trained checkpoint etc.
peft_model_finance = "nitsw/finance"
config = PeftConfig.from_pretrained(peft_model_finance)

# load LLM models and tokenizer
model_finance = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model_finance = PeftModel.from_pretrained(model_finance, peft_model_finance)
model_finance.eval()

print("Peft health & finance model loaded")
type = os.environ.get('TYPE')
openai.api_type = type
base = os.environ.get('BASE')
openai.api_base = base
version = os.environ.get('VERSION')
openai.api_version = version
key = os.environ.get('KEY')
os.environ["OPENAI_API_KEY"] = key
openai.api_key = key

def remove_repeated_phrases_and_sentences(text):
  # Split the text into sentences
  sentences = re.split(r'\.\s*', text)

  # Initialize a set to keep track of unique sentences
  unique_sentences = set()

  # Initialize a list to store non-repeated sentences
  non_repeated_sentences = []

  for sentence in sentences:
      # Check if the sentence is not already in the set of unique sentences
      if sentence not in unique_sentences:
          unique_sentences.add(sentence)
          non_repeated_sentences.append(sentence)

  # Join the non-repeated sentences back together with periods
  cleaned_text = '. '.join(non_repeated_sentences)

  # Split the cleaned text into phrases
  phrases = re.split(r'\s*-\s*', cleaned_text)

  # Initialize a set to keep track of unique phrases
  unique_phrases = set()

  # Initialize a list to store non-repeated phrases
  non_repeated_phrases = []

  for phrase in phrases:
      # Check if the phrase is not already in the set of unique phrases
      if phrase not in unique_phrases:
          unique_phrases.add(phrase)
          non_repeated_phrases.append(phrase)

  # Join the non-repeated phrases back together with hyphens
  result = ' - '.join(non_repeated_phrases)

  last_full_stop_index = result.rfind('.')
  if last_full_stop_index != -1:
    result = result[:last_full_stop_index + 1]
    # print(trimmed_text)

  return result

# Define the finance endpoint
@app.route('/finance', methods=['POST'])
@cross_origin()
def finance_endpoint():
    try:
        # Get the input text from the request JSON
        input_text = request.json['text']

        # Define the health function

        context = "The finance industry include all the things related to Banking, Financial Services , Insurance,  Budget, Investment, Portfolio, Assets, Liabilities, Stocks, Bonds, Capital, Interest, Credit, Risk, Banking, Insurance, Taxes, Mortgage, Retirement, Inflation, Dividend, Asset Allocation and Derivatives"
        def finance(question):
          # Create a prompt to determine if the question is related to healthcare
          prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: Is {question} related to finance?"
          # prompt = f" Answer the following question in yes/no Context: {context}\n\nQuestion: {question}\n\nAnswer: Is {question} related to healthcare?"

          input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
          instruct_model_outputs = model_finance.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=10, num_beams=1))
          domain = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
          start=time.time()
          input_ids = tokenizer(question, return_tensors="pt",truncation=True).input_ids
          input_token_count=len(input_ids[0])
          if "yes" in domain.lower() :
            instruct_model_outputs = model_finance.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
            Predicted_Answer = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
            end=time.time()
            result = remove_repeated_phrases_and_sentences(Predicted_Answer)
            result_ids = tokenizer(result, return_tensors="pt",truncation=True).input_ids
            output_token_count=len(result_ids[0])
            total_token_count=input_token_count+output_token_count
            execution_time=end-start
            return result,execution_time,input_token_count,output_token_count,total_token_count

          else:
            result="The question is not related to finance."
            end=time.time()
            execution_time=end-start
            result_ids = tokenizer(result, return_tensors="pt",truncation=True).input_ids
            output_token_count=len(result_ids[0])
            total_token_count=input_token_count+output_token_count
            return result,execution_time,input_token_count,output_token_count,total_token_count

        # Call the finance function with the input text
        result = finance(input_text)

        # Return the result as JSON
        return jsonify({'result': result[0],'execution_time':result[1],'input_token_used':result[2],'output_token_used':result[3],'total_token_used':result[4]})
    except Exception as e:
        return jsonify({'error': str(e)})
    

# Define the health endpoint
@app.route('/health', methods=['POST'])
@cross_origin()
def health_endpoint():
    try:
        # Get the input text from the request JSON
        input_text = request.json['text']

        # Define the health function
        context = "The healthcare industry include all the things realted to Medical treatments and procedures, Healthcare providers (doctors, nurses, etc.),Healthcare facilities (hospitals, clinics, etc.),Health insurance,Healthcare technology and innovations, Public health, Healthcare policies and regulations, Pharmaceutical industry,Health and wellness,Patient care and advocacy"

        def health(question):
            # Create a prompt to determine if the question is related to healthcare
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: Is {question} related to healthcare?"
            # prompt = f" Answer the following question in yes/no Context: {context}\n\nQuestion: {question}\n\nAnswer: Is {question} related to healthcare?"

            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
            instruct_model_outputs = model_health.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=10, num_beams=1))
            domain = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

            start=time.time()
            input_ids = tokenizer(question, return_tensors="pt",truncation=True).input_ids
            input_token_count=len(input_ids[0])            
            if domain.lower() == "yes":
              input_ids = tokenizer(question, return_tensors="pt",truncation=True).input_ids
              instruct_model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
              Predicted_Answer = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
              end=time.time()
              result = remove_repeated_phrases_and_sentences(Predicted_Answer)
              result_ids = tokenizer(result, return_tensors="pt",truncation=True).input_ids
              output_token_count=len(result_ids[0])
              total_token_count=input_token_count+output_token_count
              execution_time=end-start
              return result,execution_time,input_token_count,output_token_count,total_token_count
              # # print(f"Predicted Answer:{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

            else:
              result="The question is not related to healthcare."
              result_ids = tokenizer(result, return_tensors="pt",truncation=True).input_ids
              output_token_count=len(result_ids[0])
              total_token_count=input_token_count+output_token_count
              end=time.time()
              execution_time=end-start
              return result,execution_time,input_token_count,output_token_count,total_token_count


        # Call the health function with the input text
        result = health(input_text)

        # Return the result as JSON
        return jsonify({'result': result[0],'execution_time':result[1],'input_token_used':result[2],'output_token_used':result[3],'total_token_used':result[4]})
    except Exception as e:
        return jsonify({'error': str(e)})
      
# Define the general endpoint
@app.route('/general', methods=['POST'])
@cross_origin()
def general_endpoint():
    try:
        # Get the input text from the request JSON
        input_text = request.json['text']
      
        # Define the general chatbot function
        def chatbot(query):    
          start=time.time()
          response = openai.ChatCompletion.create(
              engine="gpt-35-turbo", # replace this value with the deployment name you chose when you deployed the associated model.
              messages=[{"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                  {"role": "user", "content": query}],
              temperature=0,
              max_tokens=1000,
              top_p=0.95,
              frequency_penalty=0,
              presence_penalty=0,
              stop=None)
          end=time.time()
          result =response['choices'][0]['message']['content']
          execution_time=end-start
          input_ids = tokenizer(query, return_tensors="pt",truncation=True).input_ids
          input_token_count=len(input_ids[0]) 
          output_ids = tokenizer(result, return_tensors="pt",truncation=True).input_ids
          output_token_count=len(input_ids[0])
          total_token_count=input_token_count + output_token_count
          return result,execution_time,input_token_count,output_token_count,total_token_count

        # Call the health function with the input text
        result = chatbot(input_text)
        # Return the result as JSON
        return jsonify({'result': result[0],'execution_time':result[1],'input_token_used':result[2],'output_token_used':result[3],'total_token_used':result[4]})
    except Exception as e:
        return jsonify({'error': str(e)})



# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
