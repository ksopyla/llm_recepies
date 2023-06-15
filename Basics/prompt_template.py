# %%
from dotenv import load_dotenv
from langchain.llms import OpenAI

# %%
load_dotenv()

from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

question = "Describe in 5 or less sentences the Pearson comapny, its history and its products and values."

# %%
llm_openai = OpenAI(model_name="text-davinci-003", temperature=0.9, max_tokens=500)

openai_chain = LLMChain(prompt=prompt, llm=llm_openai)

# %%
response = openai_chain.run(question=question)

print(response)
# %%

from langchain.llms import HuggingFaceHub

# "google/flan-t5-xl"

# crate HuggingFaceHub object with google flan t5 xl model

llm_hf = HuggingFaceHub(
    repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.1, "max_length": 1024}
)

hf_chain = LLMChain(prompt=prompt, llm=llm_hf)

# %%
response = hf_chain.run(question=question)

print(response)
# %%
