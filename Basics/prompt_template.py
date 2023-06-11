from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0.9, max_tokens=300)

prompt = "Is life on Mars is possible?"

response = llm(prompt)

print(response)