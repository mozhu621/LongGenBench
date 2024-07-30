import os
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

stream = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  messages=[{"role": "user", "content": """Olivia is a economist. He likes to keep a weekly journal, documenting: 
             1) Family life with: husband(yourself) (birthday on November 21), wife (birthday on June 24), child_1:daughter (birthday on August 19), father (birthday on August 18), mother (birthday on June 07). 
             2) Participating in a pottery and ceramics workshop in Japan in week 40.
             3) Join a weekend warriors’ adventure club every 4 weeks on weekends.
             Generate a weekly diary for the year 2018, starting from January 1st, which is a Monday, marking the beginning of the first week of the year. Continue through to December 31st, which completes the 52nd week. Each diary entry should correspond to one week, resulting in a total of 52 entries. Each diary entry must consist of at least 150 words, thoroughly documenting personal experiences, thoughts, and significant events of the week. Conclude each entry with a brief reflection on the upcoming week, and separate each weekly diary entry with '###' to clearly demarcate the end of one week and the start of the next. Please ensure that there are no interruptions or omissions in the sequence of diary entries, providing a continuous narrative throughout the year.
"""}],
    max_tokens=8072,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1,
    stop=["<|eot_id|>"],
    stream=True,
)

for chunk in stream:
  print(chunk.choices[0].delta.content or "", end="", flush=True)