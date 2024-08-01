import os
from together import Together

client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=["""Noah is a architect. He likes to keep a weekly journal, documenting:

1Family member birthday: husband (yourself) (birthday on October 17), wife (birthday on August 26), child_1 (birthday on August 20), father (birthday on September 16), mother (birthday on March 25),
2) Sailing along the Mediterranean coast in week 41-42.
3) Attend golf lessons at the local club every 2 weeks on weekends.

Generate a complete weekly diary for Noah for the entire year of 2018. Start from January 1st, a Monday, marking the first week, and continue through to December 31st, the end of the 52nd week. Ensure that the diary consists of 52 entries, one for each week. Each entry should contain at least 150 words. Use '###' to separate each weekly entry like ### Week 1 (January 1st - January 7th), ensuring clarity and continuity without any interruptions or omissions in the narrative throughout the year."""],
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1,
    stop=["<|eot_id|>"],
    stream=True
)
print(response.choices[0].message.content)