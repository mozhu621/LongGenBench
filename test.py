import random
from datetime import timedelta, date
import json
import time
# Helper functions
def random_date(start_month, end_month):
    """ Generate a random date within the specified months range. """
    start_date = date(2024, start_month, 1)
    end_date = date(2024, end_month, 28)  # Assume no month has more than 28 days for simplicity
    days_between_dates = (end_date - start_date).days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

def generate_week_number():
    """ Generate a random week number for the year (1-52). """
    return random.randint(10, 52)

def generate_person_details():
    """ Generate personal details including name and occupation. """
    names = ["Olivia", "Liam", "Emma", "Noah", "Ava", "William", "Sophia", "James", "Isabella", "Benjamin"]
    occupations = ["journalist", "architect", "engineer", "artist", "chef", "historian", "biologist", "economist", "nurse", "photographer"]
    return random.choice(names), random.choice(occupations)

def generate_family():
    """ Generate family details ensuring exactly 5 members including parents, spouse, and 1-3 children. """
    family = {"wife": "wife"}
    num_children = random.randint(1, 3)
    for i in range(num_children):
        child_type = "daughter" if i % 2 == 0 else "son"
        family[f"child_{i+1}"] = child_type
    
    # Ensure family size is 5; add parents if needed
    while len(family) < 4:
        if "father" not in family:
            family["father"] = "father"
        elif "mother" not in family:
            family["mother"] = "mother"
    
    return family

def generate_family_birthdays(family):
    """ Generate birthdays for each family member. """
    return {relation: random_date(1, 12).strftime("%B %d") for relation in family.keys()}

def generate_range_events_by_week():
    """ Define and select a random range event with a specific week number. """
    range_events = [
        "Family beach vacation in Maui",
        "Week-long road trip across the Pacific Coast Highway",
        "Skiing holiday in Aspen",
        "Cruise trip to the Caribbean",
        "Hiking and camping in the Rocky Mountains",
        "Visiting theme parks in Orlando",
        "Historical tour of Washington D.C.",
        "Exploring the wine country in Napa Valley",
        "Wildlife safari in Yellowstone National Park",
        "Scuba diving adventure in the Great Barrier Reef",
        "Attending a major music festival like Coachella",
        "Participating in a family yoga retreat",
        "Cultural and culinary exploration of New Orleans",
        "Participating in a week-long photography workshop",
        "Exploring ancient ruins in Rome",
        "Joining a guided tour of the Canadian Rockies",
        "Sailing along the Mediterranean coast",
        "Attending a language immersion camp in Spain",
        "Participating in a surfing camp in Hawaii",
        "Joining a bicycle tour of Tuscany",
        "Exploring the Amazon Rainforest",
        "Participating in a week-long film production workshop",
        "Visiting national parks during a wildlife migration season",
        "Attending a traditional dance festival in India",
        "Discovering street art and graffiti tours in Berlin",
        "Participating in a children's sports camp for soccer",
        "Joining a culinary tour focused on Mexican cuisine",
        "Attending a week-long jazz festival in Montreal",
        "Exploring the fjords of Norway by kayak",
        "Participating in a desert trek in Morocco",
        "Attending a fashion week in Paris",
        "Exploring the castles of Scotland",
        "Participating in a pottery and ceramics workshop in Japan",
        "Fishing expedition in Alaska",
        "Family camping and star-gazing in the Australian Outback",
        "Attending a kite surfing camp in Portugal",
        "Exploring the art galleries and museums of London",
        "Joining a fitness and wellness retreat in Bali",
        "Exploring the ancient temples of Cambodia",
        "Whale watching in Iceland",
        "Attending a chess camp for kids and adults",
        "Participating in a week-long sailing school",
        "Visiting the historical sites of Egypt",
        "Participating in a martial arts training camp in China",
        "Exploring the vineyards of South Africa",
        "Attending a week-long theatre festival",
        "Visiting a remote island for a digital detox retreat",
        "Participating in an ice fishing adventure in Finland",
        "Joining a horseback riding tour in Patagonia",
        "Exploring the street food and markets of Bangkok"
    ]
    return random.choice(range_events), generate_week_number()

def generate_periodic_events_by_week():
    """ Define and select a periodic event that occurs on weekends with a specific frequency. """
    periodic_events = [
        "Attend golf lessons at the local club",
        "Visit the farmers market for fresh produce",
        "Join a weekend hiking group",
        "Take part in a pottery and art class",
        "Play tennis matches at the community courts",
        "Volunteer at the animal shelter",
        "Participate in a community clean-up day",
        "Join a dance class specializing in salsa",
        "Attend wine tasting sessions",
        "Take the family to a matinee movie",
        "Explore new cafes and restaurants",
        "Visit historic sites in the area",
        "Attend live theater performances",
        "Participate in a photography club",
        "Go bird watching in local parks",
        "Attend fitness bootcamps in the park",
        "Participate in a writers' workshop",
        "Join a local amateur astronomy club",
        "Participate in weekend chess tournaments",
        "Join a cycling club for group rides",
        "Attend DIY home improvement workshops",
        "Participate in local fishing contests",
        "Join a book reading group",
        "Attend a cooking class specializing in French cuisine",
        "Join a meditation and mindfulness session",
        "Participate in a language exchange meetup",
        "Attend a historical reenactment event",
        "Volunteer for tutoring at local schools",
        "Participate in a gardening and horticulture club",
        "Attend classic car shows and meets",
        "Join a local running club for weekend marathons",
        "Participate in a kite flying club",
        "Attend a parenting workshop",
        "Join a scuba diving club for local expeditions",
        "Attend monthly tech gadget meetups",
        "Participate in a local street art tour",
        "Join a weekend warriors’ adventure club",
        "Attend a jazz music appreciation club",
        "Participate in a virtual reality gaming session",
        "Join a cocktail mixing class",
        "Attend a fashion design workshop",
        "Participate in a local startup pitch night",
        "Join a public speaking and debate club",
        "Attend a DIY pottery and ceramics class",
        "Participate in a short film club",
        "Join a karaoke night at a local bar",
        "Attend a comic book collectors' meetup",
        "Participate in a yoga retreat weekend",
        "Join a historical walking tour of the city",
        "Attend a DIY soap and candle making class"
    ]

    week_interval = random.choice([2, 3, 4])  # Every 2 to 4 weeks
    return random.choice(periodic_events), week_interval

def generate_prompt():
    """ Generate a single prompt with all necessary details. """
    person_name, person_occupation = generate_person_details()
    family = generate_family()
    family_birthdays = generate_family_birthdays(family)
    selected_range_event, week_number = generate_range_events_by_week()
    selected_periodic_event, periodic_interval = generate_periodic_events_by_week()

    prompt = f"{person_name} is a {person_occupation}. He likes to keep a weekly journal, documenting:\n"
    prompt += f"1) Family life with: {', '.join(f'{member} (birthday on {birthday})' for member, birthday in family_birthdays.items())}.\n"
    prompt += f"2) {selected_range_event} in week {week_number}.\n"
    prompt += f"3) {selected_periodic_event} every {periodic_interval} weeks on weekends.\n"
    prompt += "Each diary entry must be no less than 150 words. Separate different weekly diary with ###. Please start generating his weekly diary from  2012 January to the end of 2012 December without interruption."
    
    return prompt

def generate_prompts(n):
    """ Generate a specified number of prompts. """
    return [generate_prompt() for _ in range(n)]

# Example usage
prompts = generate_prompts(10)
for i, prompt in enumerate(prompts, 1):
    print(f"Prompt {i}:\n{prompt}\n")


from vllm import LLM, SamplingParams
import torch
# Sample prompts.

# Create a sampling params object.

sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens = 8000, seed = 42,presence_penalty = 0.2,frequency_penalty=0.2)


# Create an LLM.
start_time = time.time()
llm = LLM(model="Qwen/Qwen2-7B-Instruct", tensor_parallel_size=2)
#llm = LLM(model="Qwen/Qwen2-7B-Instruct")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")
# Print the outputs.
results = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    results.append({"prompt": prompt, "generated_text": generated_text})
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# 保存生成结果到JSON文件
with open('generated_outputs.json', 'w') as f:
    json.dump(results, f)

print("Generated outputs saved to 'generated_outputs.json'.")