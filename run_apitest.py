import openai
from config import CONFIG

def gpt35_completion(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    return openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt_text,
        max_tokens=200,
        temperature=0,
        stop=['--', '\n', ';', '#'],
    )["choices"][0]["text"]


test_prompt = """
In a given dialogue, your task is to find the slots where the value is changed by referring to the current user and system turn.
we give each slot with data_type and three examples of value:

"hotel" domain:
hotel-area, text, (south, west, centre)
hotel-stars, int, (5, 4, 1)
hotel-parking, text, (free, no, none)
hotel-internet, text, (free, no, none)
hotel-name, text, (aylesbray lodge guest house, autumn house, rosas bed and breakfast)
hotel-book_day, text, (saturday, none, wednesday)
hotel-book_people, int, (6, 8, 5)
hotel-book_number_of_days, int, (5, 4, 1)
hotel-type, text, (hotel, hotel|guest house, guest house)
hotel-pricerange, text, (expensive, cheap, $100)

"train" domain:
train-destination, text, (norwich, stevenage, london)
train-day, text, (saturday, monday, wednesday)
train-departure, text, (norwich, stevenage, camboats)
train-arrive_by_time, text, (20:54, 18:45, 06:01)
train-book_people, int, (6, 8, 5)
train-depart_time, text, (17:11, 19:35, 21:39)

"attraction" domain:
attraction-type, text, (entertainment, museum, church)
attraction-area, text, (south, west, centre)
attraction-name, text, (clare hall, christ college, museum of archaelogy and anthropogy)

"restaurant" domain:
restaurant-name, text, (tang chinese, city stop restaurant, restaurant 17)
restaurant-book_day, text, (saturday, none, wednesday)
restaurant-book_people, int, (8, 6, 5)
restaurant-book_time, text, (19:15, 20:45, 1715)
restaurant-food_type, text, (russian, british, brazilian)
restaurant-pricerange, text, (expensive, cheap, moderate)
restaurant-area, text, (south, west, centre)

"taxi" domain:
taxi-depart_time, text, (01:15, 18:45, 16:45)
taxi-departure, text, (rosas bed and breakfast, london kings cross train station, alexander bed and breakfast)
taxi-destination, text, (rosas bed and breakfast, tang chinese, alexander bed and breakfast)
taxi-arrive_by_time, text, (11:30, 04:30, 18:30)

Example #1
[context] attraction-type: park, hotel-type: guest house, hotel-parking: yes, hotel-internet: yes
[system] sure ! milton country park is at cb46az . it s free admission .
Q: [user] free is perfect . that will leave me extra to book a taxi from my hotel .
changed_slot: taxi-departure: acorn guest house


Example #2
[context] taxi-destination: pembroke college
[system] where will you be departing from ?
Q: [user] i need to leave from primavera
changed_slot: taxi-departure: primavera


Example #3
[context] 
[system] 
Q: [user] i want to get a taxi to pick me up from the cambridge train station please .
changed_slot: taxi-departure: cambridge train station


Example #4
[context] restaurant-food_type: italian, attraction-type: museum, attraction-area: west, restaurant-book day: tuesday, restaurant-pricerange: moderate, restaurant-area: west, restaurant-book time: 13:45, restaurant-book people: 2, restaurant-name: prezzo
[system] okay . your booking was successful . the reference number is 8z0bbwce . your table will be reserved for 15 minutes .
Q: [user] i need to book a taxi from the museum to the restaurant .
changed_slot: taxi-destination: prezzo, taxi-arrive_by_time: 13:45, taxi-departure: cafe jello gallery


Example #5
[context] restaurant-book day: wednesday, attraction-name: cafe jello gallery, attraction-area: west, restaurant-food_type: indian, restaurant-name: tandoori palace, restaurant-area: west, restaurant-book people: 6, restaurant-pricerange: expensive, restaurant-book time: 12:30
[system] your reference number is src2i813 , your table will be held for 15 mins . is there anything else i can assist you with ?
Q: [user] yes . i need a taxi to take me from the museum to the restaurant .
changed_slot: taxi-destination: tandoori palace, taxi-departure: cafe jello gallery


Example #6
[context] restaurant-name: j restaurant, attraction-name: abbey pool and astroturf pitch, attraction-area: east, restaurant-book time: 12:00, restaurant-book day: sunday, restaurant-book people: 3, attraction-type: swimming pool
[system] sure ! the postcode is cb68nt and the entrance fee is not known .
Q: [user] i would also like to book a taxi from the pool to the restaurant
changed_slot: taxi-departure: abbey pool and astroturf pitch, taxi-destination: j restaurant


Example #7
[context] 
[system] 
Q: [user] i need a taxi departing from tang chinese .
changed_slot: taxi-departure: tang chinese


Example #8
[context] 
[system] 
Q: [user] hello . can you book a taxi for me ? i need to travel from the grafton hotel restaurant to home from home .
changed_slot: taxi-departure: grafton hotel restaurant, taxi-destination: home from home


Example #9
[context] 
[system] 
Q: [user] i want to book a taxi . the taxi should go to da vinci pizzeria and should depart from the missing sock .
changed_slot: taxi-destination: da vinci pizzeria, taxi-departure: the missing sock


Example #10
[context] 
[system] 
Q: [user] hi , i would like to book a taxi from hakka to sidney sussex college .
changed_slot: taxi-destination: sidney sussex college, taxi-departure: hakka


Example #11
[context] 
[system] 
Q: [user] i would like a taxi from saint john s college to pizza hut fen ditton .
changed slot:
"""

completion = gpt35_completion(test_prompt)
print("completion is : {}".format(completion))