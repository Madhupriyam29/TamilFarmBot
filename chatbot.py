#using stoping word

import nltk
import spacy
import random
import json
import numpy as np
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  # Ensure this import is correct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Initialize NLP tools
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# Define stopwords list
stop_words = set(stopwords.words('english')) 

# List of intents related to Tamil Nadu farmers (50 questions divided into categories)
intents = {
    "greetings": [
    {"question": "Hi there!", "answer": "Hello! How can I assist you today?"},
    {"question": "Hello!", "answer": "Hi! How's it going?"},
    {"question": "Good morning!", "answer": "Good morning! How can I help you today?"},
    {"question": "Good evening!", "answer": "Good evening! How’s your day going?"},
    {"question": "How are you?", "answer": "I’m doing great, thank you for asking! How about you?"},
    {"question": "How’s it going?", "answer": "It’s going well! What can I help you with today?"},
    {"question": "What's up?", "answer": "Not much, just here to help! How can I assist you?"},
    {"question": "Hey!", "answer": "Hey! How can I assist you today?"}
   ],
    "polite_inquiries": [
    {"question": "Can you help me?", "answer": "Of course! What do you need help with?"},
    {"question": "Is there anything you can do for me?", "answer": "I’m here to help with anything you need! Just let me know."},
    {"question": "What can you do?", "answer": "I can assist you with questions related to farming, agriculture, and more! What can I help you with today?"},
    {"question": "How can I get your assistance?", "answer": "You can ask me anything, and I’ll do my best to provide helpful information."}
   ],
    "casual_checkins": [
    {"question": "Everything okay?", "answer": "Yes, everything’s great! How about you? Is everything alright on your end?"},
    {"question": "How’s everything going?", "answer": "Everything is going smoothly! How can I assist you today?"},
    {"question": "How can I be of service today?", "answer": "I’m here to help! What can I assist you with today?"}
  ],
    "crop_info": [
        {"question": "What are the main crops grown in Tamil Nadu?", "answer": "The main crops grown in Tamil Nadu include rice, sugarcane, cotton, bananas, mangoes, and coconut."},
        {"question": "What are the best rice varieties grown in Tamil Nadu?", "answer": "The best rice varieties in Tamil Nadu include Ponni, Sona Masoori, and Basmati."},
        {"question": "How is sugarcane farming done in Tamil Nadu?", "answer": "Sugarcane farming involves planting cuttings, irrigating crops, and harvesting after 12-18 months."},
        {"question": "How does Tamil Nadu grow bananas and mangoes?", "answer": "Bananas and mangoes thrive in Tamil Nadu's tropical climate with proper irrigation and organic fertilizers."},
        {"question": "What are the major vegetables grown in Tamil Nadu?", "answer": "Tamil Nadu grows vegetables like tomatoes, onions, brinjals, and leafy greens."},
        {"question": "How is coconut farming practiced in Tamil Nadu?", "answer": "Coconut farming requires well-drained soil and regular irrigation. The palm tree produces coconuts throughout the year."},
        {"question": "What are the major challenges in growing cotton in Tamil Nadu?", "answer": "Challenges include pest attacks, water scarcity, and fluctuating market prices for cotton."},
        {"question": "How do farmers in Tamil Nadu grow spices like black pepper?", "answer": "Spices like black pepper are grown on trellises in Tamil Nadu, requiring well-drained soil and proper irrigation."},
        {"question": "How does rice farming work in Tamil Nadu?", "answer": "Rice farming in Tamil Nadu involves preparing the fields, sowing seeds, regular irrigation, and harvesting during the monsoon season."},
        {"question": "How is tea farming managed in Tamil Nadu’s hill stations?", "answer": "Tea farming requires a cool climate, terraced hills, and regular plucking of tender tea leaves."}
    ],
    
    "weather_impact": [
        {"question": "How does the weather in Tamil Nadu affect farming?", "answer": "Cyclones, floods, and droughts often affect crops in Tamil Nadu, impacting agricultural productivity."},
        {"question": "What are the impacts of cyclones on Tamil Nadu's agriculture?", "answer": "Cyclones damage crops like rice and sugarcane, erode soil, and lead to loss of crop yields."},
        {"question": "How do Tamil Nadu farmers prepare for the monsoon season?", "answer": "Farmers prepare by planting water-resistant crops and using rainwater harvesting methods."},
        {"question": "What is the effect of drought on Tamil Nadu’s agriculture?", "answer": "Drought severely impacts irrigation-dependent crops like rice and sugarcane, reducing yields."},
        {"question": "How does flood damage affect farming in Tamil Nadu?", "answer": "Floods submerge fields, destroy crops, erode soil, and create waterlogging issues."},
        {"question": "How does temperature variation impact crop growth in Tamil Nadu?", "answer": "Extreme temperatures can damage crops, particularly those sensitive to temperature fluctuations like rice and cotton."},
        {"question": "What is the role of weather forecasting in helping Tamil Nadu farmers?", "answer": "Weather forecasting helps farmers plan their sowing, irrigation, and harvesting schedules to avoid weather-related losses."},
        {"question": "How do farmers in Tamil Nadu use traditional weather prediction methods?", "answer": "Farmers use signs like wind patterns and the behavior of animals to predict weather changes."},
        {"question": "How do farmers manage water conservation during dry seasons in Tamil Nadu?", "answer": "Farmers use techniques like drip irrigation, rainwater harvesting, and groundwater management to conserve water."},
        {"question": "How does Tamil Nadu's tropical climate affect agriculture?", "answer": "The tropical climate is ideal for crops like rice, sugarcane, and bananas, but it also makes the region prone to droughts and floods."}
    ],
    
    "water_crisis": [
        {"question": "How do farmers in Tamil Nadu manage water scarcity?", "answer": "Farmers use drip irrigation, rainwater harvesting, and efficient irrigation practices to combat water scarcity."},
        {"question": "How has the Cauvery Water Dispute affected Tamil Nadu farmers?", "answer": "The dispute has led to water shortages for irrigation, affecting crop yields in some parts of Tamil Nadu."},
        {"question": "What irrigation technologies are used in Tamil Nadu?", "answer": "Farmers use technologies like drip irrigation, sprinkler systems, and canals for efficient water usage."},
        {"question": "What is drip irrigation, and how is it used in Tamil Nadu?", "answer": "Drip irrigation delivers water directly to the roots of plants, conserving water and improving crop yields."},
        {"question": "How does groundwater depletion affect farmers in Tamil Nadu?", "answer": "Groundwater depletion leads to higher pumping costs and insufficient water for crops."},
        {"question": "What steps are being taken to improve water management for farmers in Tamil Nadu?", "answer": "The government promotes rainwater harvesting, water-efficient technologies, and sustainable irrigation practices."},
        {"question": "How does the irrigation system in Tamil Nadu support agriculture?", "answer": "The irrigation system, including rivers, reservoirs, and canals, supports agricultural productivity by providing water during critical growth stages."},
        {"question": "How can farmers save water in agriculture in Tamil Nadu?", "answer": "Farmers can save water by adopting efficient irrigation methods like drip and sprinkler irrigation and by using organic mulches."}
    ],

    "government_schemes": [
        {"question": "What government schemes are available for farmers in Tamil Nadu?", "answer": "Government schemes like PM Kisan Samman Nidhi, crop insurance, and subsidies for fertilizers help support farmers in Tamil Nadu."},
        {"question": "How does the PM Kisan Samman Nidhi help Tamil Nadu farmers?", "answer": "PM Kisan Samman Nidhi provides Rs. 6,000 annually to eligible farmers to assist with farming expenses."},
        {"question": "What is the Tamil Nadu Government’s Agriculture Department doing to help farmers?", "answer": "The Agriculture Department provides subsidies for inputs, implements disaster relief, and promotes organic farming."},
        {"question": "How can farmers apply for financial assistance in Tamil Nadu?", "answer": "Farmers can apply for financial assistance through online portals or by visiting the local agriculture offices."},
        {"question": "What is the price support scheme for crops in Tamil Nadu?", "answer": "The government buys crops at minimum support prices to ensure farmers receive a fair price for their produce."},
        {"question": "How does the government assist farmers with insurance in Tamil Nadu?", "answer": "Farmers can avail themselves of crop insurance schemes to protect against crop failures due to weather events."},
        {"question": "What are the subsidies available for Tamil Nadu farmers for fertilizers?", "answer": "Subsidies for fertilizers like urea and DAP are provided to reduce the cost of inputs for farmers."},
        {"question": "What are the benefits of the National Agriculture Market (eNAM) for farmers in Tamil Nadu?", "answer": "eNAM provides farmers with a national platform to sell their produce at competitive prices, ensuring better market access."},
        {"question": "What is the MGNREGA program, and how does it support farmers?", "answer": "The MGNREGA program provides wage employment in rural areas, including agriculture-related tasks like land development."},
        {"question": "How does the Tamil Nadu government support organic farming?", "answer": "The government provides training, certification, and subsidies to encourage organic farming practices."}
    ],

    "challenges_faced": [
        {"question": "How do farmers in Tamil Nadu deal with debt problems?", "answer": "Farmers often take loans from banks or private lenders. Government relief schemes help manage agricultural debt."},
        {"question": "What are the main agricultural labor issues in Tamil Nadu?", "answer": "Labor shortages, low wages, and migration are common agricultural labor challenges in Tamil Nadu."},
        {"question": "How do farmers in Tamil Nadu cope with fluctuating market prices?", "answer": "Farmers use government price support schemes and sell their produce through cooperatives to get better prices."},
        {"question": "What are the effects of land fragmentation on Tamil Nadu farmers?", "answer": "Land fragmentation reduces farm size, limiting economies of scale and making farming less profitable."},
        {"question": "How do farmers in Tamil Nadu deal with the impacts of industrialization?", "answer": "Industrialization affects farmers by encroaching on agricultural land, leading to reduced space for farming."},
        {"question": "What role do middlemen play in Tamil Nadu agriculture?", "answer": "Middlemen often exploit farmers by purchasing crops at low prices and selling them at higher rates to consumers."},
        {"question": "How does land acquisition for industrial projects affect farmers in Tamil Nadu?", "answer": "Land acquisition for industrial projects displaces farmers, reducing agricultural land and livelihoods."},
        {"question": "What steps are being taken to improve farmer education in Tamil Nadu?", "answer": "Farmers are being educated through agricultural extension services, farmer training programs, and community workshops."},
        {"question": "How do farmers in Tamil Nadu handle crop failures?", "answer": "Farmers seek government aid, use crop insurance, and try to diversify crops to mitigate the risk of failures."},
        {"question": "What impact do seed monopolies have on Tamil Nadu farmers?", "answer": "Seed monopolies often lead to higher seed costs and dependency on a few companies for planting material."}
    ],

    "technology_innovation": [
        {"question": "How is technology improving farming practices in Tamil Nadu?", "answer": "Technological innovations like drones, mobile apps, AI, and precision farming help improve crop yields and sustainability."},
        {"question": "What is the role of drones in Tamil Nadu agriculture?", "answer": "Drones are used to monitor crop health, apply pesticides, and collect data for precision farming."},
        {"question": "How are farmers using mobile apps for agriculture in Tamil Nadu?", "answer": "Mobile apps provide weather forecasts, market prices, and pest management advice to farmers."},
        {"question": "How does artificial intelligence benefit farming in Tamil Nadu?", "answer": "AI helps predict crop yields, detect pests, and optimize irrigation and fertilizer use for better productivity."},
        {"question": "How are weather monitoring tools being used by Tamil Nadu farmers?", "answer": "Farmers use weather apps and satellite data to predict weather patterns and plan their farming activities."},
        {"question": "What is precision farming, and how is it used in Tamil Nadu?", "answer": "Precision farming involves using technology to monitor soil, crops, and weather to optimize inputs and increase productivity."},
        {"question": "How are farmers in Tamil Nadu using machine learning for agriculture?", "answer": "Machine learning helps farmers analyze data for crop management, pest control, and yield prediction."},
        {"question": "What role do agricultural startups play in Tamil Nadu?", "answer": "Agricultural startups are introducing new technologies, providing advisory services, and linking farmers with markets."},
        {"question": "How is remote sensing technology used to improve farming in Tamil Nadu?", "answer": "Remote sensing helps monitor crop health, soil moisture, and irrigation needs through satellite imagery and sensors."},
        {"question": "How do farmers in Tamil Nadu use e-commerce platforms for selling their crops?", "answer": "Farmers use e-commerce platforms to sell produce directly to consumers or businesses, reducing reliance on middlemen."}
    ],

    "district_farming": [
        {"question": "Where is tea cultivated in Tamil Nadu?", "answer": "Tea is primarily cultivated in the Nilgiri hills, which is a famous tea-producing region in Tamil Nadu."},
        {"question": "Where is sugarcane grown in Tamil Nadu?", "answer": "Sugarcane is widely cultivated in districts like Thanjavur, Salem, and Villupuram in Tamil Nadu."},
        {"question": "Where is rice cultivated in Tamil Nadu?", "answer": "Rice is widely grown in Thanjavur, Tiruvarur, Cuddalore, and Vellore districts in Tamil Nadu."},
        {"question": "Where is groundnut cultivated in Tamil Nadu?", "answer": "Groundnut is primarily grown in districts like Erode, Tirunelveli, and Virudhunagar in Tamil Nadu."},
        {"question": "Where is cotton cultivated in Tamil Nadu?", "answer": "Cotton is cultivated in districts like Coimbatore, Salem, and Dharmapuri in Tamil Nadu."},
        {"question": "Where is turmeric grown in Tamil Nadu?", "answer": "Turmeric is mostly grown in Erode, Salem, and Namakkal districts in Tamil Nadu."},
        {"question": "Where is banana grown in Tamil Nadu?", "answer": "Banana is cultivated in districts like Kanyakumari, Tirunelveli, and Coimbatore in Tamil Nadu."},
        {"question": "Where is coconut cultivated in Tamil Nadu?", "answer": "Coconut is grown in coastal districts like Kanyakumari, Thanjavur, and Nagapattinam in Tamil Nadu."},
        {"question": "Where are vegetables like tomatoes cultivated in Tamil Nadu?", "answer": "Tomatoes are widely grown in districts like Salem, Dindigul, and Krishnagiri in Tamil Nadu."},
        {"question": "Where is pomegranate cultivated in Tamil Nadu?", "answer": "Pomegranate is mostly grown in Dindigul, Krishnagiri, and Salem districts in Tamil Nadu."},
        {"question": "Where is cashew nut cultivated in Tamil Nadu?", "answer": "Cashew nut is cultivated in Kanyakumari, Thoothukudi, and Ramanathapuram districts in Tamil Nadu."},
        {"question": "Where is coffee cultivated in Tamil Nadu?", "answer": "Coffee is mainly cultivated in the hill districts of Nilgiris and Coimbatore in Tamil Nadu."},
        {"question": "Where is ginger cultivated in Tamil Nadu?", "answer": "Ginger is mainly grown in districts like Coimbatore, Dindigul, and Namakkal in Tamil Nadu."},
        {"question": "Where is onion grown in Tamil Nadu?", "answer": "Onion is widely cultivated in districts like Erode, Dharmapuri, and Ariyalur in Tamil Nadu."},
        {"question": "Where is millet cultivated in Tamil Nadu?", "answer": "Millet is grown in dryland areas like Madurai, Theni, and Dindigul in Tamil Nadu."},
        {"question": "Where is mango grown in Tamil Nadu?", "answer": "Mangoes are cultivated in districts like Krishnagiri, Salem, and Theni in Tamil Nadu."},
        {"question": "Where is tobacco grown in Tamil Nadu?", "answer": "Tobacco is grown in districts like Virudhunagar, Thoothukudi, and Ramanathapuram in Tamil Nadu."},
        {"question": "Where are guavas cultivated in Tamil Nadu?", "answer": "Guavas are grown in districts like Coimbatore, Krishnagiri, and Kanyakumari in Tamil Nadu."},
        {"question": "Where are chillies cultivated in Tamil Nadu?", "answer": "Chillies are grown in districts like Dharmapuri, Cuddalore, and Villupuram in Tamil Nadu."},
        {"question": "Where is cardamom grown in Tamil Nadu?", "answer": "Cardamom is cultivated in the western districts like Coimbatore, Nilgiris, and Idukki (Tamil Nadu-Kerala border region)."},
        {"question": "Where is wheat cultivated in Tamil Nadu?", "answer": "Wheat is cultivated in northern parts of Tamil Nadu like Namakkal and Ariyalur."},
        {"question": "Where is lettuce grown in Tamil Nadu?", "answer": "Lettuce is cultivated in hill stations such as the Nilgiris and other cooler regions of Tamil Nadu."},
        {"question": "Where is sweet corn cultivated in Tamil Nadu?", "answer": "Sweet corn is cultivated in districts like Coimbatore, Salem, and Dindigul in Tamil Nadu."},
        {"question": "Where are grapes cultivated in Tamil Nadu?", "answer": "Grapes are cultivated in districts like Salem, Krishnagiri, and Vellore in Tamil Nadu."},
        {"question": "Where is papaya cultivated in Tamil Nadu?", "answer": "Papaya is widely grown in districts like Kanyakumari, Tirunelveli, and Coimbatore in Tamil Nadu."},
        {"question": "Where is watermelon grown in Tamil Nadu?", "answer": "Watermelon is cultivated in districts like Vellore, Dindigul, and Namakkal in Tamil Nadu."},
        {"question": "Where are apples grown in Tamil Nadu?", "answer": "Apples are grown in cooler regions such as the Nilgiris and parts of Coimbatore in Tamil Nadu."},
        {"question": "Where is saffron cultivated in Tamil Nadu?", "answer": "Saffron is cultivated in smaller quantities in the Nilgiri hills in Tamil Nadu."},
        {"question": "Where is chili pepper grown in Tamil Nadu?", "answer": "Chili pepper is cultivated in districts like Villupuram, Cuddalore, and Dharmapuri in Tamil Nadu."},
        {"question": "Where are pulses grown in Tamil Nadu?", "answer": "Pulses like chickpeas, lentils, and pigeon peas are cultivated in districts like Madurai, Dharmapuri, and Salem."},
        {"question": "Where is sesame cultivated in Tamil Nadu?", "answer": "Sesame is grown in districts like Dharmapuri, Salem, and Erode in Tamil Nadu."},
        {"question": "Where is tapioca grown in Tamil Nadu?", "answer": "Tapioca is mainly cultivated in districts like Tirunelveli, Kanyakumari, and Thoothukudi in Tamil Nadu."},
        {"question": "Where is jaggery produced in Tamil Nadu?", "answer": "Jaggery is produced from sugarcane in districts like Thanjavur, Salem, and Villupuram in Tamil Nadu."},
        {"question": "Where is betel leaf grown in Tamil Nadu?", "answer": "Betel leaf is cultivated in parts of Kanyakumari, Thoothukudi, and Chennai in Tamil Nadu."},
        {"question": "Where is pineapple grown in Tamil Nadu?", "answer": "Pineapple is cultivated in districts like Kanyakumari, Tirunelveli, and Salem in Tamil Nadu."},
        {"question": "Where is jackfruit cultivated in Tamil Nadu?", "answer": "Jackfruit is grown in districts like Coimbatore, Krishnagiri, and Kanyakumari in Tamil Nadu."},
        {"question": "Where is rubber cultivated in Tamil Nadu?", "answer": "Rubber is mainly cultivated in the Nilgiri hills and parts of Coimbatore in Tamil Nadu."},
        {"question": "Where is mustard grown in Tamil Nadu?", "answer": "Mustard is grown in districts like Villupuram, Cuddalore, and Thanjavur in Tamil Nadu."},
        {"question": "Where is coriander cultivated in Tamil Nadu?", "answer": "Coriander is cultivated in districts like Salem, Krishnagiri, and Dharmapuri in Tamil Nadu."},
        {"question": "Where is papaya grown in Tamil Nadu?", "answer": "Papaya is cultivated in Kanyakumari, Tirunelveli, and Coimbatore districts in Tamil Nadu."},
        {"question": "Where are carrots grown in Tamil Nadu?", "answer": "Carrots are grown in cooler regions like the Nilgiris and parts of Coimbatore in Tamil Nadu."},
         {"question": "What is the most famous crop in Tamil Nadu?", "answer": "Rice is the most famous and widely cultivated crop in Tamil Nadu, especially in the delta regions like Thanjavur."},
        {"question": "Where are plums grown in Tamil Nadu?", "answer": "Plums are mainly cultivated in the Nilgiri hills and parts of Coimbatore in Tamil Nadu."}
    ], 

    "default": [
        {"question": "Sorry, I didn't understand that. Please ask questions related to Tamil Nadu farmers and agriculture.", "answer": "Sorry, I didn't understand that. Please ask questions related to Tamil Nadu farmers and agriculture."}
    ]
}
# Function to preprocess input
def preprocess_input(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^a-z\s]', '', user_input)  # Remove any special characters or punctuation
    doc = nlp(user_input)

    # Lemmatize and remove stopwords
    lemmatized_words = [lemmatizer.lemmatize(token.text) for token in doc if token.is_alpha and token.text not in stop_words]
    #print(f"Processed input: {' '.join(lemmatized_words)}")  # Add this line for debugging
    return ' '.join(lemmatized_words)

# Function to respond to user input based on intent
def get_intent(user_input):
    # Check if the question mentions another state like Karnataka
    other_states = ['karnataka', 'kerala', 'andhra pradesh', 'telangana', 'goa', 'maharashtra', 'uttar pradesh', 'bihar']  # Add more states if necessary
    if any(state in user_input.lower() for state in other_states):
        # Return a default intent with a response about Tamil Nadu focus
        return "default", {"question": "Sorry, I can only provide information about Tamil Nadu's agriculture. Please ask about Tamil Nadu farmers.", "answer": "Sorry, I can only provide information about Tamil Nadu's agriculture. Please ask about Tamil Nadu farmers."}

    user_input = preprocess_input(user_input)
    vectorizer = TfidfVectorizer()

    # Flatten the intent questions into a single list
    all_intents = [item['question'] for sublist in list(intents.values()) for item in sublist]

    # Combine the user input with all predefined questions for comparison
    vectors = vectorizer.fit_transform([user_input] + all_intents)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Find the intent with the highest similarity
    best_match_index = np.argmax(cosine_similarities)

    # If the cosine similarity is high enough (50% or above), return the matched intent
    if cosine_similarities[best_match_index] > 0.3:  # Lowered the threshold to 0.5
        # Locate the matched question in the original list
        current_index = 0
        for intent, examples in intents.items():
            if best_match_index < current_index + len(examples):
                matched_example = examples[best_match_index - current_index]
                return intent, matched_example
            current_index += len(examples)
    
    # Default intent if no good match is found
    return "default", {"question": "Sorry, I didn't understand that. Please ask questions related to Tamil Nadu farmers and agriculture.", "answer": "Sorry, I didn't understand that. Please ask questions related to Tamil Nadu farmers and agriculture."}

# Main function for chatbot interaction
def chatbot():
    print("Hello! I am your Tamil Nadu Agriculture chatbot. Ask me about Tamil Nadu farmers.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        intent, answer = get_intent(user_input)
        
        # Debugging print
       # print(f"Intent: {intent}")
        #(f"Answer: {answer['answer']}")
        
        print(f"Bot: {answer['answer']}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()