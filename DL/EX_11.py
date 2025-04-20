# pip install google-generativeai
import google.generativeai as genai
API_KEY = 'AIzaSyDgZrWWrCT2bjIMdX9epqmzuWcgeUUBKMQ'
genai.configure(api_key=API_KEY)
def chatbot():
    print("Chatbot: Hello! Type 'exit' to end the chat.")
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = model.generate_content(user_input)
        print("Chatbot:", response.text)
chatbot()
