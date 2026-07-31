from flask import Flask, render_template, request, jsonify
from chatbot.llm import ask_llm
from chatbot.intent_router import classify_intent
from chatbot.recruitment_rag import handle_recruitment_query, build_index

app = Flask(__name__)

# Build the resume index once at startup so the first recruitment
# query isn't slow. Comment this out if you'd rather build lazily
# on first request (handled automatically inside recruitment_rag.py).
with app.app_context():
    try:
        build_index()
        print("Resume index built successfully.")
    except Exception as e:
        print(f"Warning: could not build resume index at startup ({e}). "
              f"It will be built on first recruitment query instead.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data["message"]

    intent = classify_intent(user_message)

    if intent == "recruitment":
        ai_response = handle_recruitment_query(user_message)
    else:
        ai_response = ask_llm(user_message)

    return jsonify({
        "response": ai_response
    })


if __name__ == "__main__":
    app.run(debug=True)