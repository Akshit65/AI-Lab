const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const messages = document.getElementById("messages");

function addUserMessage(message) {

    const div = document.createElement("div");
    div.className = "user-message";
    div.innerText = message;

    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function addBotMessage(message) {

    const div = document.createElement("div");
    div.className = "bot-message";
    div.innerText = message;

    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

async function sendMessage() {

    const text = userInput.value.trim();

    if (text === "")
        return;

    addUserMessage(text);

    userInput.value = "";

    addBotMessage("Thinking...");

    const loadingMessage = messages.lastChild;

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: text
            })

        });

        const data = await response.json();

        loadingMessage.innerText = data.response;

    } catch (error) {

        loadingMessage.innerText = "Error communicating with AI.";

    }

}

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keypress", function(event) {

    if (event.key === "Enter") {
        sendMessage();
    }

});

const newChatBtn = document.getElementById("newChat");

function startNewChat() {
    messages.innerHTML = `
        <div class="bot-message">
            👋 Hello! I am your Enterprise AI Assistant.
        </div>
    `;
}

newChatBtn.addEventListener("click", startNewChat);