document.getElementById("send-button").addEventListener("click", function() {
  var userInput = document.getElementById("user-input").value;
  var chatMessages = document.getElementById("chat-messages");

  // Display user message
  chatMessages.innerHTML += "<p><strong>You:</strong> " + userInput + "</p>";

  // Send user message to Flask server
  fetch("/send_message", {
    method: "POST",
    body: JSON.stringify({ message: userInput }),
    headers: {
      "Content-Type": "application/json"
    }
  })
  .then(response => response.json())
  .then(data => {
    // Display chatbot response
    chatMessages.innerHTML += "<p><strong>Andres Perez:</strong> " + data.message + "</p>";
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
  // Clear input field
  document.getElementById("user-input").value = "";
});