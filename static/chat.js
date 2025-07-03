// Simple Chatbot with Gemini API
// API key is handled by the backend

document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const closeChat = document.getElementById('close-chat');
    const chatToggle = document.querySelector('.chat-toggle');
    
    // Toggle chat window
    chatToggle.addEventListener('click', () => {
        if (chatContainer.style.display === 'flex') {
            chatContainer.style.display = 'none';
        } else {
            chatContainer.style.display = 'flex';
            chatInput.focus();
        }
    });
    
    closeChat.addEventListener('click', () => {
        chatContainer.style.display = 'none';
    });
    
    // Send message on button click or Enter key
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => e.key === 'Enter' && sendMessage());
    
    // Add welcome message
    addMessage('bot', 'Hello! I\'m your health assistant. How can I help you today?');
    
    // Send message function
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage('user', message);
        chatInput.value = '';
        
        // Show typing indicator
        const typingIndicator = addMessage('bot', '...');
        
        try {
            console.log('Sending message to backend...');
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message
                })
            });
            
            console.log('Received response, status:', response.status);
            const data = await response.json();
            console.log('API Response:', data);
            
            // Remove typing indicator
            chatMessages.removeChild(typingIndicator);
            
            if (response.ok) {
                if (data.response) {
                    // Use Marked.js to render markdown content
                    const htmlContent = marked.parse(data.response);
                    addMessage('bot', htmlContent, true);
                } else {
                    console.error('Unexpected response format:', data);
                    addMessage('bot', 'I received an unexpected response format. Please try again.');
                }
            } else {
                console.error('API Error:', data);
                const errorMsg = data.error?.message || 'Failed to get response';
                throw new Error(`API Error: ${errorMsg}`);
            }
        } catch (error) {
            console.error('Error:', error);
            if (typingIndicator && typingIndicator.parentNode) {
                chatMessages.removeChild(typingIndicator);
            }
            addMessage('bot', `Sorry, I encountered an error: ${error.message || 'Please try again later.'}`);
            
            // Add retry button
            const retryButton = document.createElement('button');
            retryButton.textContent = 'Retry';
            retryButton.className = 'retry-button';
            retryButton.onclick = () => {
                chatMessages.removeChild(chatMessages.lastChild);
                sendMessage();
            };
            addMessage('bot', '');
            chatMessages.lastChild.appendChild(retryButton);
        }
    }
    
    // Add message to chat
    function addMessage(sender, message, isHtml = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message`;
        
        if (isHtml) {
            messageDiv.innerHTML = message;
        } else {
            messageDiv.textContent = message;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }
});
