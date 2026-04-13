"""
Gradio Web Interface for Boston School Chatbot

This script creates a web interface for your chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py

    # Access in browser:
    # http://localhost:7860
"""

import gradio as gr
from pathlib import Path

BACKGROUND_IMAGE = "background_image.jpeg"

CUSTOM_CSS = f"""
body, .gradio-container {{
    background:
        linear-gradient(rgba(15, 23, 42, 0.58), rgba(15, 23, 42, 0.70)),
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
}}

.gradio-container {{
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

footer {{
    display: none !important;
}}

#dani-shell {{
    max-width: 980px;
    margin: 28px auto;
    padding: 18px;
    border-radius: 28px;
    background: rgba(15, 23, 42, 0.58);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
}}

#dani-shell .prose h1,
#dani-shell .prose p,
#dani-shell label,
#dani-shell .message,
#dani-shell .message-row,
#dani-shell .message-wrap,
#dani-shell .placeholder,
#dani-shell .wrap {{
    color: #f8fafc !important;
}}

#dani-shell .bubble-wrap.svelte-1lcyrx4,
#dani-shell .message-wrap,
#dani-shell .message {{
    border-radius: 18px !important;
}}

#dani-shell textarea,
#dani-shell input {{
    background: rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}}

#dani-shell button {{
    border-radius: 16px !important;
}}

#dani-shell .primary {{
    background: linear-gradient(135deg, #f472b6, #a78bfa) !important;
    border: none !important;
    color: white !important;
}}

#dani-shell .secondary {{
    background: rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
}}
"""

from src.chat import Chatbot

def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    chatbot = Chatbot()

    def chat(message, history):
        """
        TODO:Generate a response for the current message in a Gradio chat interface.

        This function is called by Gradio's ChatInterface every time a user sends a message.
        You only need to generate and return the assistant's response - Gradio handles the
        chat display and history management automatically.

        Args:
            message (str): The current message from the user
            history (list): List of previous message pairs, where each pair is
                           [user_message, assistant_message]
                           Example:
                           [
                               ["What schools offer Spanish?", "The Hernandez School..."],
                               ["Where is it located?", "The Hernandez School is in Roxbury..."]
                           ]

        Returns:
            str: The assistant's response to the current message.


        Note:
            - Gradio automatically:
                - Displays the user's message
                - Displays your returned response
                - Updates the chat history
                - Maintains the chat interface
            - You only need to:
                - Generate an appropriate response to the current message
                - Return that response as a string
        """
        # TODO: Generate and return response
        response = chatbot.get_response(message, history=history)
        return response



    # Create Gradio interface. Customize the interface however you'd like!
    demo = gr.ChatInterface(
        chat,
        title="dani bot ✨",
        description=(
            "a chatbot with a more playful, thoughtful, dani-style voice. "
            "it is best at casual conversation, little reactions, supportive replies, "
            "and explaining things in a way that still feels human."
        ),
        examples=[
            "hello dani",
            "i have tea",
            "you are mean",
            "explain dynamic programming",
            "tell me about myself",
        ],
    )

    demo.css = CUSTOM_CSS
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch(share=True)
