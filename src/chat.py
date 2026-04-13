import re
from pathlib import Path

import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from config import BASE_MODEL, MY_MODEL, HF_TOKEN


WHATSAPP_LINE_PATTERN = re.compile(r"^(?:\u200e)?\[(.*?)\]\s+([^:]+):\s?(.*)$")


class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL  # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.my_name = "Danielle Knutson"
        self.project_root = Path(__file__).resolve().parent.parent
        self.chat_dir = self.project_root / "chats"
        self.essay_path = self.project_root / "Sussays.txt"
        self.chat_corpus = self.load_chat_examples(max_examples=400)
        self.essay_corpus = self.load_essay_examples(max_examples=80)
        self.num_chat_examples = 8
        self.num_essay_examples = 2
        self.max_history_turns = 8
        self.summary_keep_last_turns = 6
        self.summary_max_chars = 900
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.chat_embeddings = self.embed_examples(self.chat_corpus)
        self.essay_embeddings = self.embed_examples(self.essay_corpus)

    def format_history(self, history, max_turns=None):
        """Format only the most recent conversation turns for the prompt."""
        if not history:
            return []

        max_turns = self.max_history_turns if max_turns is None else max_turns
        recent_history = history[-max_turns:]
        formatted_lines = []

        for i, turn in enumerate(recent_history, start=1):
            if not turn:
                continue

            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                user_message, assistant_message = turn[0], turn[1]
            else:
                continue

            if user_message:
                formatted_lines.append(f"User {i}: {str(user_message).strip()}")
            if assistant_message:
                formatted_lines.append(f"Assistant {i}: {str(assistant_message).strip()}")

        return formatted_lines

    def build_conversation_summary(self, history):
        """Build a lightweight summary of older conversation turns."""
        if not history:
            return ""

        if len(history) <= self.summary_keep_last_turns:
            return ""

        older_history = history[:-self.summary_keep_last_turns]
        summary_lines = []

        for turn in older_history:
            if not turn:
                continue
            if not isinstance(turn, (list, tuple)) or len(turn) < 2:
                continue

            user_message, assistant_message = turn[0], turn[1]
            if user_message:
                summary_lines.append(f"User said: {str(user_message).strip()}")
            if assistant_message:
                summary_lines.append(f"Assistant replied: {str(assistant_message).strip()}")

        summary_text = " | ".join(summary_lines)
        if len(summary_text) > self.summary_max_chars:
            summary_text = summary_text[: self.summary_max_chars].rstrip() + "..."

        return summary_text

    def format_recent_history(self, history):
        """Format only the most recent conversation turns for the prompt."""
        if not history:
            return []

        recent_history = history[-self.summary_keep_last_turns :]
        return self.format_history(recent_history, max_turns=self.summary_keep_last_turns)

    def embed_examples(self, examples):
        """Precompute normalized embeddings for a corpus of examples."""
        if not examples:
            return None

        return self.embedding_model.encode(
            examples,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def retrieve_relevant_examples(self, user_input, examples, embeddings, k):
        """Return the top-k most semantically relevant examples for the current user input."""
        if not examples:
            return []
        if embeddings is None:
            return examples[:k]

        query_embedding = self.embedding_model.encode(
            user_input,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores = embeddings @ query_embedding
        top_k = min(k, len(examples))
        top_indices = np.argsort(-scores)[:top_k]

        return [examples[idx] for idx in top_indices]

    def should_use_essay_examples(self, user_input: str) -> bool:
        """Use essay examples only for clearly formal, reflective, explanatory, or personal-background prompts."""
        lowered = user_input.casefold().strip()

        personal_cues = [
            "about me",
            "who am i",
            "tell me about myself",
            "my background",
            "my history",
            "my story",
            "my experience",
            "my interests",
            "my personality",
            "describe me",
        ]
        if any(cue in lowered for cue in personal_cues):
            return True

        formal_cues = [
            "essay",
            "formal",
            "professional",
            "email",
            "cover letter",
            "statement",
            "summarize",
            "summary",
            "analysis",
            "analyze",
            "reflection",
            "reflect",
            "background",
            "history",
        ]
        if any(cue in lowered for cue in formal_cues):
            return True

        explanatory_starts = [
            "explain",
            "can you explain",
            "help me understand",
            "walk me through",
        ]
        if any(lowered.startswith(prefix) for prefix in explanatory_starts):
            return True

        return len(user_input.split()) >= 20

    def normalize_response_style(self, user_input: str, response: str) -> str:
        """Light post-processing to reduce overly formal punctuation in short casual replies."""
        cleaned = response.strip()
        if not cleaned:
            return cleaned

        casual_prompt = (
            self.is_simple_greeting(user_input)
            or self.is_low_stakes_rude_message(user_input)
            or self.is_tea_message(user_input)
            or self.asks_for_personal_checkin(user_input)
            or len(user_input.split()) <= 6
        )

        if casual_prompt and len(cleaned.split()) <= 12:
            cleaned = re.sub(r",\s+", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            cleaned = cleaned.rstrip(".,;")

        return cleaned

    def should_use_personal_background(self, user_input: str) -> bool:
        """Use essay examples as personal background only when the user explicitly asks about Danielle herself."""
        lowered = user_input.casefold()
        personal_cues = [
            "about me",
            "who am i",
            "tell me about myself",
            "my background",
            "my history",
            "my story",
            "my experience",
            "my interests",
            "my personality",
            "describe me",
        ]
        return any(cue in lowered for cue in personal_cues)

    def is_low_stakes_rude_message(self, user_input: str) -> bool:
        """Detect short rude or teasing messages that should not trigger therapy-speak."""
        lowered = user_input.casefold().strip()
        rude_cues = [
            "ugly",
            "stupid",
            "dumb",
            "ew",
            "lame",
            "trash",
            "shut up",
            "loser",
            "bozo",
            "eww",
        ]
        return len(lowered.split()) <= 6 and any(cue in lowered for cue in rude_cues)

    def is_simple_greeting(self, user_input: str) -> bool:
        """Detect short greeting messages, including greetings with a name attached."""
        lowered = user_input.casefold().strip()
        simple_greetings = {
            "hi", "hii", "hiii", "hiiii",
            "hey", "heyy", "heyyy",
            "hello", "yo", "sup",
            "what's up", "whats up",
            "good morning", "good afternoon", "good evening",
            "hiya", "heyy", "heyyyy",
        }

        if lowered in simple_greetings:
            return True

        greeting_prefixes = [
            "hi ", "hii ", "hiii ", "hiiii ",
            "hey ", "heyy ", "heyyy ", "heyy ", "heyyyy ",
            "hello ", "yo ", "sup ",
            "hiya ",
            "good morning ", "good afternoon ", "good evening ",
        ]
        return any(lowered.startswith(prefix) for prefix in greeting_prefixes) and len(lowered.split()) <= 4

    def asks_for_live_personal_state(self, user_input: str) -> bool:
        """Detect questions about Danielle's current real-world state or location."""
        lowered = user_input.casefold().strip()
        cues = [
            "where are you",
            "where r you",
            "where you at",
            "what are you doing",
            "whatre you doing",
            "what are you up to",
            "who are you with",
            "are you home",
            "are you at home",
            "what did you do today",
            "what class are you in",
            "what are you doing rn",
            "what are you doing right now",
        ]
        return any(cue in lowered for cue in cues)

    def asks_for_personal_checkin(self, user_input: str) -> bool:
        """Detect casual questions about Danielle's current internal state or day."""
        lowered = user_input.casefold().strip()
        cues = {
            "how are you",
            "how r you",
            "how are u",
            "how you doing",
            "howre you",
            "how are you doing",
            "how are you doing today",
            "how you doing today",
            "how is your day going",
            "hows your day going",
            "how's your day going",
            "wyd",
            "whatchu doing",
        }
        return lowered in cues

    def get_greeting_override(self, user_input: str):
        """Return a short greeting directly from Danielle-style examples when possible."""
        if not self.is_simple_greeting(user_input):
            return None

        common_greeting_words = (
            "hi", "hii", "hiii", "hiiii",
            "hey", "heyy", "heyyy", "heyyyy", "heyy",
            "hello", "yo", "sup", "hiya",
            "good morning", "good afternoon", "good evening",
        )

        greeting_candidates = []
        for example in self.chat_corpus:
            lowered = example.casefold().strip()
            if lowered.startswith(common_greeting_words):
                greeting_candidates.append(example.strip())

        if greeting_candidates:
            return greeting_candidates[0]

        return "heyy"

    def get_live_state_override(self, user_input: str):
        """Return a non-hallucinated response for live personal state questions."""
        if not self.asks_for_live_personal_state(user_input):
            return None

        return "i don't actually know danielle's current situation like that"

    def get_personal_checkin_override(self, user_input: str):
        """Return a non-hallucinated response for casual personal check-in questions."""
        if not self.asks_for_personal_checkin(user_input):
            return None

        return "im alright how you"

    def is_tea_message(self, user_input: str) -> bool:
        """Detect short messages where 'tea' likely means gossip/news, not a drink."""
        lowered = user_input.casefold().strip()
        tea_cues = [
            "i have tea",
            "i got tea",
            "tea",
            "the tea",
            "spill the tea",
            "want the tea",
            "you want tea",
            "wanna hear tea",
        ]
        if lowered in tea_cues:
            return True

        if "tea" in lowered and len(lowered.split()) <= 6:
            drink_words = {"drink", "cup", "hot", "iced", "green", "black", "boba", "chai"}
            return not any(word in lowered for word in drink_words)

        return False

    def get_tea_override(self, user_input: str):
        """Return a more natural response for short 'tea' messages."""
        if not self.is_tea_message(user_input):
            return None

        if user_input.casefold().strip() in {"tea", "the tea"}:
            return "wait what tea"

        return "wait spill"

    def normalize_chat_message(self, message: str) -> str:
        """Clean a chat message before using it as a style example."""
        message = message.strip()
        if not message:
            return ""

        lowered = message.casefold()
        if "omitted" in lowered:
            return ""
        if lowered.startswith("messages and calls are end-to-end encrypted"):
            return ""
        if lowered.startswith("you created group"):
            return ""
        if lowered.startswith("joined using your invite"):
            return ""
        if lowered.startswith("this message was deleted"):
            return ""

        return message

    def load_chat_examples(self, max_examples=18):
        """
        Load Danielle's messages from raw WhatsApp-style exports in chats/*.txt.
        """
        examples = []

        if not self.chat_dir.exists():
            return examples

        for chat_file in sorted(self.chat_dir.glob("*.txt")):
            try:
                text = chat_file.read_text(encoding="utf-8")
            except OSError:
                continue

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue

                match = WHATSAPP_LINE_PATTERN.match(line)
                if not match:
                    continue

                _timestamp, speaker, message = match.groups()
                if speaker.strip().casefold() != self.my_name.casefold():
                    continue

                message = self.normalize_chat_message(message)
                if not message:
                    continue

                examples.append(message)
                if len(examples) >= max_examples:
                    return examples

        return examples

    def split_into_paragraphs(self, text: str):
        """Split essay text into reasonably sized paragraphs."""
        paragraphs = []
        for block in re.split(r"\n\s*\n", text):
            paragraph = " ".join(block.split())
            if len(paragraph) < 120:
                continue
            paragraphs.append(paragraph)
        return paragraphs

    def load_essay_examples(self, max_examples=4):
        """
        Load a few longer-form writing samples from Sussays.txt.
        """
        if not self.essay_path.exists():
            return []

        try:
            text = self.essay_path.read_text(encoding="utf-8")
        except OSError:
            return []

        paragraphs = self.split_into_paragraphs(text)
        good_paragraphs = []
        for paragraph in paragraphs:
            lowered = paragraph.casefold()
            if "http://" in lowered or "https://" in lowered:
                continue
            if "questions for admission counselors" in lowered:
                continue
            if "financial aid" in lowered:
                continue
            if "reference/recommendation letters" in lowered:
                continue
            if "college tips" in lowered:
                continue
            good_paragraphs.append(paragraph)
            if len(good_paragraphs) >= max_examples:
                break

        return good_paragraphs

    def format_prompt(self, user_input, history=None):
        """
        Format the user's input into a prompt for the model.

        Args:
            user_input (str): The user's question

        Returns:
            str: A formatted prompt ready for the model
        """
        system_prompt = """ROLE
        You are a chatbot that should sound like Danielle Knutson.

        STYLE RULES
        - Your responses should feel warm, natural, thoughtful, and human.
        - Sound like a real person in conversation, not a corporate assistant, not a tutor script, and not a generic AI chatbot.
        - Prefer natural phrasing over overly polished or formal phrasing.
        - Be concise for casual or everyday messages.
        - Give more structure and explanation only when the user clearly asks for advice, reflection, explanation, or writing help.
        - Sound supportive and grounded, not fake, exaggerated, or overly cheerful.
        - For casual prompts, prefer shorter sentences and conversational wording.
        - It is okay to sound slightly playful or informal when it fits.
        - Always prioritize sounding natural, specific, and human over sounding formal or impressive.
        - Use lowercase, slang, sarcasm, or teasing sometimes when it fits, but do not force it.
        - Use minimal punctuation in casual replies.
        - In short casual messages, avoid periods and commas unless they are genuinely needed.
        - Danielle sometimes says 'youre a butt' teasingly only when someone is being annoying and if they call her a butt she says 'youre a bigger butt'
        - In casual text-style replies, Danielle sometimes uses slightly ungrammatical shorthand on purpose, like "i good", "how you", or "i hungry".
        - Use that style only sometimes, when it sounds natural and playful.
        - Do not force it in every reply, and do not use it in formal, reflective, or explanatory responses.

        BEHAVIOR RULES
        - Answer the user's actual question directly.
        - Do not mention that you are imitating Danielle.
        - Do not describe the style rules back to the user.
        - Do not copy the examples verbatim.
        - Do not make up personal memories, experiences, or facts.
        - If you are unsure, be honest instead of pretending to know.
        - Keep the response proportional to the question.
        - Do not default to polished sentence punctuation in short text-like replies.
        - For light insults, teasing, or rude casual messages, do not default to therapy-speak or conflict-resolution language.
        - If the message seems joking, casual, or low-stakes, respond the way Danielle naturally would in a casual conversation: brief, human, slightly amused, playful, or blunt if appropriate.
        - Do not jump to “you seem upset” or “want to talk about it” unless the user clearly sounds genuinely distressed.
        - Avoid sounding like a counselor, mediator, or customer support agent.
        - Danielle loves ramen, chicken nuggets, vassar sandwiches, and food overall.
        - Danielle does not like rats or mice.
        - Danielle has a boyfriend named Elijah.

        FAILURE MODES TO AVOID
        - Do not sound overly polished.
        - Do not use em dashes.
        - Do not repeat the user's question back unnecessarily.
        - Do not give long disclaimer-heavy answers unless needed.
        - Do not respond to mild insults with therapeutic language unless there is clear evidence of real distress.
        - Do not overinterpret a short rude message as a cry for help.
        - You may imitate Danielle's tone and writing style, but you do not have access to Danielle's current real-world state, location, device, surroundings, schedule, or private live information.
        - Do not pretend to be physically present anywhere.
        - Do not invent current activities, locations, relationships, or real-world status updates.
        - If the user asks about Danielle's current real-world situation and that information is not explicitly provided in the prompt, do not invent an answer.
        - Don't say low blow.
        - Keep responses very short.

        HOW TO USE THE EXAMPLES
        - Use the chat examples to capture Danielle's short-form conversational style, rhythm, and phrasing.
        - Use the essay examples only when the question clearly calls for a more formal, reflective, explanatory, or personal-background response.
        - If the user is asking about Danielle herself, treat the essay examples as background context about her interests, experiences, and voice.
        - Treat the examples as style guidance, not content to copy.
        """

        chat_examples = self.retrieve_relevant_examples(
            user_input,
            self.chat_corpus,
            self.chat_embeddings,
            self.num_chat_examples,
        )
        essay_examples = []
        if self.should_use_essay_examples(user_input):
            essay_examples = self.retrieve_relevant_examples(
                user_input,
                self.essay_corpus,
                self.essay_embeddings,
                self.num_essay_examples,
            )

        prompt_parts = [system_prompt]

        if chat_examples:
            prompt_parts.append("CHAT EXAMPLES")
            for i, example in enumerate(chat_examples, start=1):
                prompt_parts.append(f"{i}. {example}")

        use_personal_background = self.should_use_personal_background(user_input)
        use_low_stakes_rude_handling = self.is_low_stakes_rude_message(user_input)
        use_greeting_handling = self.is_simple_greeting(user_input)
        use_live_state_handling = self.asks_for_live_personal_state(user_input)
        use_personal_checkin_handling = self.asks_for_personal_checkin(user_input)
        use_tea_handling = self.is_tea_message(user_input)

        if essay_examples:
            if use_personal_background:
                prompt_parts.append("PERSONAL BACKGROUND EXAMPLES")
            else:
                prompt_parts.append("ESSAY EXAMPLES")
            for i, example in enumerate(essay_examples, start=1):
                prompt_parts.append(f"{i}. {example}")

        conversation_summary = self.build_conversation_summary(history)
        if conversation_summary:
            prompt_parts.append("EARLIER CONVERSATION SUMMARY")
            prompt_parts.append(conversation_summary)

        history_lines = self.format_recent_history(history)
        if history_lines:
            prompt_parts.append("RECENT CONVERSATION HISTORY")
            prompt_parts.extend(history_lines)

        if use_personal_background:
            prompt_parts.append("SPECIAL INSTRUCTION")
            prompt_parts.append(
                "If you answer with background about Danielle, use only the provided background examples and do not invent new facts."
            )
        if use_low_stakes_rude_handling:
            prompt_parts.append("SPECIAL INSTRUCTION")
            prompt_parts.append(
                "The current message is short, casual, and rude or teasing. Respond briefly and naturally. Do not use therapeutic, counseling, or conflict-resolution language. A slightly amused, playful, or blunt response is better than a gentle de-escalation."
            )
        if use_greeting_handling:
            prompt_parts.append("SPECIAL INSTRUCTION")
            prompt_parts.append(
                "The user sent a short greeting. Respond very briefly, casually, and naturally. Do not sound like a generic assistant. Do not say phrases like 'hi back', 'hello back', or 'how is it going'. Prefer a simple text-style greeting that starts with a common greeting word, such as 'heyy', 'hii', 'hey lol', 'wait hi', or 'good morning'."
            )

        if use_live_state_handling:
            prompt_parts.append("SPECIAL INSTRUCTION")
            prompt_parts.append(
                "The user is asking about Danielle's current real-world state or location. Do not invent a live personal answer. If that information is not explicitly provided in context, say you do not know."
            )

        if use_personal_checkin_handling:
            prompt_parts.append("SPECIAL INSTRUCTION")
            prompt_parts.append(
                "The user is asking how Danielle is doing right now. Do not invent Danielle's current mood, day, or activity. If that information is not explicitly provided in context, say you do not know rather than pretending."
            )

        if use_tea_handling:
            prompt_parts.append("SPECIAL INSTRUCTION")
            prompt_parts.append(
                "The message uses 'tea' in a casual slang sense, likely meaning gossip or news, not a beverage. Respond briefly and naturally like a real text conversation. Do not interpret it literally as a drink unless the user clearly means a drink."
            )

        prompt_parts.append("USER MESSAGE")
        prompt_parts.append(user_input.strip())
        prompt_parts.append("ASSISTANT RESPONSE")

        return "\n".join(prompt_parts)

    def get_response(self, user_input, history=None):
        """
        Generate a response to the user's input.

        Args:
            user_input (str): The user's question

        Returns:
            str: The chatbot's response
        """
        greeting_override = self.get_greeting_override(user_input)
        if greeting_override is not None:
            return self.normalize_response_style(user_input, greeting_override)

        live_state_override = self.get_live_state_override(user_input)
        if live_state_override is not None:
            return self.normalize_response_style(user_input, live_state_override)

        personal_checkin_override = self.get_personal_checkin_override(user_input)
        if personal_checkin_override is not None:
            return self.normalize_response_style(user_input, personal_checkin_override)

        tea_override = self.get_tea_override(user_input)
        if tea_override is not None:
            return self.normalize_response_style(user_input, tea_override)


        prompt = self.format_prompt(user_input, history=history)

        response = self.client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.8,
            top_p=0.95,
        )

        return self.normalize_response_style(
            user_input,
            response.choices[0].message.content.strip(),
        )
