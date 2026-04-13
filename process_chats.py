import re
from pathlib import Path


MY_NAME = "Danielle Knutson"

# Matches lines like:
# [9/21/24, 10:36:15 PM] Elijah Rice: When Matthew sent that image so like 9:30
# [9/21/24, 10:36:23 PM] ~ Mide: Okok my thing lasted super long 😭
LINE_PATTERN = re.compile(r"^\[(.*?)\]\s+([^:]+):\s?(.*)$")

def replace_names_in_text(message: str, name_map: dict[str, str], my_name: str) -> str:
    """Replace participant names inside a message body, keeping `my_name` unchanged."""
    for original_name, replacement in sorted(name_map.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"\b{re.escape(original_name)}\b", re.IGNORECASE)
        message = pattern.sub(replacement, message)
    return message


def clean_whatsapp_export(text: str, my_name: str = MY_NAME, extra_names_to_replace=None) -> str:
    """
    Remove timestamps and replace all speaker names except `my_name`.

    - Keeps the message text.
    - Keeps your own name unchanged.
    - Replaces each other speaker with a stable label like "Person 1", "Person 2", etc.
    - Preserves continuation lines that belong to the previous message.
    """
    cleaned_lines = []
    name_map = {}
    next_person_id = 1

    extra_names_to_replace = list(extra_names_to_replace or [])
    normalized_my_name = my_name.casefold()

    for line in text.splitlines():
        match = LINE_PATTERN.match(line)
        if match:
            _timestamp, speaker, _message = match.groups()
            speaker = speaker.strip()
            normalized_speaker = speaker.casefold()

            if normalized_speaker != normalized_my_name and speaker not in name_map:
                name_map[speaker] = f"Person {next_person_id}"
                next_person_id += 1

    for extra_name in extra_names_to_replace:
        extra_name = extra_name.strip()
        if (
            extra_name
            and extra_name.casefold() != normalized_my_name
            and extra_name not in name_map
        ):
            name_map[extra_name] = f"Person {next_person_id}"
            next_person_id += 1

    for line in text.splitlines():
        match = LINE_PATTERN.match(line)
        if match:
            _timestamp, speaker, message = match.groups()
            speaker = speaker.strip()
            normalized_speaker = speaker.casefold()

            if normalized_speaker == normalized_my_name:
                label = my_name
            else:
                label = name_map[speaker]

            message = replace_names_in_text(message, name_map, my_name)
            cleaned_lines.append(f"{label}: {message}")
        else:
            # Continuation lines from multiline messages: keep text only.
            cleaned_lines.append(replace_names_in_text(line, name_map, my_name))

    return "\n".join(cleaned_lines)

if __name__ == "__main__":
#     sample = """[9/21/24, 10:36:15 PM] elijah rice: When matthew sent that image so like 9:30
# [9/21/24, 10:36:23 PM] ~ Mide: Okok my thing lasted super long 😭
# [9/21/24, 10:39:27 PM] ~ mide: How far did y'all get?
# [9/21/24, 10:39:34 PM] danielle knutson: we finished
# [9/21/24, 10:40:00 PM] Elijah Rice: Nice
# [9/21/24, 10:40:10 PM] Danielle Knutson: Elijah Rice told ~ MIDE already"""
#     extra_names = ["Matthew", "Elijah", "Mide", "David", "Nikitha"]
    # print(clean_whatsapp_export(sample, extra_names_to_replace=extra_names))
    for filename in ["chats/ayo_mide_chat.txt", "chats/grind_squad_chat.txt"]:
        raw_text = Path(filename).read_text(encoding="utf-8")
        cleaned = clean_whatsapp_export(
            raw_text,
            extra_names_to_replace=["Matthew", "Elijah", "Mide", "David", "Nikitha"],
        )
        output_name = filename.replace(".txt", "_cleaned.txt")
        Path(output_name).write_text(cleaned, encoding="utf-8")
