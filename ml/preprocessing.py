def format_experiences(experiences):
    content = []
    for exp in experiences:
        content.append(
            f"{exp['title']}. {exp['description']}"
        )
    return " ".join(content)

def build_model_text(about_text, experiences):
    return about_text + " " + format_experiences(experiences)
