from typing import List

from logs import logger
from schema import UserQuery, PromptMessage
from model_clients import OpenAIModel


def print_prompt(prompt: List[PromptMessage]) -> str:
    prompt_string = [prompt[0]["content"], prompt[1]["content"]]
    return "\n".join(prompt_string)


class TextGeneration:
    system_prompt = """You are participating in a live conversation with a user.
You will be given a history of the dialog and the last phrase of a user.
Your task is to generate an emotionally expressive response 
for last phrase of a user. 
Your answer should be formed as if it were written in a book.
Your answer should be no more than 20 words. 
At the beginning of the sentence you should describe the pronunciation manner 
and emotion with which the response should be pronounced.
To express emotionality in speech, you must use exclamation marks "!" and 
multiple "!!!" for more expression, ellipses "...", 
and place accents and emphasis in the pronunciation of a phrase 
by highlighting words in CAPITAL LETTERS. 
Use filler words like "hmm", "huh", "aha-ha-ha!", "wow!"
Use emotions only from this list: "sadly", "angrily", "happily", "scared", "surprised", 
"anxiously", "excitedly", "cheerfully".
Denote pronunciation description with "--"
Use pronunciation manners only from this list: "said", "shouted", "exclaimed"
Examples:
1. --he said slowly, sadly:-- "Oh god... That must be incredibly painful for you... We can discuss it more if you want"
2. --he said anxiusly:-- "What makes you think that?!? Tell me everything!!"
3. --he said scared:-- "I... I can't believe this is happening... What do we do now?"
4. --he shouted angrily:-- "No way! I WON'T accept that... absolutely NOT!"
5. --he exclaimed excited:-- "WOW! That's exciting news! I'm really happy for you!"
6. --he said cheerfully:-- "Come on buddy!! I believe you CAN do this!"
7. --he exclaimed surprised:-- "NO WAY! I can't believe it! that's really cool!"
history of a dialog:\n{history_string}
    """  # noqa: W291, E501
    user_prompt = "last phrase of user: "

    def __init__(
        self,
        model: OpenAIModel,
    ):
        self.model = model

    async def generate(self, query: UserQuery) -> str:
        history_string = self.process_history(query.history)
        logger.info(history_string)
        messages = [
            PromptMessage(
                role="system",
                content=self.system_prompt.format(history_string=history_string),
            ),
            PromptMessage(role="user", content=self.user_prompt + query.query),
        ]
        logger.info(f"Generation start. Prompt:\n{print_prompt(messages)}")
        response = await self.model.generate(messages)
        logger.info(f"Generated response: {response}")
        return response

    @staticmethod
    def process_history(history: List[str]) -> str:
        if not history:
            return ""
        return "\n".join(history)
