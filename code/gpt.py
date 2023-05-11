from typing import List, Tuple, Union

from constants import GPT_PROMPT, ADVERB_EMBEDDING_SIZE

import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_convert_env_result_to_adverb_input(result_A: Tuple[float, float, float], result_B: Tuple[float, float, float]) -> Tuple[Union[List[Tuple[str, str]], None], str]:
    (disp_x_A, vel_x_A, max_height_A) = result_A
    (disp_x_B, vel_x_B, max_height_B) = result_B

    messages = [
        {
            "role": "system",
            "content": GPT_PROMPT
        },
        {
            "role": "user",
            "content": f"Describe the following event: ball_a had {disp_x_A} distance, a2 {vel_x_A}, and a3 {max_height_A}, and ball_b had {disp_x_B} distance, a2 {vel_x_B}, and a3 {max_height_B}"
        }
    ]

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        n=1,
        temperature=0.1,
        max_tokens=200
    )

    raw_output: str = res.choices[0].message.content

    quote_segments: List[str] = raw_output.rsplit("\"", ADVERB_EMBEDDING_SIZE * 2)

    if len(quote_segments) == ADVERB_EMBEDDING_SIZE * 2 + 1:
        pairs = [quote_segments[2 * i + 1] for i in range(ADVERB_EMBEDDING_SIZE)]
        adverb_input = [pair.strip().split(" ") for pair in pairs]

        return adverb_input, raw_output
    
    comma_segments: List[str] = raw_output.rsplit(",", ADVERB_EMBEDDING_SIZE - 1)

    if len(comma_segments) == ADVERB_EMBEDDING_SIZE:
        pair_1 = comma_segments[0].rsplit(" ", 2)[1:]
        pair_2 = comma_segments[1].strip().split(" ")
        pair_3 = comma_segments[2].strip().split(" ")[1:3]
        adverb_input = [pair_1, pair_2, pair_3]

        return adverb_input, raw_output

    return None, raw_output
