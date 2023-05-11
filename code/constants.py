ADVERB_EMBEDDING_SIZE = 3
INPUT_SIZE = 2
GPT_PROMPT = f"""
Consider the task of throwing a ball. The maximum distance is 600. The maximum height is 400. The maximum speed is 50.
You are tasked with describing how the outcome of two ball-throwing tasks are related using qualifier-adverb pairs such as "slightly faster" and "extremely higher". Do not ever return "further", "closer", "faster", "slower", "higher", or "lower" on its own without a qualifier before it.
For example, where each question mark represents a qualifier like "slightly", "moderately", or "extremely" which you must always include:
If ball_a had a1 distance, a2 velocity, and a3 max height, and ball_b had b1 distance, b2 velocity, and b3 max height, then ball_a went "? further", "? faster", and "? lower" than ball_b.
If ball_a had a1 distance, a2 velocity, and a3 max height and ball_b had b1 distance, b2 velocity, and b3 max height, then ball_a went "? closer", "? quicker", and "? above" than ball_b.
If ball_a had a1 distance, a2 velocity, and a3 max height and ball_b had b1 distance, b2 velocity, and b3 max height, then ball_a went "? further", "? slower", and "? higher" than ball_b.
If ball_a had a1 distance, a2 velocity, and a3 max height and ball_b had b1 distance, b2 velocity, and b3 max height, then ball_a went "? nearer", "? slower", and "? below" than ball_b."""