
import asyncio
import edge_tts

async def text_to_speech(text: str):
    voice = "en-IN-NeerjaNeural"  # Indian English female voice

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%"
    )

    await communicate.save("output.mp3")
    print("Speech saved to output.mp3")


if __name__ == "__main__":
    text = (
        """
        Question: What performance indicators does the RAGAS library provide?
        Answer:
        For evaluating a RAG pipeline using the RAGAS library, we measure performance across three main pillars: Generation, Retrieval, and End-to-End correctness using the following core metrics:

First, Faithfulness: Measures whether the generated answer is derived strictly from the retrieved context without any hallucinations.

Second, Answer Relevancy: Checks how directly relevant the generated answer is to the user's input question.

Third, Context Precision: Evaluates whether the most relevant documents appear at the top ranks of the retrieved context.

Fourth, Context Recall: Measures whether the retrieval system successfully fetched all the necessary information required to answer the question.

Fifth, Context Relevancy: Assesses how much noise or irrelevant information is present in the retrieved context.

Sixth, Answer Correctness: Compares the generated answer against the ground truth for accuracy.

Seventh, Answer Similarity: Measures the semantic similarity between the generated answer and the reference answer.
        """
    )

    asyncio.run(text_to_speech(text))
