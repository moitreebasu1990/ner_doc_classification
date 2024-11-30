import os
from typing import List

import pandas as pd
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException
from mistralai.models.chat_completion import ChatMessage
from tqdm import tqdm


def doc_labeler(api_key: str, list_tokens: List[List[str]]) -> List[int]:
    """
    This function classifies a list of text documents into categories using Mistral's model.

    Args:
        api_key (str): The API key for Mistral.
        list_tokens (List[List[str]]): A list of lists containing the tokens of the documents.

    Returns:
        doc_labels (List[int]): A list of integers representing the categories of the documents.

        Categories:
        0: World
        1: Sport
        2: Business
        3: Technology
        4: Other

    Raises:
        MistralAPIException: If there's an authentication error or API issue
    """
    if not api_key:
        raise MistralAPIException("API key is required")

    # Initializing the Mistral client
    client = MistralClient(api_key=api_key)

    # Initializing the list to store the document labels
    doc_labels = []

    # Iterating over the list of tokens
    for tokens in tqdm(list_tokens):
        # Joining the tokens to form a text document
        text = " ".join(tokens)

        # Creating a chat completion using Mistral
        messages = [
            ChatMessage(
                role="system",
                content="You are a document classifier. Respond only with a single number (just the number, nothing else) representing the category (World:0, Sport:1, Business:2, Technology:3, Other:4).",
            ),
            ChatMessage(role="user", content=f"Classify this text: {text}"),
        ]

        try:
            response = client.chat(
                model="mistral-tiny",
                messages=messages,
            )
        except MistralAPIException as e:
            print(f"Error: {e}")
            # Default to OTHER category if we can't parse the response
            label = 4
        else:
            try:
                # Extracting the label from the response
                label = int(response.choices[0].message.content.strip())
                # Validate the label is in the correct range
                if not 0 <= label <= 4:
                    label = 4  # Default to OTHER if out of range
            except (ValueError, IndexError):
                # Default to OTHER category if we can't parse the response
                label = 4

        # Printing the label and the text for debugging purposes
        # print(f"Got label {label} for the text: '{text}'")

        # Appending the label to the list of document labels
        doc_labels.append(label)

    # Returning the list of document labels
    return doc_labels


def load_or_create_doc_labels(
    tokens: List[List[str]], api_key: str, file_name: str = "data/doc_labels.csv"
) -> List[int]:
    """
    Load document labels from a CSV file if it exists, otherwise create them using the Mistral model.

    Args:
        tokens (List[List[str]]): List of token sequences.
        api_key (str): Mistral API key.
        file_name (str, optional): Path to the CSV file for storing/loading labels. Defaults to "data/doc_labels.csv".

    Returns:
        List[int]: Document labels.

    Raises:
        MistralAPIException: If there's an authentication error or API issue when creating new labels
    """
    if os.path.isfile(file_name):
        try:
            # If yes, read them from the file
            df = pd.read_csv(file_name)
            # Convert labels to integers and get them as a list
            doc_labels = df["label"].astype(int).tolist()
            print(f"Loaded labels from cache file: {file_name}")
            return doc_labels
        except Exception as e:
            print(f"Error reading cache file: {str(e)}")
            # If there's an error reading the file, proceed to create new labels

    # If no file exists or there was an error reading it, use the `doc_labeler` function
    doc_labels = doc_labeler(api_key, tokens)

    # Try to save the labels to a CSV file
    try:
        pd.DataFrame({"label": doc_labels}).to_csv(file_name, index=False)
        print(f"Saved labels to file: {file_name}")
    except Exception as e:
        print(f"Warning: Could not save labels to file: {str(e)}")

    return doc_labels
