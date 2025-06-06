import asyncio
import logging
import random
import re
from asyncio import sleep

import aiofiles
import flatlatex
import ujson
from better_profanity import profanity

logger = logging.getLogger(__name__)


# https://stackoverflow.com/questions/70640701/python-logger-limit-string-size
class NotTooLongStringFormatter(logging.Formatter):
    """
    A logging formatter that truncates long messages.

    It replaces newlines with '|n' for a single-line preview in logs
    and truncates the message if it exceeds a specified maximum length,
    appending '...' to indicate truncation.

    Attributes:
        DEFAULT_MAX_LENGTH (int): The default maximum length for log messages.
        max_length (int): The maximum length for log messages for this instance.
    """
    DEFAULT_MAX_LENGTH = 1000 # Increased default, and made it a class attribute

    def __init__(self, max_length: int | None = None, *args, **kwargs):
        """
        Initializes the NotTooLongStringFormatter.

        Args:
            max_length (int | None, optional): The maximum length for log messages.
                If None, DEFAULT_MAX_LENGTH is used. Defaults to None.
            *args: Additional arguments for the base Formatter class.
            **kwargs: Additional keyword arguments for the base Formatter class.
        """
        super().__init__(*args, **kwargs) # Pass through other formatter args
        self.max_length = max_length if max_length is not None else self.DEFAULT_MAX_LENGTH

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record, truncating the message if necessary.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message.
        """
        # Ensure record.msg is a string before processing
        original_msg = str(getattr(record, 'msg', ''))
        
        # Replace newlines for single-line log preview, but do it on a copy
        # This formatted_msg is only for the length check logic.
        formatted_msg_for_length_check = original_msg.replace("\n", "|n")
        
        if len(formatted_msg_for_length_check) > self.max_length: # Simplified length check
            # Truncate the original message for the actual log output
            record.msg = original_msg[:self.max_length - 3] + "..."
        else:
            # If not truncating, use the original message
            record.msg = original_msg

        # Important: The above modifications to record.msg are for the Formatter's internal processing.
        # If the original record.msg should remain unchanged for other handlers,
        # a copy of the record should be made or the formatted string returned directly.
        # For typical logging, modifying record.msg before super().format() is standard.
        return super().format(record)


async def smart_text_splitter(text: str, max_chunk_size: int = 2000) -> list[str]:
    """
    Splits a string into chunks of a maximum size, trying to preserve words.

    Words longer than max_chunk_size will be split.

    Args:
        text (str): The text to split.
        max_chunk_size (int, optional): The maximum size of each chunk.
            Defaults to 2000.

    Returns:
        list[str]: A list of text chunks.

    Doctests:
    >>> import asyncio
    >>> asyncio.run(smart_text_splitter("This is a test sentence.", 10))
    ['This is a', 'test', 'sentence.']
    >>> asyncio.run(smart_text_splitter("Short", 10))
    ['Short']
    >>> asyncio.run(smart_text_splitter("", 10))
    []
    >>> asyncio.run(smart_text_splitter("ThisIsAVeryLongWordThatExceedsTheChunkSize", 10))
    ['ThisIsAVer', 'yLongWordT', 'hatExceeds', 'TheChunkSi', 'ze']
    >>> asyncio.run(smart_text_splitter("Word WordVeryLong Word", 10))
    ['Word', 'WordVeryLo', 'ng Word']
    >>> asyncio.run(smart_text_splitter("Test sentence with multiple spaces   like this.", 20))
    ['Test sentence with', 'multiple spaces like', 'this.']
    >>> asyncio.run(smart_text_splitter(" leading space", 10))
    ['leading', 'space']
    >>> asyncio.run(smart_text_splitter("trailing space ", 10))
    ['trailing', 'space']
    """
    if not text:
        return []

    # Normalize spaces: split by space and then rejoin with single spaces.
    # This handles multiple spaces between words and leading/trailing spaces effectively.
    words = [word for word in text.split(" ") if word] # Filter out empty strings from multiple spaces

    if not words: # If text was only spaces or empty
        return []

    chunks = []
    current_chunk = ""

    for word in words:
        # Handle words longer than max_chunk_size
        if len(word) > max_chunk_size:
            # If current_chunk is not empty, add it first
            if current_chunk:
                chunks.append(current_chunk) # No strip needed as current_chunk is built carefully
                current_chunk = ""
            
            # Split the long word itself
            for i in range(0, len(word), max_chunk_size):
                chunks.append(word[i:i + max_chunk_size])
            continue

        # Check if adding the new word (plus a space) exceeds the limit
        if len(current_chunk) + len(word) + (1 if current_chunk else 0) > max_chunk_size:
            if current_chunk: # Add the current_chunk if it has content
                chunks.append(current_chunk)
            current_chunk = word # Start new chunk with the current word
        else:
            if current_chunk: # Add space if current_chunk is not empty
                current_chunk += " " + word
            else:
                current_chunk = word # Start new chunk
    
    # Add the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Filter out any potential empty strings that might have been added (should be less likely now)
    return [chunk for chunk in chunks if chunk]


async def remove_latex(text: str) -> str:
    """
    Removes LaTeX from a string and converts it to Unicode, wrapping it in asterisks.

    Handles inline LaTeX ($...$) and display LaTeX ($$...$$ or \[...\]).
    Invalid LaTeX will be returned with an error marker.

    Args:
        text (str): The text containing LaTeX.

    Returns:
        str: The text with LaTeX converted to Unicode and styled.

    Doctests:
    >>> import asyncio
    >>> asyncio.run(remove_latex("This is $E=mc^2$ inline."))
    'This is *E = mc²* inline.'
    >>> asyncio.run(remove_latex("Display math: \\[\\sum_{i=1}^n i = \\frac{n(n+1)}{2}\\]"))
    'Display math: \\n*Σᵢ₌₁ⁿ i = (n(n + 1))/2*\\n'
    >>> asyncio.run(remove_latex("No LaTeX here."))
    'No LaTeX here.'
    >>> asyncio.run(remove_latex("$a^2 + b^2 = c^2$"))
    '*a² + b² = c²*'
    >>> asyncio.run(remove_latex("$$x^2$$"))
    '\\n*x²*\\n'
    >>> asyncio.run(remove_latex("Text with $$\\alpha$$ and also $$\\beta$$ display blocks."))
    'Text with \\n*α*\\n and also \\n*β*\\n display blocks.'
    >>> asyncio.run(remove_latex("Text with \\(x\\) and also \\(y\\) inline blocks."))
    'Text with *x* and also *y* inline blocks.'
    >>> asyncio.run(remove_latex("Invalid LaTeX: $\\invalidcommand$"))
    'Invalid LaTeX: [LaTeX Error: \\\\invalidcommand]'
    >>> asyncio.run(remove_latex("An empty inline $ $ equation.")) # Should effectively remove it or leave minimal space
    'An empty inline  equation.'
    >>> asyncio.run(remove_latex("An empty display $$ $$ equation."))
    'An empty display \\n\\n equation.'
    >>> asyncio.run(remove_latex("$")) # Single delimiter
    '$'
    >>> asyncio.run(remove_latex("$$")) # Only display delimiters
    '\\n*DISPLAY\\_MATH\\_MARKER*\\n'
    >>> asyncio.run(remove_latex("Text $latex$ text"))
    'Text *latex* text'
    >>> asyncio.run(remove_latex("$latex$ text"))
    '*latex* text'
    >>> asyncio.run(remove_latex("Text $latex$"))
    'Text *latex*'
    """
    if not text:
        return ""

    # Standardize LaTeX delimiters
    text = text.replace(r"\(", "$")
    text = text.replace(r"\)", "$")
    text = text.replace(r"\[", " $$ ") # Add spaces to help isolate $$
    text = text.replace(r"\]", " $$ ")

    # Further normalize to ensure single $ for splitting, but mark display math
    text_for_splitting = text.replace("$$", "\n$DISPLAY_MATH_MARKER$\n")
    
    parts = text_for_splitting.split('$')
    # If only one part and no display marker, it means no '$' was found or only '$$' which became a marker.
    # If text was just "$$", parts would be ['', 'DISPLAY_MATH_MARKER', '\n']
    if len(parts) == 1 and "DISPLAY_MATH_MARKER" not in parts[0]:
        return text # Return original if no actual $ delimiters were processed to split

    c = flatlatex.converter()
    result_parts = []
    
    is_latex_segment = text_for_splitting.startswith('$')
    
    for i, segment in enumerate(parts):
        # This logic handles cases where split() creates empty strings
        # e.g., "$latex$".split('$') -> ['', 'latex', '']
        # or "text$latex$".split('$') -> ['text', 'latex', '']
        # or "$$latex$$".split('$') -> ['\n', 'DISPLAY_MATH_MARKER', '\nlatex\n', 'DISPLAY_MATH_MARKER', '\n'] (incorrect example, see below)
        # Correct for "$$latex$$" -> text_for_splitting = "\n$DISPLAY_MATH_MARKER$\nlatex\n$DISPLAY_MATH_MARKER$\n"
        # parts = ['', 'DISPLAY_MATH_MARKER', '\nlatex\n', 'DISPLAY_MATH_MARKER', '\n']

        if is_latex_segment:
            is_display_math = "DISPLAY_MATH_MARKER" in segment
            actual_latex_content = segment.replace("DISPLAY_MATH_MARKER", "").strip()

            if actual_latex_content: # If there's content to convert
                try:
                    converted_text = c.convert(actual_latex_content)
                    safe_converted_text = converted_text.replace("*", r"\*")
                    final_segment_text = "*" + safe_converted_text + "*"
                    if is_display_math:
                         final_segment_text = "\n" + final_segment_text + "\n"
                    result_parts.append(final_segment_text)
                except Exception as e:
                    logger.error(f"Failed to convert LaTeX: '{actual_latex_content}'. Error: {e}")
                    result_parts.append(f"[LaTeX Error: {actual_latex_content}]")
            elif is_display_math: # Empty display math block e.g. $$ $$ or marked by DISPLAY_MATH_MARKER
                result_parts.append("\n\n")
            # else: empty inline math e.g., $ $ - this segment would be empty or just whitespace.
            # If it was just whitespace, it gets skipped. If it was empty, it's fine.
        else: # This is a plain text segment
            # Restore display marker if it was part of a text segment (shouldn't happen with current logic)
            # but important if segment could contain "DISPLAY_MATH_MARKER" as plain text.
            result_parts.append(segment.replace("DISPLAY_MATH_MARKER", "$$"))

        is_latex_segment = not is_latex_segment

    return "".join(result_parts)


async def model_text_replace(text: str, replace_str: str) -> str:
    """
    Replaces substrings in text based on a comma-separated replacement string.

    The replacement string should have an even number of elements,
    representing find1,replace1,find2,replace2,...

    Args:
        text (str): The text to perform replacements on.
        replace_str (str): A comma-separated string of find/replace pairs.

    Returns:
        str: The text with replacements made.

    Doctests:
    >>> import asyncio
    >>> asyncio.run(model_text_replace("Hello world", "world,there"))
    'Hello there'
    >>> asyncio.run(model_text_replace("apple banana apple", "apple,orange,banana,grape"))
    'orange grape orange'
    >>> asyncio.run(model_text_replace("test", ""))
    'test'
    >>> asyncio.run(model_text_replace("test", "odd,number,of,elements")) # Odd number, no replacement
    'test'
    >>> asyncio.run(model_text_replace("find me", "find,replace,me,too"))
    'replace too'
    """
    logger.info(f"Attempting to replace text from model output. Input snippet: {text[:100]}...")
    
    if not replace_str:
        return text
        
    replace_list = replace_str.split(",")

    if len(replace_list) % 2 != 0:
        logger.warning(f"Replacement string '{replace_str}' has an odd number of elements. Skipping replacements.")
        return text

    for i in range(0, len(replace_list), 2):
        # Ensure there's a pair to replace (already guaranteed by even length check, but good practice)
        if i + 1 < len(replace_list):
            text_to_find = replace_list[i]
            text_to_replace_with = replace_list[i+1]
            text = text.replace(text_to_find, text_to_replace_with)
        # No 'else' needed because the length check ensures pairs.
    return text


async def clear_text(string: str) -> str:
    """
    Cleans a string by removing HTML-like comments and censoring profanity.

    HTML comments (<!--- ... -->) are removed.
    Profane words are wrapped in Discord spoiler tags (||word||).
    If the string becomes empty after cleaning, a zero-width space is returned.

    Args:
        string (str): The string to clean.

    Returns:
        str: The cleaned string.

    Doctests:
    >>> import asyncio
    >>> # profanity.load_censor_words() # Ensure profanity words are loaded for doctests if not globally done
    >>> asyncio.run(clear_text("Hello <!--- comment ---> world"))
    'Hello  world'
    >>> asyncio.run(clear_text("This is a damn test.")) # Assuming 'damn' is a profane word
    'This is a ||damn|| test.'
    >>> asyncio.run(clear_text("Clean string."))
    'Clean string.'
    >>> asyncio.run(clear_text("<!--- entire string is a comment --->"))
    '\\u200e'
    >>> asyncio.run(clear_text("  leading and trailing spaces  "))
    'leading and trailing spaces'
    >>> asyncio.run(clear_text(""))
    '\\u200e'
    """
    logger.info(f"Cleaning text. Input snippet: {string[:100]}...")
    
    # Remove HTML-like comments using a non-greedy match
    cleaned_string = re.sub(r"<!---.*?-->", "", string, flags=re.DOTALL)
    
    # Process profanity
    # Split by space, process, then rejoin. This handles multiple spaces better.
    words = cleaned_string.split(" ")
    processed_words = []
    for word in words:
        # Check non-empty word to avoid issues with multiple spaces creating empty strings in `words`
        if word and profanity.contains_profanity(word.strip("||").strip()):
            processed_words.append(f'||{word.strip().strip("|")}||')
        else:
            processed_words.append(word) # Append even if empty to preserve space structure if needed
    
    cleaned_string = " ".join(processed_words) # Rejoin
    
    # Strip leading/trailing whitespace that might result from comment removal or original string
    cleaned_string = cleaned_string.strip()
    
    # If the string becomes empty after cleaning, add a zero-width space.
    if not cleaned_string:
        cleaned_string = "\u200E"  # Zero-width space (LEFT-TO-RIGHT MARK)
        
    return cleaned_string


PERSONALITY_FILE_PATH = "./config/personality.json"

# Module-level variable to store the personalities
# Type hint for a single personality dictionary
PersonalityDict = dict[str, str | list[dict[str, str]] | bool | int | float] # More general type
PersonalitiesTuple = tuple[PersonalityDict, ...]

# Initialize personalities as None to indicate it hasn't been loaded/set.
personalities: PersonalitiesTuple | None = None

def non_async_get_personalties() -> PersonalitiesTuple:
    """
    Synchronously loads personalities from the personality JSON file.

    Reads the personality configuration file, parses the JSON, and returns
    the 'systems' list as a tuple of personality dictionaries.
    Logs errors if the file is not found, improperly formatted, or if
    other issues occur during reading.

    Returns:
        PersonalitiesTuple: A tuple of personality dictionaries.
                            Returns an empty tuple if loading fails.
    """
    try:
        with open(PERSONALITY_FILE_PATH, "r", encoding="utf-8") as file:
            data = ujson.load(file) # ujson.load directly from file object
            systems = data.get("systems", [])
            if not isinstance(systems, list):
                logger.error(f"Invalid format in {PERSONALITY_FILE_PATH}: 'systems' is not a list.")
                return ()
            # Further validation could be added here to check structure of each dict in systems
            # Ensure all items in systems are dictionaries
            if not all(isinstance(item, dict) for item in systems):
                logger.error(f"Invalid format in {PERSONALITY_FILE_PATH}: Not all items in 'systems' are dictionaries.")
                return ()
            return tuple(systems)
    except FileNotFoundError:
        logger.error(f"Personality file not found: {PERSONALITY_FILE_PATH}")
        return ()
    except (ujson.JSONDecodeError, KeyError) as e:
        logger.error(f"Error decoding JSON or key error in {PERSONALITY_FILE_PATH}: {e}")
        return ()
    except Exception as e:
        logger.error(f"Unexpected error reading personalities: {e}")
        return ()

async def get_personalties() -> PersonalitiesTuple:
    """
    Asynchronously loads personalities from the personality JSON file.

    Reads the personality configuration file asynchronously, parses the JSON,
    and returns the 'systems' list as a tuple of personality dictionaries.
    Logs errors if the file is not found, improperly formatted, or if
    other issues occur during reading.

    Returns:
        PersonalitiesTuple: A tuple of personality dictionaries.
                            Returns an empty tuple if loading fails.
    """
    try:
        async with aiofiles.open(PERSONALITY_FILE_PATH, "r", encoding="utf-8") as file:
            content = await file.read()
            data = ujson.loads(content)
            systems = data.get("systems", [])
            if not isinstance(systems, list):
                logger.error(f"Invalid format in {PERSONALITY_FILE_PATH}: 'systems' is not a list.")
                return ()
            # Ensure all items in systems are dictionaries
            if not all(isinstance(item, dict) for item in systems):
                logger.error(f"Invalid format in {PERSONALITY_FILE_PATH}: Not all items in 'systems' are dictionaries.")
                return ()
            return tuple(systems)
    except FileNotFoundError:
        logger.error(f"Personality file not found: {PERSONALITY_FILE_PATH}")
        return ()
    except (ujson.JSONDecodeError, KeyError) as e:
        logger.error(f"Error decoding JSON or key error in {PERSONALITY_FILE_PATH}: {e}")
        return ()
    except Exception as e:
        logger.error(f"Unexpected error reading personalities async: {e}")
        return ()

async def update_personality(k: int = 6) -> PersonalitiesTuple:
    """
    Updates the global 'personalities' tuple.

    If it's the first time or the list is empty, it samples 'k' personalities
    from the available ones. Otherwise, it keeps 'k-1' from the current
    list and adds one new random personality, trying to avoid immediate
    repetition.

    Args:
        k (int, optional): The desired number of personalities in the active set.
                           Defaults to 6.

    Returns:
        PersonalitiesTuple: The updated tuple of active personalities.
                            Returns an empty tuple if no personalities are available
                            or if issues occur.
    """
    global personalities # Declare that we are modifying the module-level 'personalities'
    
    available_personalities = await get_personalties()
    if not available_personalities:
        logger.warning("No personalities available from file to update the current set.")
        if personalities is None: # If never initialized and file is empty/corrupt
             personalities = ()
        return personalities # Return current (possibly empty) if no new ones to choose from

    if personalities is None or not personalities: # First time loading or current list is empty
        num_to_sample = min(k, len(available_personalities))
        if num_to_sample == 0 and len(available_personalities) > 0:
            personalities = (random.choice(available_personalities),)
        elif num_to_sample > 0 :
            personalities = tuple(random.sample(available_personalities, num_to_sample))
        else:
            personalities = ()
    else:
        current_list = list(personalities)
        
        new_selection = current_list[1:]
        
        if k > 0 :
            choices_for_new = [p for p in available_personalities if p not in new_selection]
            if not choices_for_new:
                choices_for_new = available_personalities
            
            if choices_for_new:
                 new_random_personality = random.choice(choices_for_new)
                 if len(new_selection) < k:
                    new_selection.append(new_random_personality)
                 elif not new_selection and k==1:
                    new_selection = [new_random_personality]

        personalities = tuple(new_selection)

    if personalities:
        first_personality_name = personalities[0].get('user_name', 'Unknown')
        logger.info(f"Updated personalities. Current primary: {first_personality_name}. Count: {len(personalities)}")
    else:
        logger.info("Personalities list is empty after update attempt.")
    
    return personalities


async def update_personality_wrapper(ttl: int = 3600) -> None:
    """
    Periodically updates the global personalities.

    This function runs in an infinite loop, calling `update_personality`
    and then sleeping for `ttl` seconds. It's intended to be run as an
    asyncio task.

    Args:
        ttl (int, optional): Time-to-live in seconds before the next update.
                             Defaults to 3600 (1 hour).
    """
    while True:
        await update_personality()
        await sleep(ttl)
    # This function runs indefinitely, so 'return' is effectively unreachable in normal operation.

async def get_personality() -> PersonalitiesTuple:
    """
    Retrieves the current set of active personalities.

    If personalities have not been loaded yet (i.e., global `personalities`
    is None), it triggers an initial load via `update_personality`.

    Returns:
        PersonalitiesTuple: The current tuple of active personalities.
                            Returns an empty tuple if loading failed or none are set.
    """
    global personalities
    if personalities is None:
        logger.info("Personalities not yet loaded, initializing...")
        await update_personality()
        # update_personality sets the global 'personalities'
    
    return personalities if personalities is not None else ()


async def start_personality() -> None:
    """
    Initializes and starts the personality management system.

    Ensures personalities are loaded at startup by calling `get_personality`
    (which in turn calls `update_personality` if needed).
    Then, it creates a background asyncio task to periodically update
    the personalities using `update_personality_wrapper`.
    """
    await get_personality() # Ensures initial load
    
    asyncio.create_task(update_personality_wrapper())
    logger.info("Personality update wrapper task started.")
    return
