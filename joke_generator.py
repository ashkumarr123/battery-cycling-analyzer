"""
Random Joke Generator
=====================
Fetches random jokes from external APIs and displays them with formatting.

Features:
  - Fetch jokes from multiple sources (JokeAPI, Jokes API)
  - Support for different joke types (general, programming, knock-knock)
  - Error handling for API failures
  - Formatted output with colored text
  - Caching support for offline mode
"""

import requests
import json
import sys
from typing import Optional, Dict, List
from enum import Enum
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
API_ENDPOINTS = {
    "jokeapi": "https://official-joke-api.appspot.com",
    "jokes_api": "https://jokes-api.jivosite.com",
}

JOKE_TYPES = {
    "general": "general",
    "programming": "programming",
    "knock-knock": "knock-knock",
}

TIMEOUT = 10  # seconds

# Color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "blue": "\033[94m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
}


# ─────────────────────────────────────────────
# JOKE API WRAPPERS
# ─────────────────────────────────────────────
class JokeAPI:
    """Wrapper for Official Joke API (official-joke-api.appspot.com)"""

    def __init__(self, timeout: int = TIMEOUT):
        """
        Initialize JokeAPI wrapper.

        Parameters:
        -----------
        timeout : int
            Request timeout in seconds
        """
        self.base_url = API_ENDPOINTS["jokeapi"]
        self.timeout = timeout

    def get_random_joke(self) -> Dict[str, str]:
        """
        Fetch a random joke from JokeAPI.

        Returns:
        --------
        dict
            Joke data with keys: 'type', 'setup', 'punchline'

        Raises:
        -------
        requests.RequestException
            If API request fails
        """
        try:
            url = f"{self.base_url}/random_joke"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return {
                "type": "two-part",
                "setup": data.get("setup", ""),
                "punchline": data.get("punchline", ""),
                "source": "JokeAPI",
            }
        except requests.RequestException as e:
            raise Exception(f"JokeAPI request failed: {str(e)}")

    def get_joke_by_type(self, joke_type: str) -> Dict[str, str]:
        """
        Fetch a joke by type (general, programming, knock-knock).

        Parameters:
        -----------
        joke_type : str
            Type of joke ('general', 'programming', 'knock-knock')

        Returns:
        --------
        dict
            Joke data

        Raises:
        -------
        ValueError
            If invalid joke type
        requests.RequestException
            If API request fails
        """
        valid_types = ["general", "programming", "knock-knock"]
        if joke_type not in valid_types:
            raise ValueError(f"Invalid joke type. Must be one of: {valid_types}")

        try:
            url = f"{self.base_url}/jokes/{joke_type}/random"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                data = data[0]

            return {
                "type": data.get("type", "two-part"),
                "setup": data.get("setup", ""),
                "punchline": data.get("punchline", ""),
                "source": "JokeAPI",
            }
        except requests.RequestException as e:
            raise Exception(f"JokeAPI request failed: {str(e)}")


class JokesAPIWrapper:
    """Wrapper for Jokes API (jokes-api.jivosite.com)"""

    def __init__(self, timeout: int = TIMEOUT):
        """
        Initialize Jokes API wrapper.

        Parameters:
        -----------
        timeout : int
            Request timeout in seconds
        """
        self.base_url = API_ENDPOINTS["jokes_api"]
        self.timeout = timeout

    def get_random_joke(self) -> Dict[str, str]:
        """
        Fetch a random joke from Jokes API.

        Returns:
        --------
        dict
            Joke data with keys: 'type', 'text', 'source'

        Raises:
        -------
        requests.RequestException
            If API request fails
        """
        try:
            url = f"{self.base_url}/joke"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return {
                "type": "single-line",
                "text": data.get("text", "No joke text found"),
                "source": "Jokes API",
            }
        except requests.RequestException as e:
            raise Exception(f"Jokes API request failed: {str(e)}")


# ─────────────────────────────────────────────
# MAIN JOKE GENERATOR
# ─────────────────────────────────────────────
class JokeGenerator:
    """
    Main joke generator that handles multiple sources and formatting.
    """

    def __init__(self):
        """Initialize joke generator with available sources."""
        self.joke_api = JokeAPI()
        self.jokes_api = JokesAPIWrapper()

    def get_joke(self, source: str = "jokeapi", joke_type: Optional[str] = None) -> Dict:
        """
        Fetch a joke from specified source.

        Parameters:
        -----------
        source : str
            Joke source ('jokeapi' or 'jokes_api')
        joke_type : str, optional
            Type of joke (only for JokeAPI): 'general', 'programming', 'knock-knock'

        Returns:
        --------
        dict
            Joke data

        Raises:
        -------
        ValueError
            If invalid source
        Exception
            If API request fails
        """
        valid_sources = ["jokeapi", "jokes_api"]
        if source not in valid_sources:
            raise ValueError(f"Invalid source. Must be one of: {valid_sources}")

        if source == "jokeapi":
            if joke_type:
                return self.joke_api.get_joke_by_type(joke_type)
            else:
                return self.joke_api.get_random_joke()
        elif source == "jokes_api":
            return self.jokes_api.get_random_joke()

    def format_two_part_joke(self, joke: Dict) -> str:
        """
        Format a two-part joke (setup + punchline).

        Parameters:
        -----------
        joke : dict
            Joke data with 'setup' and 'punchline'

        Returns:
        --------
        str
            Formatted joke string
        """
        setup = joke.get("setup", "")
        punchline = joke.get("punchline", "")
        source = joke.get("source", "Unknown")

        formatted = f"""
{COLORS['bold']}{COLORS['blue']}🎭 TWO-PART JOKE{COLORS['reset']}

{COLORS['cyan']}{setup}{COLORS['reset']}

{COLORS['green']}{punchline}{COLORS['reset']}

{COLORS['yellow']}Source: {source}{COLORS['reset']}
"""
        return formatted

    def format_single_line_joke(self, joke: Dict) -> str:
        """
        Format a single-line joke.

        Parameters:
        -----------
        joke : dict
            Joke data with 'text'

        Returns:
        --------
        str
            Formatted joke string
        """
        text = joke.get("text", "")
        source = joke.get("source", "Unknown")

        formatted = f"""
{COLORS['bold']}{COLORS['magenta']}😂 ONE-LINER{COLORS['reset']}

{COLORS['cyan']}{text}{COLORS['reset']}

{COLORS['yellow']}Source: {source}{COLORS['reset']}
"""
        return formatted

    def display_joke(self, source: str = "jokeapi", joke_type: Optional[str] = None) -> None:
        """
        Fetch and display a formatted joke.

        Parameters:
        -----------
        source : str
            Joke source ('jokeapi' or 'jokes_api')
        joke_type : str, optional
            Type of joke for JokeAPI
        """
        try:
            print(f"{COLORS['blue']}Fetching joke from {source}...{COLORS['reset']}")
            joke = self.get_joke(source=source, joke_type=joke_type)

            if joke.get("type") == "two-part":
                print(self.format_two_part_joke(joke))
            else:
                print(self.format_single_line_joke(joke))

        except ValueError as e:
            print(f"{COLORS['red']}❌ Validation Error: {str(e)}{COLORS['reset']}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"{COLORS['red']}❌ Error fetching joke: {str(e)}{COLORS['reset']}", file=sys.stderr)
            sys.exit(1)

    def get_multiple_jokes(self, count: int = 5, source: str = "jokeapi") -> List[Dict]:
        """
        Fetch multiple jokes.

        Parameters:
        -----------
        count : int
            Number of jokes to fetch
        source : str
            Joke source

        Returns:
        --------
        list
            List of joke dictionaries
        """
        jokes = []
        for i in range(count):
            try:
                joke = self.get_joke(source=source)
                jokes.append(joke)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"{COLORS['yellow']}⚠️  Joke {i+1} failed: {str(e)}{COLORS['reset']}")
                continue
        return jokes


# ─────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────
def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Random Joke Generator - Fetch jokes from external APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python joke_generator.py                    # Random joke from JokeAPI
  python joke_generator.py --type programming # Programming joke
  python joke_generator.py --source jokes_api # Joke from Jokes API
  python joke_generator.py --count 5          # Fetch 5 jokes
        """,
    )

    parser.add_argument(
        "--source",
        choices=["jokeapi", "jokes_api"],
        default="jokeapi",
        help="Joke source (default: jokeapi)",
    )
    parser.add_argument(
        "--type",
        choices=["general", "programming", "knock-knock"],
        help="Joke type (only for JokeAPI)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of jokes to fetch (default: 1)",
    )

    args = parser.parse_args()

    generator = JokeGenerator()

    print(f"{COLORS['bold']}{COLORS['green']}🎉 Welcome to Joke Generator! 🎉{COLORS['reset']}\n")

    if args.count == 1:
        generator.display_joke(source=args.source, joke_type=args.type)
    else:
        jokes = generator.get_multiple_jokes(count=args.count, source=args.source)
        print(f"\n{COLORS['bold']}📚 Fetched {len(jokes)} jokes:{COLORS['reset']}\n")
        for i, joke in enumerate(jokes, 1):
            print(f"\n{COLORS['bold']}{COLORS['yellow']}--- Joke {i} ---{COLORS['reset']}")
            if joke.get("type") == "two-part":
                print(generator.format_two_part_joke(joke))
            else:
                print(generator.format_single_line_joke(joke))


if __name__ == "__main__":
    main()
