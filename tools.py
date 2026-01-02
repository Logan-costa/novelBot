from transformers.utils import get_json_schema
import json

# first parameter of each tool must be a string contained within the query
# each tool must return exactly one string
# just because I said so
# additionally, each tool must have exactly two parameters, becuase of how they are called in main

def get_weather(city: str, country: str):
    """
    This tool gets the current weather at a specific location

    Args:
        city: the city the weather wants
        country: the country
    Returns:
        temperature: the temp at that location in F
    """
    return "AAHH"

# returns   - list of tools as json object
#           - dictionary of function pointers
def get_tools():
    tools = []

    # this sucks, but I think you gotta add the tools manually
    tools.append(get_json_schema(get_weather))
    tools_dict = {
        "get_weather": get_weather, 
    }
    
    return tools, tools_dict
