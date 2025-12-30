from transformers.utils import get_json_schema

tools = []

def get_weather(location: str):
    """
    You do not know the weather at any specific location.
    This tool gets the current weather at a specific location

    Args:
        location: the location for the weather
    Returns:
        temperature: the temp at that location in F
    """
    print("AAHH")

def get_tools():
    # this sucks, but I think you gotta add the tools manually
    tools.append(get_json_schema(get_weather))
    
    return tools