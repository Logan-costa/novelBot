from transformers.utils import get_json_schema

tools = []

def get_weather(city: str, country: str):
    """
    gets the current weather at a specific location

    Args:
        city: the city
        country: the country of the city
    Returns:
        temperature: the temp at that location in F
    """
    print("AAHH")

def get_tools():
    # this sucks, but I think you gotta add the tools manually
    tools.append(get_json_schema(get_weather))
    
    return tools