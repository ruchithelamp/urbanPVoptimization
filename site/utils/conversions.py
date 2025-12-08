# utils/conversions
# business logic, constants

# business logic: Concersion rules
# AZ: 0.5 meters per pixel, MI: 1 foot per pixels
# using feet
onefootinmeters = 3.28084

CITY_RULES = {
    "AZ": 0.5 * onefootinmeters
    , "MI": 1.0
}


def get_city_conversion(city_code): 
    """
    Different city's sat images seem to have varying altitude, 
    which causes different pixel : distance metric ratios
    
    :param city_code: city name, for which rule to use
    out: float value of ratio
    """
    if city_code not in CITY_RULES:
        raise ValueError(f"City Code {city_code} not found. ")
    return CITY_RULES[city_code]

